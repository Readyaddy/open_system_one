"""Experiment 7 model: the hybrid dual-path decision architecture.

See NOTES.md for the full rationale. This file implements, in order:

  1. Packed-sequence construction with the budget allocator (Sec 2.2) --
     `PackedSequenceBuilder`, pure-Python + tokenizer, no torch state.
  2. `[MASK]`-position vector injection (Sec 2.3) -- `Projector`, and the
     injection logic inside `HybridDecisionModel._embed_packed_sequence`.
  3. Context compression via learned codes (Sec 2.4).
  4. The decision head: one entry layer + a shared-weight recurrent block,
     looped K times, scored at every depth (Sec 2.5) -- `RecurrentDecisionBlock`.
  5. The MaxSim / late-interaction path, built but gated off by default
     via a learned scalar initialized at 0 (Sec 2.6) -- `MaxSimScorer`.

Three "modality" channels carry option information into a decision:
  - TEXT   : the option's (budget-truncated) text tokens inside the packed
             sequence, read via ordinary self-attention.
  - VECTOR : a separately-encoded, untruncated pooled option embedding,
             injected into the option's [MASK] position.
  - MAXSIM : token-level late interaction between context tokens and the
             option's own (untruncated, separately-encoded) tokens.
Each is independently controllable per example (for modality-dropout
training, see data.py) and independently ablatable at eval time (set the
corresponding *_scale to 0), which is what makes the Sec 7 cardinality
sweep a valid, single-run ablation instead of three separately-trained
models.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BACKBONE = "answerdotai/ModernBERT-large"

QUESTION_TYPES = ["choice", "bool", "score"]
QTYPE_IDX = {t: i for i, t in enumerate(QUESTION_TYPES)}

# Natural-language type prefix folded into the instructions text. A fresh
# special token per type was considered and rejected for the same reason
# exp3's [OPT]/[NOUL]/[LVL] markers were flagged as risky in its own
# NOTES.md: a new token starts with a randomly-initialized embedding and has
# to learn its role from scratch on a comparatively small corpus. Plain text
# costs nothing and the pretrained backbone already understands these words.
# The *mechanistically load-bearing* type signal is the learned type
# embedding added at every [MASK] position (see _embed_packed_sequence) --
# this text prefix is a redundant, free hint on top of it.
TYPE_PREFIX = {
    "choice": "[Question type: choice -- pick exactly one option.] ",
    "bool": "[Question type: yes/no.] ",
    "score": "[Question type: ordinal score -- pick the closest level.] ",
}


def get_tokenizer(backbone: str = BACKBONE):
    tok = AutoTokenizer.from_pretrained(backbone)
    for name in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id"):
        if getattr(tok, name) is None:
            raise ValueError(
                f"{backbone} tokenizer has no {name} -- required by the packed-sequence "
                f"builder (CLS/SEP framing) and the [MASK] injection mechanism (Sec 2.3)."
            )
    return tok


def _safe_n_heads(hidden: int, preferred: int = 8) -> int:
    """Largest divisor of `hidden` that's <= preferred. Real backbones
    (ModernBERT-large, hidden=1024) always get 8; this only matters for the
    tiny from-scratch configs used by smoke_test.py, where hidden might be
    e.g. 32 or 48."""
    for h in range(min(preferred, hidden), 0, -1):
        if hidden % h == 0:
            return h
    return 1


# --------------------------------------------------------------------------
# 1. Packed-sequence construction (pure Python, no torch) -- Sec 2.2
# --------------------------------------------------------------------------

@dataclass
class PackedExample:
    """One request, fully specified, pre-tokenization. Built by data.py's
    augmentation pipeline; consumed by PackedSequenceBuilder."""
    context: str
    instructions: str
    option_texts: List[str]           # rendered option strings, already
    # shuffled by data.py if order-shuffle augmentation is on.
    qtype: str                        # "choice" | "bool" | "score"
    answer_idx: int
    use_text: bool = True             # modality-dropout switches (Sec 4.1)
    use_vector: bool = True
    source: str = ""


@dataclass
class PackedBatch:
    input_ids: torch.Tensor            # (B, L)
    attention_mask: torch.Tensor       # (B, L)
    context_token_mask: torch.Tensor   # (B, L) bool -- True over context span
    mask_positions: torch.Tensor       # (B, Nmax) long, -1 = padding slot
    valid_mask: torch.Tensor           # (B, Nmax) bool -- True = real option
    inject_scale: torch.Tensor         # (B,) float in {0., 1.} -- VECTOR modality switch
    text_scale: torch.Tensor           # (B,) float in {0., 1.} -- diagnostic only;
    # TEXT modality is actually realized by the budget allocator giving 0
    # tokens to every option when use_text=False (data.py sets per_option
    # budget to 0), not by a runtime mask here -- kept as a field for eval
    # bookkeeping/logging, not consumed by the model.
    qtype_idx: torch.Tensor            # (B,) long
    answer_idx: torch.Tensor           # (B,) long
    option_full_texts: List[List[str]]  # per-example list of UNTRUNCATED
    # option strings -- feeds the separate weight-tied encoding pass used
    # for both vector injection and MaxSim (Sec 2.3, 2.6).


class PackedSequenceBuilder:
    """Budget allocator + sequence assembly (Sec 2.2). Holds no learnable
    state -- a plain tokenizer wrapper, safe to share across processes."""

    def __init__(self, tokenizer, budget_total: int = 2048, l_max_per_option: int = 64,
                 l_instructions: int = 96, l_context: int = 768, pad_multiple: int = 1):
        self.pad_multiple = pad_multiple
        self.tok = tokenizer
        self.budget_total = budget_total
        self.l_max_per_option = l_max_per_option
        self.l_instructions = l_instructions
        self.l_context = l_context

    def _ids(self, text: str, max_len: int) -> List[int]:
        if max_len <= 0 or not text:
            return []
        return self.tok(text, add_special_tokens=False, truncation=True,
                         max_length=max_len)["input_ids"]

    def per_option_budget(self, n_options: int, instr_len: int, ctx_len: int) -> int:
        """Sec 2.2's allocator: split whatever's left of the total budget,
        after instructions/context/overhead, evenly across N options. Floors
        at 0 (pure [MASK]-slot / vector-only regime), caps at l_max_per_option
        (no reason to keep growing an option's text budget once the whole
        option text likely already fits)."""
        overhead = 4 + n_options  # [CLS] + 2x[SEP] + 1 spare, + one [MASK] per option
        avail = self.budget_total - instr_len - ctx_len - overhead
        if n_options <= 0 or avail <= 0:
            return 0
        return max(0, min(self.l_max_per_option, avail // n_options))

    def build_one(self, ex: PackedExample):
        """Returns (input_ids: List[int], mask_positions: List[int],
        context_token_mask: List[bool]) for a single example, unpadded."""
        instr_text = TYPE_PREFIX[ex.qtype] + ex.instructions
        instr_ids = self._ids(instr_text, self.l_instructions)
        ctx_ids = self._ids(ex.context, self.l_context)

        n_opt = len(ex.option_texts)
        per_opt = self.per_option_budget(n_opt, len(instr_ids), len(ctx_ids)) if ex.use_text else 0

        ids: List[int] = [self.tok.cls_token_id] + instr_ids + [self.tok.sep_token_id]
        mask_positions: List[int] = []
        for opt_text in ex.option_texts:
            mask_positions.append(len(ids))
            ids.append(self.tok.mask_token_id)
            if per_opt > 0:
                ids.extend(self._ids(opt_text, per_opt))
        ids.append(self.tok.sep_token_id)

        ctx_start = len(ids)
        ids.extend(ctx_ids)
        ctx_end = len(ids)
        ids.append(self.tok.sep_token_id)

        context_token_mask = [False] * len(ids)
        for p in range(ctx_start, ctx_end):
            context_token_mask[p] = True

        return ids, mask_positions, context_token_mask

    def build_batch(self, examples: List[PackedExample], device) -> PackedBatch:
        built = [self.build_one(ex) for ex in examples]
        max_len = max(len(ids) for ids, _, _ in built)
        # Round up to a multiple of pad_multiple. With token-budget batching
        # every batch had a unique length, so the CUDA caching allocator could
        # rarely reuse a freed block and kept reserving new ones -- on Windows
        # that growth pages silently into system RAM instead of failing.
        # Fewer distinct shapes -> more block reuse.
        if self.pad_multiple > 1:
            max_len = -(-max_len // self.pad_multiple) * self.pad_multiple
        max_n = max(len(mp) for _, mp, _ in built)
        pad_id = self.tok.pad_token_id

        B = len(examples)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        context_token_mask = torch.zeros((B, max_len), dtype=torch.bool)
        mask_positions = torch.full((B, max_n), -1, dtype=torch.long)
        valid_mask = torch.zeros((B, max_n), dtype=torch.bool)

        for i, (ids, mp, ctm) in enumerate(built):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :L] = 1
            context_token_mask[i, :L] = torch.tensor(ctm, dtype=torch.bool)
            n = len(mp)
            mask_positions[i, :n] = torch.tensor(mp, dtype=torch.long)
            valid_mask[i, :n] = True

        inject_scale = torch.tensor([1.0 if ex.use_vector else 0.0 for ex in examples])
        text_scale = torch.tensor([1.0 if ex.use_text else 0.0 for ex in examples])
        qtype_idx = torch.tensor([QTYPE_IDX[ex.qtype] for ex in examples], dtype=torch.long)
        answer_idx = torch.tensor([ex.answer_idx for ex in examples], dtype=torch.long)
        option_full_texts = [ex.option_texts for ex in examples]

        return PackedBatch(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
            context_token_mask=context_token_mask.to(device), mask_positions=mask_positions.to(device),
            valid_mask=valid_mask.to(device), inject_scale=inject_scale.to(device),
            text_scale=text_scale.to(device), qtype_idx=qtype_idx.to(device),
            answer_idx=answer_idx.to(device), option_full_texts=option_full_texts,
        )


# --------------------------------------------------------------------------
# 2. Modules
# --------------------------------------------------------------------------

class Projector(nn.Module):
    """2-layer MLP mapping a pooled option embedding into the backbone's own
    input-embedding space (Sec 2.3). LLaVA-style -- a bare linear
    underperforms for this kind of cross-space mapping in every reported
    ablation of that pattern; not worth re-discovering the hard way here."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.net(x)


def load_flexible(model, state_dict):
    """Load a checkpoint into a model whose head may have grown since.

    Plain load_state_dict(strict=False) tolerates missing/unexpected keys but
    still RAISES on a shape mismatch, so resizing context_codes (16 -> 128)
    would crash on load. Handled here:
      * context_codes with a different count: the trained codes are tiled to
        fill the new size. Copies 2..n get small noise -- identical codes
        produce identical attention outputs and identical gradients, so
        without noise the copies would never diverge. The original codes are
        kept exact, so the first m slots are precisely the trained summary.
      * any other shape mismatch: dropped and reported, never silently kept.
      * newly added modules: initialized from trained weights via
        model.init_new_from_pretrained.
    Returns (missing, unexpected, notes).
    """
    sd = dict(state_dict)
    own = model.state_dict()
    notes = []

    k = "context_codes"
    if k in sd and k in own and sd[k].shape != own[k].shape and sd[k].shape[1] == own[k].shape[1]:
        old = sd.pop(k).to(own[k].dtype)
        m_old, m_new = old.shape[0], own[k].shape[0]
        reps = math.ceil(m_new / m_old)
        tiled = old.repeat(reps, 1)[:m_new].clone()
        noise = torch.randn_like(tiled) * old.std() * 0.1
        noise[:m_old] = 0
        with torch.no_grad():
            model.context_codes.copy_(tiled + noise)
        notes.append(f"context_codes: tiled {m_old} trained codes -> {m_new} (originals exact, copies +10% noise)")

    for key in list(sd):
        if key in own and sd[key].shape != own[key].shape:
            notes.append(f"DROPPED {key}: checkpoint {tuple(sd[key].shape)} vs model {tuple(own[key].shape)}")
            sd.pop(key)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "context_codes"]
    notes += model.init_new_from_pretrained(missing)
    return missing, unexpected, notes


class DepthLoRALinear(nn.Linear):
    """nn.Linear plus one low-rank delta per recurrent depth.

    The recurrent block shares its weights across every pass, so the only
    things telling pass 1 from pass 6 were an additive depth embedding and
    FiLM. This gives each pass its own rank-r correction to the FFN, which is
    where most of a transformer layer's per-token transformation happens --
    the "relaxed recursion" idea (Bae et al. 2024, layer-wise LoRA on a
    shared block): the base weights stay shared, each depth gets a cheap
    private adjustment. At rank 16 over 6 depths this is ~1M params per FFN
    matrix pair, against the 33.6M the shared block already has.

    Subclasses nn.Linear on purpose so `weight`/`bias` keep their names and
    existing checkpoints load unchanged -- only lora_A/lora_B are new keys.
    lora_B starts at zero, so every depth is exactly the pretrained layer
    until training moves it.
    """

    def __init__(self, in_features, out_features, bias=True, n_depths=6, rank=16):
        super().__init__(in_features, out_features, bias=bias)
        self.rank = rank
        self.lora_A = nn.Parameter(torch.empty(n_depths, rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(n_depths, out_features, rank))
        nn.init.normal_(self.lora_A, std=in_features ** -0.5)
        self.depth = 0

    @classmethod
    def from_linear(cls, lin: nn.Linear, n_depths, rank):
        new = cls(lin.in_features, lin.out_features, bias=lin.bias is not None,
                  n_depths=n_depths, rank=rank)
        with torch.no_grad():
            new.weight.copy_(lin.weight)
            if lin.bias is not None:
                new.bias.copy_(lin.bias)
        return new

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        d = min(self.depth, self.lora_A.size(0) - 1)
        return y + F.linear(F.linear(x, self.lora_A[d]), self.lora_B[d])


class AttentiveScorer(nn.Module):
    """Multi-head attention before the per-option scoring MLP.

    The old scorer was an MLP applied to each option slot independently, so at
    classification time an option never looked at the other options or at the
    evidence -- it could only be scored in isolation. For ProofWriter that is
    exactly the wrong shape: "True", "False" and "Unknown" are only meaningful
    relative to each other, and the deciding facts sit thousands of tokens
    away in the context.

    Each block does two pre-norm residual sublayers:
      1. option <-> option self-attention  (options compare)
      2. option -> evidence cross-attention (options re-read the context and
         the loop's scratchpad thoughts at full resolution)

    Output projections start at zero, so a freshly added scorer is exactly the
    old MLP-only scorer until trained. The q/k/v projections are copied from
    the trained recurrent block (see HybridDecisionModel.init_new_from_pretrained)
    rather than starting random.
    """

    def __init__(self, dim, n_heads, n_blocks=1, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            blk = nn.ModuleDict({
                "norm_self": nn.LayerNorm(dim),
                "self_attn": nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True),
                "norm_cross": nn.LayerNorm(dim),
                "cross_attn": nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True),
            })
            for a in ("self_attn", "cross_attn"):
                nn.init.zeros_(blk[a].out_proj.weight)
                nn.init.zeros_(blk[a].out_proj.bias)
            self.blocks.append(blk)

    def forward(self, opt, opt_pad, mem, mem_pad):
        for blk in self.blocks:
            h = blk["norm_self"](opt)
            a, _ = blk["self_attn"](h, h, h, key_padding_mask=opt_pad, need_weights=False)
            opt = opt + a
            h = blk["norm_cross"](opt)
            a, _ = blk["cross_attn"](h, mem, mem, key_padding_mask=mem_pad, need_weights=False)
            opt = opt + a
        return opt


class RecurrentDecisionBlock(nn.Module):
    """The looped part of the decision head (Sec 2.5 / exp7c revision). Two
    TransformerDecoderLayers, SHARED across every pass k=1..K -- this module
    is instantiated once and called repeatedly in HybridDecisionModel.forward,
    so its weights are identical at every depth by construction (there is no
    per-depth parameter anywhere in this class).

    exp7b (TransformerEncoderLayer, self-attention only over the compressed
    1+m+Nmax head sequence) showed a completely flat depth curve: K=1..6 all
    scored within noise of each other. The diagnosis: with only self-attention
    over its own (already-compressed) state, plus s0 re-injected as a residual
    every pass, the block is close to a contraction map with a fixed point --
    each pass can only re-mix information already present in s0, never pull in
    anything new, so there's no mechanistic reason more passes should help.

    Fix: give each layer a cross-attention sub-layer (TransformerDecoderLayer's
    `memory` argument) over the FULL backbone token sequence H, not just the
    compressed head_seq. Every recurrent pass can now re-read the original
    context/option tokens directly, so depth is no longer purely "re-mix a
    fixed summary" -- it's "re-mix summary, then re-query raw evidence."
    Combined with a per-depth embedding added to the query side (see
    HybridDecisionModel.forward) so the block can also tell which pass it's
    on, which plain weight-sharing across depths otherwise hides completely.
    """

    def __init__(self, dim: int, n_layers: int = 2, n_heads: Optional[int] = None,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        n_heads = n_heads or _safe_n_heads(dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim * ff_mult,
                                        dropout=dropout, activation="gelu", batch_first=True,
                                        norm_first=True)
            for _ in range(n_layers)
        ])

    def forward(self, x, memory, key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_key_padding_mask=key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask)
        return x


class HybridDecisionModel(nn.Module):
    """The full exp7 architecture. See NOTES.md Sec 2 for every design
    decision; this docstring only orients the forward pass.

    forward() runs:
      1. Weight-tied option encoding (independent per-option forward pass,
         full untruncated text) -> pooled vectors for injection, and
         per-token states for MaxSim.
      2. Packed-sequence encoding with vector injection at [MASK] positions.
      3. Context-code compression (m learned queries over context tokens).
      4. Entry layer -> K recurrent passes, scored at every depth (list of
         K logit tensors returned, for the depth loss -- Sec 3.2).
      5. Optional MaxSim contribution added to every depth's logits.
    """

    def __init__(self, backbone: str = BACKBONE, mask_token_id: int = None,
                 n_context_codes: int = 16, k_max: int = 6, head_n_layers: int = 2,
                 maxsim_dim: int = 128, dropout: float = 0.1, use_maxsim: bool = False,
                 gradient_checkpointing: bool = True, backbone_override: nn.Module = None,
                 n_scratch: int = 0, use_film_depth: bool = False,
                 depth_lora_rank: int = 0, n_scorer_blocks: int = 0):
        super().__init__()
        if mask_token_id is None:
            raise ValueError("mask_token_id is required -- pass tokenizer.mask_token_id.")
        # `backbone_override` lets smoke_test.py inject a from-scratch,
        # randomly-initialized tiny config (e.g. hidden_size=64) instead of
        # downloading the real ~395M-param pretrained checkpoint every time
        # the architecture itself is being checked, not the pretrained
        # weights. Real training/eval never passes this.
        self.backbone = backbone_override if backbone_override is not None else AutoModel.from_pretrained(backbone)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        hidden = self.backbone.config.hidden_size
        self.hidden = hidden
        self.mask_token_id = mask_token_id
        self.k_max = k_max
        self.use_maxsim = use_maxsim

        # --- vector injection (Sec 2.3) ---
        self.option_projector = Projector(hidden)
        self.type_embed = nn.Embedding(len(QUESTION_TYPES), hidden)
        # Scalar gate, initialized at 0.5 so vector path is active:
        self.inject_gate = nn.Parameter(torch.tensor(0.5))
        # emb_scale matches the projected vector's norm to the embedding
        # table's own typical token norm -- computed once at init, NOT
        # trained, so it doesn't drift into degenerate solutions (e.g.
        # scaling itself to zero to dodge a poorly-tuned gate). Registered
        # as a buffer so it moves with .to(device) and saves in state_dict.
        with torch.no_grad():
            emb_table = self.backbone.get_input_embeddings().weight
            emb_scale = emb_table.norm(dim=-1).mean().clone()
        self.register_buffer("emb_scale", emb_scale)

        # --- context compression (Sec 2.4) ---
        self.n_context_codes = n_context_codes
        self.context_codes = nn.Parameter(torch.randn(n_context_codes, hidden) * (hidden ** -0.5))
        self.code_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=_safe_n_heads(hidden),
                                                 batch_first=True, dropout=dropout)

        # --- decision head (Sec 2.5) ---
        self.entry_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=_safe_n_heads(hidden), dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.recurrent_block = RecurrentDecisionBlock(hidden, n_layers=head_n_layers, dropout=dropout)
        
        # Residual normalization & gated re-injection for recurrent loop
        self.s0_gate = nn.Parameter(torch.tensor(0.5))
        self.recurrent_norm = nn.LayerNorm(hidden)
        self.recurrent_norm_out = nn.LayerNorm(hidden)
        # exp7c: one learned vector per depth, added to the recurrent block's
        # query side at every pass. Plain weight-sharing across depths means
        # the block has NO way to tell pass 1 from pass 6 unless something
        # external marks it -- this is that mark. Small init (0.02 std) so it
        # starts as a near-negligible perturbation the model can grow into,
        # same rationale as inject_gate's small init.
        self.depth_embed = nn.Embedding(k_max, hidden)
        nn.init.normal_(self.depth_embed.weight, std=0.02)

        # --- per-depth FiLM modulation (use_film_depth) ---
        # depth_embed can only SHIFT the loop's input by a fixed vector. With
        # weights shared across every pass, that's the only thing making pass 6
        # differ from pass 1 -- one additive offset against a 1024-dim state.
        # A per-depth scale as well as a shift lets each pass apply a genuinely
        # different transformation, which is what actually breaks the
        # shared-weight symmetry that makes the loop contract to a fixed point.
        # Initialized scale=1 / shift=0, so switching this on is an exact no-op
        # until trained and existing checkpoints keep their behavior.
        self.use_film_depth = use_film_depth
        if use_film_depth:
            self.depth_scale = nn.Embedding(k_max, hidden)
            self.depth_shift = nn.Embedding(k_max, hidden)
            nn.init.ones_(self.depth_scale.weight)
            nn.init.zeros_(self.depth_shift.weight)

        # --- thought scratchpad (n_scratch > 0) ---
        # The loop's real gap: `s` is the ONLY thing carried between passes, so
        # pass 5 has no addressable record of what pass 2 concluded -- the outer
        # residual accumulates prior passes into one summed state, but nothing
        # can be attended to individually. Chain-of-thought works precisely
        # because step 3 can re-read steps 1-2.
        # Each pass now writes n_scratch latent vectors describing what it just
        # concluded; later passes cross-attend over [H ; all prior thoughts], so
        # the raw evidence AND the loop's own reasoning trace are both readable.
        # scratch_pos marks which pass a thought came from, so "what I thought
        # first" and "what I thought last" aren't interchangeable.
        self.n_scratch = n_scratch
        if n_scratch > 0:
            self.scratch_proj = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, n_scratch * hidden),
            )
            self.scratch_pos = nn.Embedding(k_max, hidden)
            nn.init.normal_(self.scratch_pos.weight, std=0.02)
            # Small but NOT zero. This gate multiplies scratch_proj's output, so
            # at exactly 0 the projection receives precisely zero gradient
            # (d(th)/d(proj) = gate = 0) and can never start learning -- the gate
            # itself only gets ~1e-4, so the bootstrap would take longer than any
            # run here. Verified empirically before changing it: at 0.0 all four
            # scratch_proj tensors sit at grad_sum == 0.0 after a full backward.
            # 0.1 keeps the initial perturbation small while letting both the
            # gate and the projection train from step one.
            self.scratch_gate = nn.Parameter(torch.tensor(0.1))

        # --- per-depth LoRA on the shared recurrent FFN (depth_lora_rank > 0) ---
        self.depth_lora_rank = depth_lora_rank
        self._lora_modules = []
        if depth_lora_rank > 0:
            for layer in self.recurrent_block.layers:
                layer.linear1 = DepthLoRALinear.from_linear(layer.linear1, k_max, depth_lora_rank)
                layer.linear2 = DepthLoRALinear.from_linear(layer.linear2, k_max, depth_lora_rank)
                self._lora_modules += [layer.linear1, layer.linear2]

        # --- attention before scoring (n_scorer_blocks > 0) ---
        self.n_scorer_blocks = n_scorer_blocks
        if n_scorer_blocks > 0:
            self.attn_scorer = AttentiveScorer(hidden, _safe_n_heads(hidden), n_scorer_blocks, dropout)

        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

        # --- MaxSim / late interaction (Sec 2.6), built but gated off ---
        self.late_proj = nn.Linear(hidden, maxsim_dim)
        self.maxsim_gate = nn.Parameter(torch.tensor(0.0))  # init 0: no effect until enabled+trained

    # ---- weight-tied option encoding (independent pass) ----

    def encode_options_raw(self, tokenizer, option_texts: List[str], device, max_length: int = 64,
                            chunk_size: int = 128, need_tokens: bool = True):
        """Runs the SAME backbone (weight-tied, not a second copy) over each
        option's full text, independently of any context. Returns:
          pooled:  (K, hidden)                  mean-pooled, for injection
          tokens:  (K, max_L, hidden) or None    per-token, for MaxSim only
          tok_mask:(K, max_L) bool  or None       real-token validity
        Chunked (same reasoning as exp6's encode_outcome_bank: a single-shot
        forward over hundreds of options is the OOM risk, not batch size).

        `need_tokens=False` (the caller passes `self.use_maxsim`) skips
        building and padding the per-token (K, max_L, hidden) tensors
        entirely -- these are ONLY consumed by MaxSim (Sec 2.6). Building
        them unconditionally was a real bug: at high N (this task samples up
        to 255 options per example) this tensor alone can be multiple GB,
        pure waste on every exp7a/7b run where MaxSim is off. Confirmed live
        -- this fix follows directly from an OOM crash at batch_size=64,
        option_chunk_size=384 that traced back to exactly this buffer.
        """
        pooled_chunks = []
        token_chunks, mask_chunks = ([], []) if need_tokens else (None, None)
        for i in range(0, len(option_texts), chunk_size):
            chunk = option_texts[i:i + chunk_size]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt")
            ids = enc["input_ids"].to(device)
            amask = enc["attention_mask"].to(device)
            out = self.backbone(input_ids=ids, attention_mask=amask)
            h = out.last_hidden_state  # (b, L, D)
            m = amask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            pooled_chunks.append(pooled)
            if need_tokens:
                token_chunks.append(h)
                mask_chunks.append(amask.bool())
        if not need_tokens:
            return torch.cat(pooled_chunks, dim=0), None, None
        # Pad token-level chunks to a common length before concatenating.
        max_L = max(t.size(1) for t in token_chunks)
        padded_tokens, padded_masks = [], []
        for h, m in zip(token_chunks, mask_chunks):
            if h.size(1) < max_L:
                pad_h = h.new_zeros(h.size(0), max_L - h.size(1), h.size(2))
                pad_m = m.new_zeros(m.size(0), max_L - m.size(1))
                h = torch.cat([h, pad_h], dim=1)
                m = torch.cat([m, pad_m], dim=1)
            padded_tokens.append(h)
            padded_masks.append(m)
        return (torch.cat(pooled_chunks, dim=0),
                torch.cat(padded_tokens, dim=0),
                torch.cat(padded_masks, dim=0))

    # ---- vector injection into the packed sequence ----

    def _embed_packed_sequence(self, batch: PackedBatch, option_pooled: torch.Tensor,
                                option_pooled_valid: torch.Tensor):
        """Builds inputs_embeds for the packed sequence, injecting a
        projected+scaled+gated option vector at each [MASK] position.

        Injecting BEFORE the backbone's embeddings-level norm/dropout
        (i.e. via `inputs_embeds=` to the top-level model, which routes
        through that norm the same as an ordinary input_ids forward would)
        keeps [MASK] positions on equal footing with every other token --
        see NOTES.md Sec 2.3 and this file's module docstring. ModernBERT's
        RoPE positional encoding is applied inside attention based on
        sequence position, not added at the embedding layer, so substituting
        embeddings here loses no positional information (a real advantage
        of this backbone choice for this technique, worth recording).

        option_pooled:       (B, Nmax, hidden) -- already projected/scaled/
                              gated by the caller (encode_and_project_options).
        option_pooled_valid: (B, Nmax) bool -- real vs. padding option slot.
        """
        word_embed = self.backbone.get_input_embeddings()
        embeds = word_embed(batch.input_ids)  # (B, L, D)
        B, L, D = embeds.shape

        type_vec = self.type_embed(batch.qtype_idx)  # (B, D)

        # Fully vectorized (no per-element Python loop -- an earlier draft
        # looped over B*Nmax in Python, which at N up to 255 would have been
        # a real GPU-utilization killer). `add` carries the gated/scaled
        # option vector plus the type embedding, zeroed at padding slots;
        # scatter_add_ writes it into a same-shaped-as-embeds zero buffer at
        # each option's [MASK] position, then that buffer is added onto the
        # ordinary token embeddings in one op.
        add = (option_pooled + type_vec.unsqueeze(1)) * option_pooled_valid.unsqueeze(-1).to(embeds.dtype)
        safe_pos = batch.mask_positions.clamp(min=0)  # padding slots alias to 0; their `add` row is
        # already zeroed above, so aliasing there contributes nothing (see scatter_add_ below).
        idx = safe_pos.unsqueeze(-1).expand(-1, -1, D)
        injection = embeds.new_zeros(B, L, D)
        injection.scatter_add_(1, idx, add.to(embeds.dtype))
        return embeds + injection

    def encode_and_project_options(self, tokenizer, batch: PackedBatch, device, chunk_size: int = 128):
        """Runs the weight-tied option encoder over every option in the
        batch (flattened, padded back into (B, Nmax, hidden)), projects,
        scales, and applies the per-example VECTOR-modality gate
        (batch.inject_scale). Also returns the raw per-token states for
        MaxSim -- but ONLY when self.use_maxsim is actually on; otherwise
        the (B, Nmax, L, hidden) buffer this would need is skipped
        entirely (see encode_options_raw's need_tokens docstring -- this
        was a real multi-GB waste at high option counts before the fix).
        """
        flat_texts, owner, per_owner_n = [], [], []
        for texts in batch.option_full_texts:
            per_owner_n.append(len(texts))
            for t in texts:
                flat_texts.append(t)
                owner.append(len(per_owner_n) - 1)

        pooled, tokens, tok_mask = self.encode_options_raw(tokenizer, flat_texts, device,
                                                            chunk_size=chunk_size,
                                                            need_tokens=self.use_maxsim)
        projected = self.option_projector(pooled)
        projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.emb_scale

        B = len(batch.option_full_texts)
        Nmax = batch.mask_positions.size(1)
        D = pooled.size(-1)
        out_pooled = pooled.new_zeros(B, Nmax, D)
        out_valid = torch.zeros(B, Nmax, dtype=torch.bool, device=device)
        pos = 0
        for b, n in enumerate(per_owner_n):
            out_pooled[b, :n] = projected[pos:pos + n] * self.inject_gate * batch.inject_scale[b]
            out_valid[b, :n] = True
            pos += n

        if not self.use_maxsim:
            return out_pooled, out_valid, None, None

        # Token-level states, same (B, Nmax, L, D) scatter, for MaxSim.
        L = tokens.size(1)
        out_tokens = tokens.new_zeros(B, Nmax, L, D)
        out_tok_mask = torch.zeros(B, Nmax, L, dtype=torch.bool, device=device)
        pos = 0
        for b, n in enumerate(per_owner_n):
            out_tokens[b, :n] = tokens[pos:pos + n]
            out_tok_mask[b, :n] = tok_mask[pos:pos + n]
            pos += n

        return out_pooled, out_valid, out_tokens, out_tok_mask

    # ---- context compression ----

    def _context_codes(self, hidden_states: torch.Tensor, context_token_mask: torch.Tensor):
        B = hidden_states.size(0)
        query = self.context_codes.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = ~context_token_mask  # True = ignore
        # Guard against an example with an empty context span (would make
        # every key masked, which MultiheadAttention would turn into NaNs).
        # Branchless on purpose: `if empty_row.any():` forced a GPU->CPU sync on
        # every forward pass, draining the queue the CPU had built up ahead of
        # the GPU. Same result -- an empty row gets every key unmasked.
        empty_row = key_padding_mask.all(dim=-1, keepdim=True)
        key_padding_mask = key_padding_mask & ~empty_row
        codes, _ = self.code_attn(query, hidden_states, hidden_states,
                                   key_padding_mask=key_padding_mask, need_weights=False)
        return codes  # (B, m, D)

    # ---- MaxSim ----

    def _maxsim_scores(self, ctx_tokens: torch.Tensor, ctx_valid: torch.Tensor,
                        opt_tokens: torch.Tensor, opt_tok_mask: torch.Tensor,
                        opt_valid: torch.Tensor, chunk_size: int = 32):
        """Late-interaction score per option (Sec 2.6). Chunked over the
        option dimension to bound memory (B, N, L_ctx, L_opt) grows fast).

        ctx_tokens: (B, L_ctx, D) backbone hidden states over context span.
        opt_tokens: (B, N, L_opt, D) SEPARATE per-option token states.
        Returns: (B, N) maxsim scores (unscaled; caller applies maxsim_gate).
        """
        B, L_ctx, D = ctx_tokens.shape
        N = opt_tokens.size(1)
        c = F.normalize(self.late_proj(ctx_tokens), dim=-1)  # (B, Lctx, d)
        c = c.masked_fill(~ctx_valid.unsqueeze(-1), 0.0)
        ctx_count = ctx_valid.sum(dim=-1).clamp(min=1).float()  # (B,)

        out = ctx_tokens.new_zeros(B, N)
        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            e = F.normalize(self.late_proj(opt_tokens[:, start:end]), dim=-1)  # (B, n, Lopt, d)
            e = e.masked_fill(~opt_tok_mask[:, start:end].unsqueeze(-1), -1e4)
            # (B, n, Lctx, Lopt) via einsum, then max over Lopt, mask invalid
            # context positions to 0 before summing so padding never counts.
            sim = torch.einsum("btd,bnld->bntl", c, e)  # (B, n, Lctx, Lopt)
            max_over_opt_tokens = sim.max(dim=-1).values  # (B, n, Lctx)
            max_over_opt_tokens = max_over_opt_tokens.masked_fill(~ctx_valid.unsqueeze(1), 0.0)
            score = max_over_opt_tokens.sum(dim=-1) / ctx_count.unsqueeze(1)  # (B, n)
            out[:, start:end] = score
        return out.masked_fill(~opt_valid, 0.0)

    # ---- full forward ----

    def forward(self, tokenizer, batch: PackedBatch, device, k: Optional[int] = None,
                option_chunk_size: int = 128, maxsim_chunk_size: int = 32):
        """Returns a list of `k` (default self.k_max) logit tensors, one per
        recurrent pass, each (B, Nmax) with padding at -inf -- this is what
        the depth loss (Sec 3.2) sums over. Also returns the raw scorer
        state at the final depth for downstream halting-feature extraction
        (Sec 6, exp7c -- not used in exp7a/7b training)."""
        k = k or self.k_max

        opt_pooled, opt_pooled_valid, opt_tokens, opt_tok_mask = self.encode_and_project_options(
            tokenizer, batch, device, chunk_size=option_chunk_size)

        embeds = self._embed_packed_sequence(batch, opt_pooled, opt_pooled_valid)
        out = self.backbone(inputs_embeds=embeds, attention_mask=batch.attention_mask)
        H = out.last_hidden_state  # (B, L, D)

        codes = self._context_codes(H, batch.context_token_mask)  # (B, m, D)
        cls_vec = H[:, 0:1, :]  # (B, 1, D)

        B, Nmax = batch.mask_positions.shape
        gather_idx = batch.mask_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, H.size(-1))
        option_states = torch.gather(H, 1, gather_idx)  # (B, Nmax, D)

        head_seq = torch.cat([cls_vec, codes, option_states], dim=1)  # (B, 1+m+Nmax, D)
        head_pad = torch.cat([
            torch.zeros(B, 1 + self.n_context_codes, dtype=torch.bool, device=device),
            ~batch.valid_mask,
        ], dim=1)

        s0 = self.entry_layer(head_seq, src_key_padding_mask=head_pad)

        maxsim_raw = None
        if self.use_maxsim:
            maxsim_raw = self._maxsim_scores(
                H, batch.context_token_mask, opt_tokens, opt_tok_mask, batch.valid_mask,
                chunk_size=maxsim_chunk_size)  # (B, Nmax)

        memory_key_padding_mask = ~batch.attention_mask.bool()

        option_offset = 1 + self.n_context_codes
        logits_per_depth = []
        s = s0
        thoughts = []  # per-pass scratchpad entries, each (B, n_scratch, D)
        for depth in range(k):
            # Scaled s0 re-injection + LayerNorm prevents variance explosion across depths
            if depth == 0:
                block_input = s0
            else:
                block_input = self.recurrent_norm(s + self.s0_gate * s0)

            depth_idx = torch.full((B,), depth, dtype=torch.long, device=device)
            block_input = block_input + self.depth_embed(depth_idx).unsqueeze(1)
            if self.use_film_depth:
                block_input = (block_input * self.depth_scale(depth_idx).unsqueeze(1)
                               + self.depth_shift(depth_idx).unsqueeze(1))

            # Cross-attention memory: the backbone output, plus everything this
            # loop has concluded on earlier passes (empty on pass 1).
            if self.n_scratch > 0 and thoughts:
                thought_mem = torch.cat(thoughts, dim=1)  # (B, depth*n_scratch, D)
                mem = torch.cat([H, thought_mem], dim=1)
                mem_pad = torch.cat([
                    memory_key_padding_mask,
                    torch.zeros(B, thought_mem.size(1), dtype=torch.bool, device=device),
                ], dim=1)
            else:
                mem, mem_pad = H, memory_key_padding_mask

            for lm in self._lora_modules:
                lm.depth = depth

            # Outer residual skip connection (s = LayerNorm(s + block_out))
            block_out = self.recurrent_block(block_input, mem, key_padding_mask=head_pad,
                                            memory_key_padding_mask=mem_pad)
            s = self.recurrent_norm_out(s + block_out)

            if self.n_scratch > 0:
                # Masked mean over the head sequence -- padded option slots must
                # not leak into the summary, or examples with few options would
                # write systematically different thoughts than examples with many.
                keep = (~head_pad).unsqueeze(-1).to(s.dtype)
                summary = (s * keep).sum(1) / keep.sum(1).clamp(min=1.0)
                th = self.scratch_proj(summary).view(B, self.n_scratch, self.hidden)
                th = th * self.scratch_gate + self.scratch_pos(depth_idx).unsqueeze(1)
                thoughts.append(th)

            opt_repr = s[:, option_offset:option_offset + Nmax, :]
            if self.n_scorer_blocks > 0:
                opt_repr = self.attn_scorer(opt_repr, ~batch.valid_mask, mem, mem_pad)
            scores = self.scorer(opt_repr).squeeze(-1)  # (B, Nmax)
            if maxsim_raw is not None:
                scores = scores + self.maxsim_gate * maxsim_raw
            scores = scores.masked_fill(~batch.valid_mask, float("-inf"))
            logits_per_depth.append(scores)

        return logits_per_depth

    def freeze_backbone(self):
        """For the recursion/head-only continuation phase: freeze every
        backbone parameter (both weight-tied uses -- the packed-sequence
        pass and the independent option-encoding pass share these same
        weights, so this affects both at once) after loading a fully
        fine-tuned exp7a checkpoint. The idea: exp7a already let the
        backbone adapt to this task (the thing that made freezing fail
        badly in the earlier project history, e.g. Experiment 4's frozen
        1.5B decoder -- see PROJECT_HISTORY.md); freezing it AFTER that
        adaptation, rather than from random/pretrained init, is a different
        and much cheaper regime for training deeper/adaptive recursion,
        since backbone backward passes and their optimizer state are gone
        entirely. Gradient checkpointing is also disabled here if it was
        on -- it exists purely to reduce backbone activation memory during
        backbone backward, so it's pure overhead once the backbone no
        longer needs gradients at all (same reasoning as model_v4.py's
        `_freeze` helper from the poly-encoder era).

        Call this AFTER model.load_state_dict(...), not before -- freezing
        first and then loading weights would work fine functionally, but
        loading a fine-tuned checkpoint into a model where backbone params
        were already marked non-trainable is the wrong order to reason
        about if anything goes wrong.

        CORRECTION (found before the first local run, not caught on the
        A100 where 40GB hid it): this method used to also disable gradient
        checkpointing, on the theory that backward never flows through a
        frozen backbone. That's true for the SEPARATE option-encoding pass
        (encode_options_raw) -- its input (token ids) never requires grad,
        so autograd skips it entirely, exactly as intended. It is FALSE for
        the main packed-sequence pass: `embeds` (the backbone's input there)
        is `word_embed(input_ids) + injection`, and `injection` depends on
        the trainable option_projector -- so `embeds` itself requires grad,
        and autograd must still build and store the full backward graph
        through all 24 frozen layers to reach it, even though none of
        THEIR weights get a gradient. Activation memory for that pass is
        therefore close to full fine-tuning's, not near-zero. Gradient
        checkpointing is left exactly as constructed (respecting
        --no_grad_checkpoint) rather than force-disabled here.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def init_new_from_pretrained(self, missing_keys):
        """Initialize freshly added modules from trained weights instead of
        from scratch. Called by load_flexible after a checkpoint is loaded.

        attn_scorer: q/k/v input projections and the pre-attention LayerNorms
        are copied from the last recurrent layer's self-attention (-> option
        self-attention) and cross-attention (-> evidence cross-attention).
        Those projections already know how to form useful queries/keys over
        this model's hidden states; a random init would have to rediscover
        that from a few thousand steps. out_proj stays at zero, so the copied
        scorer still contributes nothing until the gradient says it should.
        """
        notes = []
        if self.n_scorer_blocks > 0 and any(k.startswith("attn_scorer.") for k in missing_keys):
            src = self.recurrent_block.layers[-1]
            for blk in self.attn_scorer.blocks:
                blk["self_attn"].in_proj_weight.copy_(src.self_attn.in_proj_weight)
                blk["self_attn"].in_proj_bias.copy_(src.self_attn.in_proj_bias)
                blk["cross_attn"].in_proj_weight.copy_(src.multihead_attn.in_proj_weight)
                blk["cross_attn"].in_proj_bias.copy_(src.multihead_attn.in_proj_bias)
                blk["norm_self"].load_state_dict(src.norm1.state_dict())
                blk["norm_cross"].load_state_dict(src.norm2.state_dict())
            notes.append("attn_scorer: q/k/v + norms copied from recurrent_block.layers[-1]; out_proj=0")
        return notes

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""Experiment 9 model: the exp7/8 decision architecture scaled to a ~4B
embedding-LLM backbone (Qwen3-Embedding-4B; 0.6B for local tests, 8B optional).

Why the packing changed. ModernBERT is bidirectional, so exp7/8 could put
every option before the context and read a [MASK] slot. Qwen3 is causal: a
slot placed before the context never sees it, and options placed after it
would each see the ones before them (order bias). exp9 therefore packs

    [prefix: instructions + context][branch 1][branch 2]...[branch N]

with a 4D attention mask where every branch sees the whole prefix and itself
only, and every branch's position ids restart at len(prefix). Each option is
scored exactly as if it had been run alone after the prefix (verified by
test_branch_mask.py: max relative diff 1.8e-6), so the score of an option
cannot depend on which other options are present or their order.

A branch is   [MARK] "Option: <text, budget-truncated>" [EOS]
  MARK  -- the injection slot: the option's vector is added to its embedding
  EOS   -- the readout: Qwen3-Embedding is trained to pool its sentence
           embedding at the final EOS token, so this is the position the
           pretrained model already uses to summarise what came before it.

Vector path. The option vector is Qwen3-Embedding's OWN pretrained embedding
of the untruncated option text (last-token pooling, LoRA adapters disabled,
no grad) -- the best embedding this backbone can produce, not a mean pool of
a model that was never trained to embed. It is projected (2-layer MLP + LN),
rescaled to the embedding table's typical norm, gated, and added at MARK.
The budget allocator shrinks each branch's text as N grows; at 0 text tokens
the branch is [MARK][EOS] and the vector carries the option alone.

Head: same as exp7a/8 -- 16 learned codes cross-attend over the prefix, then
[last prefix state ; codes ; option readouts] -> entry layer -> one block ->
per-option scorer. The loss (RLCD proper-score reward) is unchanged.
"""
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

QUESTION_TYPES = ["choice", "bool", "score"]
QTYPE_IDX = {t: i for i, t in enumerate(QUESTION_TYPES)}
TYPE_PREFIX = {"choice": "Pick exactly one option.", "bool": "Answer yes or no.",
               "score": "Pick the closest level on the ordinal scale."}
MARK_TOKEN = "<|object_ref_start|>"   # unused by the text-only embedding model; our injection slot


def n_heads_for(d, preferred=16):
    for h in range(preferred, 0, -1):
        if d % h == 0:
            return h
    return 1


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------

@dataclass
class Batch9:
    input_ids: torch.Tensor        # (B, L)
    position_ids: torch.Tensor     # (B, L)
    attn4d: torch.Tensor           # (B, 1, L, L) bool, True = may attend
    seg: torch.Tensor              # (B, L) 0 = prefix, i = branch i, -1 = padding (FlexAttention mask)
    prefix_mask: torch.Tensor      # (B, L) bool, prefix (context-bearing) tokens
    prefix_last: torch.Tensor      # (B,) index of the last prefix token
    mark_pos: torch.Tensor         # (B, N) MARK position per option, -1 = padding slot
    read_pos: torch.Tensor         # (B, N) EOS readout position per option
    valid_mask: torch.Tensor       # (B, N) bool
    inject_scale: torch.Tensor     # (B,) 1 = vector path on
    qtype_idx: torch.Tensor        # (B,)
    answer_idx: torch.Tensor       # (B,)
    opt_ids: torch.Tensor          # (U, T) unique option strings for the embedding pass
    opt_mask: torch.Tensor         # (U, T)
    opt_index: torch.Tensor        # (sum N,) flat option -> unique row
    opt_owner: torch.Tensor        # (sum N,) flat option -> example
    opt_slot: torch.Tensor         # (sum N,) flat option -> slot

    def pin_memory(self):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(self, k, v.pin_memory())
        return self

    def to(self, device):
        return Batch9(**{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in self.__dict__.items()})

    @property
    def n_tokens(self):
        return int(self.prefix_mask.numel())


class Packer9:
    """Turns exp8-style PackedExamples (context, instructions, option_texts,
    qtype, answer_idx, use_text, use_vector) into Batch9 micro-batches under
    a padded-token budget."""

    def __init__(self, tok, budget_total=2048, l_context=768, l_instructions=128, l_max_per_option=48,
                 opt_embed_len=64, max_tokens=12000, max_options=512, pad_multiple=64):
        self.tok = tok
        self.budget_total, self.l_context, self.l_instr = budget_total, l_context, l_instructions
        self.l_max_opt, self.opt_embed_len = l_max_per_option, opt_embed_len
        self.max_tokens, self.max_options, self.pad_multiple = max_tokens, max_options, pad_multiple
        self.mark_id = tok.convert_tokens_to_ids(MARK_TOKEN)
        if self.mark_id is None or self.mark_id == tok.unk_token_id:
            self.mark_id = tok.eos_token_id
        self.eos_id = tok.eos_token_id
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self._opt_word = tok("Option: ", add_special_tokens=False)["input_ids"]

    def _ids(self, text, n):
        if n <= 0 or not text:
            return []
        return self.tok(text, add_special_tokens=False, truncation=True, max_length=n)["input_ids"]

    def build_one(self, ex):
        instr = f"{TYPE_PREFIX[ex.qtype]} {ex.instructions}".strip()
        head = self._ids(f"Instruct: {instr}\nQuery: ", self.l_instr + 8)
        ctx = self._ids(ex.context, self.l_context)
        prefix = head + ctx
        n = len(ex.option_texts)
        overhead = 2 + len(self._opt_word)
        avail = self.budget_total - len(prefix) - n * overhead
        per = max(0, min(self.l_max_opt, avail // max(1, n))) if ex.use_text else 0
        ids, pos, seg = list(prefix), list(range(len(prefix))), [0] * len(prefix)
        marks, reads = [], []
        P = len(prefix)
        for i, t in enumerate(ex.option_texts, 1):
            body = (self._opt_word + self._ids(t, per)) if per > 0 else []
            branch = [self.mark_id] + body + [self.eos_id]
            marks.append(len(ids))
            reads.append(len(ids) + len(branch) - 1)
            ids += branch
            pos += list(range(P, P + len(branch)))
            seg += [i] * len(branch)
        return ids, pos, seg, P, marks, reads

    def assemble(self, items):
        """items: list of (PackedExample, meta, built)."""
        L = max(len(b[0]) for _, _, b in items)
        L = -(-L // self.pad_multiple) * self.pad_multiple
        N = max(len(b[4]) for _, _, b in items)
        B = len(items)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        pos = torch.zeros((B, L), dtype=torch.long)
        seg = torch.full((B, L), -1, dtype=torch.long)
        pmask = torch.zeros((B, L), dtype=torch.bool)
        plast = torch.zeros(B, dtype=torch.long)
        marks = torch.full((B, N), -1, dtype=torch.long)
        reads = torch.full((B, N), -1, dtype=torch.long)
        valid = torch.zeros((B, N), dtype=torch.bool)
        for i, (_, _, (x, p, s, P, m, r)) in enumerate(items):
            ids[i, :len(x)] = torch.tensor(x)
            pos[i, :len(x)] = torch.tensor(p)
            seg[i, :len(x)] = torch.tensor(s)
            pmask[i, :P] = True
            plast[i] = P - 1
            marks[i, :len(m)] = torch.tensor(m)
            reads[i, :len(r)] = torch.tensor(r)
            valid[i, :len(m)] = True
        ar = torch.arange(L)
        causal = ar[None, :] <= ar[:, None]
        q, k = seg[:, :, None], seg[:, None, :]
        allowed = causal[None] & ((k == 0) | (k == q)) & (k >= 0)
        # Padding rows attend to themselves only, so no row is fully masked (NaN-free SDPA).
        allowed = allowed | torch.eye(L, dtype=torch.bool)[None]
        exs = [e for e, _, _ in items]
        uniq, index, owner, slot = {}, [], [], []
        for b, e in enumerate(exs):
            for j, t in enumerate(e.option_texts):
                index.append(uniq.setdefault(t, len(uniq)))
                owner.append(b)
                slot.append(j)
        # Qwen3-Embedding pools at the final EOS: truncate the TEXT, then always
        # append EOS, then left-pad -- so the last position is EOS for every row.
        toks = [self._ids(t, self.opt_embed_len - 1) + [self.eos_id] for t in uniq]
        T = max(len(t) for t in toks)
        opt_ids = torch.full((len(toks), T), self.pad_id, dtype=torch.long)
        opt_mask = torch.zeros((len(toks), T), dtype=torch.long)
        for i, t in enumerate(toks):
            opt_ids[i, T - len(t):] = torch.tensor(t)
            opt_mask[i, T - len(t):] = 1
        enc = {"input_ids": opt_ids, "attention_mask": opt_mask}
        return Batch9(
            input_ids=ids, position_ids=pos, attn4d=allowed[:, None], seg=seg, prefix_mask=pmask, prefix_last=plast,
            mark_pos=marks, read_pos=reads, valid_mask=valid,
            inject_scale=torch.tensor([1.0 if e.use_vector else 0.0 for e in exs]),
            qtype_idx=torch.tensor([QTYPE_IDX[e.qtype] for e in exs]),
            answer_idx=torch.tensor([e.answer_idx for e in exs]),
            opt_ids=enc["input_ids"], opt_mask=enc["attention_mask"], opt_index=torch.tensor(index),
            opt_owner=torch.tensor(owner), opt_slot=torch.tensor(slot))

    def pack(self, pairs, max_tokens=None, max_options=None):
        """pairs: [(PackedExample, meta)] -> [(Batch9, metas)], length-sorted, budgeted."""
        mt, mo = max_tokens or self.max_tokens, max_options or self.max_options
        items = [(e, m, self.build_one(e)) for e, m in pairs]
        items.sort(key=lambda it: len(it[2][0]))
        out, cur = [], []
        for it in items:
            L = -(-len(it[2][0]) // self.pad_multiple) * self.pad_multiple
            n_opt = sum(len(c[0].option_texts) for c in cur) + len(it[0].option_texts)
            # attention is O(L^2) per row with an explicit mask: budget rows * L
            if cur and (L * (len(cur) + 1) > mt or n_opt > mo):
                out.append((self.assemble(cur), [c[1] for c in cur]))
                cur = []
            cur.append(it)
        if cur:
            out.append((self.assemble(cur), [c[1] for c in cur]))
        return out


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class Projector(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, x):
        return self.net(x)


class S9DecisionModel(nn.Module):
    def __init__(self, backbone="Qwen/Qwen3-Embedding-4B", lora_r=64, lora_alpha=128, lora_dropout=0.05,
                 full_finetune=False, n_codes=16, head_layers=2, dropout=0.1, gate_init=0.1,
                 gradient_checkpointing=True, dtype=torch.bfloat16, attn_impl="sdpa",
                 head="rethink", n_latents=128, resampler_layers=2, block_layers=4, k_max=6, n_scratch=8):
        super().__init__()
        # attn_impl "flex_attention": a fused kernel built from the branch mask FUNCTION that
        # skips fully-masked blocks (branches never see each other; ~50% of blocks on a
        # typical batch). Needs Triton (Linux). "sdpa": explicit 4D bool mask, works anywhere.
        self.attn_impl = attn_impl
        base = AutoModel.from_pretrained(backbone, dtype=dtype, attn_implementation=attn_impl)
        d = base.config.hidden_size
        self.hidden = d
        self.full_finetune = full_finetune
        with torch.no_grad():
            emb = base.get_input_embeddings().weight
            self.register_buffer("emb_scale", emb.float().norm(dim=-1).mean().clone())
        if full_finetune:
            self.backbone = base
            self.has_lora = False
        else:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"])
            self.backbone = get_peft_model(base, cfg)
            self.has_lora = True
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if self.has_lora:
                self.backbone.enable_input_require_grads()

        h = n_heads_for(d)
        self.option_projector = Projector(d)
        self.type_embed = nn.Embedding(len(QUESTION_TYPES), d)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        self.inject_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.head_kind = head
        self.head_ckpt = gradient_checkpointing
        if head == "rethink":
            # exp9's scaled head: resampler + K-pass recurrent block + gated scratchpad (s9_head.py)
            from s9_head import RethinkHead
            self.head = RethinkHead(d, n_latents, resampler_layers, block_layers, k_max, n_scratch, dropout)
            self.k_max = k_max
        else:
            # exp7a/8 head, for an ablation at the same backbone
            self.k_max = 1
            self.n_codes = n_codes
            self.context_codes = nn.Parameter(torch.randn(n_codes, d) * d ** -0.5)
            self.code_attn = nn.MultiheadAttention(d, h, batch_first=True, dropout=dropout)
            self.entry_layer = nn.TransformerEncoderLayer(d, h, d * 4, dropout, activation="gelu",
                                                          batch_first=True, norm_first=True)
            self.block = nn.ModuleList([nn.TransformerEncoderLayer(d, h, d * 4, dropout, activation="gelu",
                                                                   batch_first=True, norm_first=True)
                                        for _ in range(head_layers)])
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(d, 1))

    # -- backbone access (LoRA-wrapped or plain) --
    def _core(self):
        return self.backbone

    def input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @torch.no_grad()
    def embed_options(self, ids, mask, chunk=256):
        """Qwen3-Embedding's pretrained sentence embedding (last-token pooling,
        left-padded), LoRA adapters disabled: the stock embedding model's vector."""
        was = self.backbone.training
        self.backbone.eval()
        outs = []
        ctx = self.backbone.disable_adapter() if self.has_lora else _nullctx()
        with ctx:
            for s in range(0, ids.size(0), chunk):
                h = self.backbone(input_ids=ids[s:s + chunk], attention_mask=mask[s:s + chunk]).last_hidden_state
                outs.append(F.normalize(h[:, -1].float(), dim=-1))
        if was:
            self.backbone.train()
        return torch.cat(outs, 0)

    def forward(self, b: Batch9, all_depths=False, k=None):
        """Final-depth logits (B, N); with all_depths=True, the list of every pass's logits."""
        B, N = b.mark_pos.shape
        d = self.hidden
        # [B] option vectors -> projected, norm-matched, gated
        vec = self.embed_options(b.opt_ids, b.opt_mask)[b.opt_index]
        proj = self.option_projector(vec.to(self.option_projector.net[0].weight.dtype))
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.emb_scale
        gate = (self.inject_gate * b.inject_scale.to(proj.dtype))[b.opt_owner].unsqueeze(-1)
        add_opt = proj.new_zeros(B, N, d)
        add_opt[b.opt_owner, b.opt_slot] = (proj * gate).to(add_opt.dtype)
        add = (add_opt + self.type_embed(b.qtype_idx).unsqueeze(1)) * b.valid_mask.unsqueeze(-1).to(add_opt.dtype)

        embeds = self.input_embeddings()(b.input_ids)
        idx = b.mark_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, d)
        inj = torch.zeros_like(embeds).scatter_add_(1, idx, add.to(embeds.dtype))
        H = self.backbone(inputs_embeds=embeds + inj, attention_mask=self._attn_mask(b),
                          position_ids=b.position_ids).last_hidden_state
        opt_read = H.gather(1, b.read_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, d))
        if self.head_kind == "rethink":
            hdt = self.head.scorer[1].weight.dtype
            Hf, of = H.to(hdt), opt_read.to(hdt)
            self.head.ckpt = self.head_ckpt
            outs = self.head(Hf, b.prefix_mask, b.prefix_last, of, b.valid_mask, k)
            return outs if all_depths else outs[-1]

        Hf = H.to(self.context_codes.dtype)
        kpm = ~b.prefix_mask
        codes, _ = self.code_attn(self.context_codes.unsqueeze(0).expand(B, -1, -1), Hf, Hf,
                                  key_padding_mask=kpm, need_weights=False)
        last = Hf.gather(1, b.prefix_last.view(B, 1, 1).expand(-1, 1, d))
        opt = Hf.gather(1, b.read_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, d))
        seq = torch.cat([last, codes, opt], 1)
        pad = torch.cat([torch.zeros(B, 1 + self.n_codes, dtype=torch.bool, device=H.device), ~b.valid_mask], 1)
        s = self.entry_layer(seq, src_key_padding_mask=pad)
        for layer in self.block:
            s = layer(s, src_key_padding_mask=pad)
        off = 1 + self.n_codes
        logits = self.scorer(s[:, off:off + N]).squeeze(-1).float()
        logits = logits.masked_fill(~b.valid_mask, float("-inf"))
        return [logits] if all_depths else logits

    def _attn_mask(self, b):
        if self.attn_impl != "flex_attention":
            return b.attn4d
        from torch.nn.attention.flex_attention import create_block_mask
        seg = b.seg
        Bn, L = seg.shape

        def mask_mod(bi, h, q, k):
            sq, sk = seg[bi, q], seg[bi, k]
            return ((k <= q) & ((sk == 0) | (sk == sq)) & (sk >= 0)) | (q == k)

        return create_block_mask(mask_mod, Bn, None, L, L, device=seg.device)

    # -- checkpointing: only what trains (LoRA + head), never the frozen 4B base --
    def trainable_state_dict(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().to("cpu") for k, v in self.state_dict().items() if k in keep}

    def load_trainable_state_dict(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        need = {n for n, p in self.named_parameters() if p.requires_grad}
        not_loaded = sorted(need - set(sd))
        if not_loaded or unexpected:
            raise RuntimeError(f"checkpoint mismatch: {len(not_loaded)} trainable params missing "
                               f"(e.g. {not_loaded[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})")

    def param_counts(self):
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bb = sum(p.numel() for n, p in self.named_parameters() if n.startswith("backbone.") and not p.requires_grad)
        return {"total": tot, "trainable": tr, "frozen_backbone": bb}


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def get_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

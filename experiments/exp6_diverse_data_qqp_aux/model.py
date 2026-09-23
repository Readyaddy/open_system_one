"""Experiment 4 model: Poly-Encoder architecture, same mechanism as
iteration 4 / experiment 1, on a decoder-LLM backbone instead of an
encoder. See NOTES.md for the full rationale and the trail of
trust_remote_code breakages that led here.

Backbone: `Qwen/Qwen2.5-1.5B` -- the PLAIN, natively-supported (no custom
code) Qwen2.5 decoder LLM. No embedding-specific conversion has been
applied to it by anyone; our own contrastive fine-tuning is what turns it
into an embedder, same as every backbone in this project, just starting
from a much larger, more recent pretraining run.

One real architectural adaptation vs. every prior backbone: Qwen2.5 is a
CAUSAL decoder -- each token's hidden state only encodes itself and the
tokens BEFORE it (no bidirectional attention, unlike RoBERTa / e5 /
ModernBERT). This means the outcome side's single pooled vector must use
LAST-TOKEN pooling (the last token is the only position that has attended
to the entire sequence), not the mean-pooling used everywhere else in this
project, which would average in many under-informed early-token vectors.
Left-padding is used specifically so "the last token" is always at a fixed
position (-1) across a padded batch, per standard practice for decoder-LLM
embeddings (e.g. the E5-Mistral / LLM2Vec / GritLM line of work).

The poly-encoder's CONTEXT side does not need this change: it already
operates on the full token-level hidden states (not a single pooled
vector), so causality is just an inherent property of what information
is available at each position, not something the pooling strategy needs
to work around.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BACKBONE = "Qwen/Qwen2.5-0.5B"  # EXPERIMENT 6: same backbone as experiment 5 (validated:
# fits comfortably fully unfrozen, ~11.6GB, reasonable speed), now trained
# on the diverse 5-source local data (data/intent_corpus/) plus a genuine
# QQP paraphrase-pair auxiliary objective (data/qqp_paraphrase_pairs/),
# instead of the narrow 3-dataset corpus and template-only augmentation
# used everywhere before. See train.py and data/README.md.


class OutcomeEncoder(nn.Module):
    """Last-token-pooled: one vector per candidate, computed independently
    of any context, so the whole candidate bank stays cacheable."""

    def __init__(self, out_dim: int, hidden_dim: int, backbone: str, freeze_layers: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        _freeze(self.backbone, freeze_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _last_token_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled)


class PolyContextEncoder(nn.Module):
    """Poly-encoder context side: produces m parallel context vectors via
    m learned query codes attending over the full (causal) token sequence."""

    def __init__(self, out_dim: int, hidden_dim: int, backbone: str, freeze_layers: int, n_codes: int = 16):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        _freeze(self.backbone, freeze_layers)

        self.n_codes = n_codes
        self.codes = nn.Parameter(torch.randn(n_codes, hidden) * (hidden ** -0.5))
        self.code_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=8, batch_first=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state  # (B, T, H)
        B = hidden_states.size(0)

        query = self.codes.unsqueeze(0).expand(B, -1, -1)  # (B, m, H)
        key_padding_mask = attention_mask == 0  # True where PADDED
        ctx_codes, _ = self.code_attn(query, hidden_states, hidden_states,
                                       key_padding_mask=key_padding_mask, need_weights=False)
        return self.proj(ctx_codes)  # (B, m, out_dim)


def _last_token_pool(last_hidden_state, attention_mask):
    """With left-padding, the last real token of every row is always at
    position -1 -- no per-row index arithmetic needed."""
    return last_hidden_state[:, -1, :]


def _freeze(backbone, freeze_layers: int):
    if freeze_layers <= 0:
        return
    for p in backbone.embed_tokens.parameters():
        p.requires_grad = False
    for layer in backbone.layers[:freeze_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    # The final RMSNorm (applied after all decoder layers) was previously
    # missed here -- leaving it trainable meant the backbone was never
    # TRULY fully frozen even at freeze_layers == total layer count, which
    # needlessly re-triggered gradient checkpointing (forcing a full
    # forward recompute through all 28 layers during backward, just to
    # get gradients for this one small vector) and cost real throughput.
    if freeze_layers >= len(backbone.layers):
        for p in backbone.norm.parameters():
            p.requires_grad = False


class JEPAPolyEncoderV4(nn.Module):
    def __init__(self, out_dim: int = 512, hidden_dim: int = 1024, n_codes: int = 16,
                 temperature: float = 0.07, backbone: str = BACKBONE, freeze_layers: int = 16):
        super().__init__()
        self.context_encoder = PolyContextEncoder(out_dim, hidden_dim, backbone, freeze_layers, n_codes)
        self.outcome_encoder = OutcomeEncoder(out_dim, hidden_dim, backbone, freeze_layers)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        self.out_dim = out_dim

    def encode_context(self, input_ids, attention_mask):
        """Returns (B, m, out_dim) -- m parallel, un-normalized context codes."""
        return self.context_encoder(input_ids, attention_mask)

    def encode_outcome(self, input_ids, attention_mask):
        """Returns (K, out_dim) -- one L2-normalized vector per candidate."""
        emb = self.outcome_encoder(input_ids, attention_mask)
        return nn.functional.normalize(emb, dim=-1)

    def compatibility(self, ctx_codes, outcome_emb):
        """Identical poly-encoder scoring mechanism as iteration 4 /
        experiment 1 -- see those files' docstrings for the full
        explanation. ctx_codes: (B, m, D). outcome_emb: (K, D). Returns
        (B, K) compatibility logits."""
        ctx_codes = nn.functional.normalize(ctx_codes, dim=-1)
        d = ctx_codes.size(-1)

        scores = torch.einsum("bmd,kd->bmk", ctx_codes, outcome_emb) / (d ** 0.5)
        weights = torch.softmax(scores, dim=1)

        final_ctx = torch.einsum("bmk,bmd->bkd", weights, ctx_codes)
        final_ctx = nn.functional.normalize(final_ctx, dim=-1)

        compat = torch.einsum("bkd,kd->bk", final_ctx, outcome_emb)
        temp = self.log_temperature.exp().clamp(min=1e-3)
        return compat / temp

    def compatibility_pairwise(self, ctx_codes, outcome_emb):
        """EXPERIMENT 6 addition, for the QQP auxiliary task: one-to-one
        pairing (text1[i] vs text2[i]) instead of the main task's cross
        product (every context vs every candidate). QQP already supplies
        explicit positive AND negative labels per pair, so there's no
        need for in-batch/bank negatives here -- this just scores each
        pair independently.

        ctx_codes: (B, m, D). outcome_emb: (B, D) -- SAME batch dimension,
        paired by index. Returns (B,) compatibility logits.
        """
        ctx_codes = nn.functional.normalize(ctx_codes, dim=-1)
        d = ctx_codes.size(-1)

        scores = torch.einsum("bmd,bd->bm", ctx_codes, outcome_emb) / (d ** 0.5)  # (B, m)
        weights = torch.softmax(scores, dim=1)

        final_ctx = torch.einsum("bm,bmd->bd", weights, ctx_codes)  # (B, D)
        final_ctx = nn.functional.normalize(final_ctx, dim=-1)

        compat = (final_ctx * outcome_emb).sum(dim=-1)  # (B,) cosine similarity
        temp = self.log_temperature.exp().clamp(min=1e-3)
        return compat / temp

    def compatibility_grouped(self, ctx_codes, outcome_emb, valid_mask):
        """MCQ-style scoring: unlike `compatibility` (one outcome bank
        SHARED across the whole batch), each example here has its OWN
        candidate set -- e.g. one question's 4 answer options are only
        ever compared against that question's context, never against
        another example's options. Batches of examples with differing
        option counts are padded to the batch's max count; `valid_mask`
        marks which positions are real options vs. padding.

        ctx_codes: (B, m, D). outcome_emb: (B, K, D) -- K = this batch's
        max option count, L2-normalized, padding rows can be anything
        (masked out below). valid_mask: (B, K) bool, True = real option.
        Returns (B, K) logits with padding positions set to -inf, so
        softmax/cross-entropy can never select or draw gradient from a
        padding slot.
        """
        ctx_codes = nn.functional.normalize(ctx_codes, dim=-1)
        d = ctx_codes.size(-1)

        scores = torch.einsum("bmd,bkd->bmk", ctx_codes, outcome_emb) / (d ** 0.5)
        weights = torch.softmax(scores, dim=1)

        final_ctx = torch.einsum("bmk,bmd->bkd", weights, ctx_codes)
        final_ctx = nn.functional.normalize(final_ctx, dim=-1)

        compat = torch.einsum("bkd,bkd->bk", final_ctx, outcome_emb)
        temp = self.log_temperature.exp().clamp(min=1e-3)
        logits = compat / temp
        return logits.masked_fill(~valid_mask, float("-inf"))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self):
        self.context_encoder.backbone.gradient_checkpointing_enable()
        self.outcome_encoder.backbone.gradient_checkpointing_enable()


def get_tokenizer(backbone: str = BACKBONE):
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    tokenizer.padding_side = "left"  # required for fixed-position last-token pooling
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

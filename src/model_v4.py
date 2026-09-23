"""Iteration 4 model: Poly-Encoder architecture (Humeau et al., 2019,
"Poly-encoders: Transformer Architectures and Pre-training Strategies for
Fast and Accurate Multi-sentence Scoring"), on a bigger backbone.

This is the concrete architecture behind "a transformer with parallel
outputs": instead of collapsing the context transformer's whole token
sequence into ONE mean-pooled vector (a bi-encoder, what v1-v3 did), the
context encoder learns `m` query vectors ("codes") that attend over all
of the context's token representations *in parallel*, in a single forward
pass, producing `m` distinct context vectors instead of 1.

At scoring time, each candidate outcome vector then attends over those m
context codes (a second, tiny attention op) to pull out the one blend of
the m codes most relevant to THAT candidate, before the final dot product.
This gives each candidate its own view of the context -- much closer to a
full cross-encoder's accuracy -- while still (a) computing the outcome
side once, independent of context, so candidates stay pre-computable/
cacheable, and (b) scoring all candidates for a whole batch of contexts in
one batched matrix op, not a python loop. That combination -- one shared
transformer pass producing several parallel outputs, cheaply recombined
per-candidate -- is the standard, well-established way to "crack" the
bi-encoder-vs-cross-encoder tradeoff, rather than inventing something ad
hoc.

Other standard techniques applied here vs. earlier iterations:
  - Bigger backbone: all-roberta-large-v1 (355M params/encoder, 1024-dim)
    instead of MiniLM (22M) / mpnet-base (110M).
  - Partial layer freezing: bottom `freeze_layers` transformer blocks are
    frozen (standard large-model fine-tuning practice -- lower layers hold
    generic syntax/semantics, upper layers adapt to the task), cutting
    trainable-parameter memory substantially.
  - Gradient checkpointing support (enabled in train_v4.py) to afford a
    bigger effective batch size on a 12GB GPU.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BACKBONE = "sentence-transformers/all-roberta-large-v1"


class OutcomeEncoder(nn.Module):
    """Standard mean-pooled bi-encoder side: one vector per candidate,
    computed independently of any context, so the whole candidate bank can
    be pre-computed and cached at inference time."""

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
        pooled = _mean_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled)  # (B, out_dim) -- NOT normalized here; normalized at scoring time


class PolyContextEncoder(nn.Module):
    """Poly-encoder context side: produces m parallel context vectors via
    m learned query codes attending over the full token sequence."""

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
        key_padding_mask = attention_mask == 0  # True where PADDED (nn.MultiheadAttention convention)
        ctx_codes, _ = self.code_attn(query, hidden_states, hidden_states,
                                       key_padding_mask=key_padding_mask, need_weights=False)
        return self.proj(ctx_codes)  # (B, m, out_dim)


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def _freeze(backbone, freeze_layers: int):
    if freeze_layers <= 0:
        return
    for p in backbone.embeddings.parameters():
        p.requires_grad = False
    for layer in backbone.encoder.layer[:freeze_layers]:
        for p in layer.parameters():
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
        """ctx_codes: (B, m, D) un-normalized poly codes.
        outcome_emb: (K, D) L2-normalized candidate vectors.
        Returns (B, K) compatibility logits.

        This is the poly-encoder final scoring step: each candidate
        attends over the m context codes to build a candidate-specific
        context vector, then dot-products with that candidate. Done for
        every (context, candidate) pair in the batch at once via einsum --
        no python-level loop over candidates.
        """
        ctx_codes = nn.functional.normalize(ctx_codes, dim=-1)  # (B, m, D)
        d = ctx_codes.size(-1)

        # attention scores: for each (batch, candidate), how relevant is each of the m codes
        scores = torch.einsum("bmd,kd->bmk", ctx_codes, outcome_emb) / (d ** 0.5)  # (B, m, K)
        weights = torch.softmax(scores, dim=1)  # softmax over the m codes

        # candidate-specific context vector: weighted blend of the m codes, per candidate
        final_ctx = torch.einsum("bmk,bmd->bkd", weights, ctx_codes)  # (B, K, D)
        final_ctx = nn.functional.normalize(final_ctx, dim=-1)

        compat = torch.einsum("bkd,kd->bk", final_ctx, outcome_emb)  # (B, K) cosine similarity
        temp = self.log_temperature.exp().clamp(min=1e-3)
        return compat / temp

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self):
        self.context_encoder.backbone.gradient_checkpointing_enable()
        self.outcome_encoder.backbone.gradient_checkpointing_enable()


def get_tokenizer(backbone: str = BACKBONE):
    return AutoTokenizer.from_pretrained(backbone)

"""Iteration 3 model: bigger pretrained backbone + deeper projection head.

Same JEPA-style dual-encoder design as v1/v2 -- two independent encoders,
compatibility = cosine similarity / learned temperature, no decoding -- but
scaled up: all-mpnet-base-v2 (110M params, 768-dim) instead of MiniLM
(22M params, 384-dim), with a deeper projection head (with residual +
layernorm) on each side.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BACKBONE = "sentence-transformers/all-mpnet-base-v2"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class ProjectionHead(nn.Module):
    """2-layer MLP with a residual connection and layernorm, projecting the
    pooled transformer output into the shared compatibility space."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, x):
        h = self.norm_in(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        out = h + self.residual(x)
        return self.norm_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, out_dim: int = 512, hidden_dim: int = 1024, backbone: str = BACKBONE,
                 freeze_layers: int = 0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        self.head = ProjectionHead(hidden, hidden_dim, out_dim)

        if freeze_layers > 0:
            # Freeze the embeddings + first N transformer layers, leave the
            # rest trainable -- cheaper fine-tuning without losing all
            # adaptability, kept configurable rather than hardcoded on.
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False
            for layer in self.backbone.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        proj = self.head(pooled)
        return nn.functional.normalize(proj, dim=-1)


class JEPADecisionModelV3(nn.Module):
    def __init__(self, out_dim: int = 512, hidden_dim: int = 1024, temperature: float = 0.07,
                 backbone: str = BACKBONE, freeze_layers: int = 0):
        super().__init__()
        self.context_encoder = TransformerEncoder(out_dim, hidden_dim, backbone, freeze_layers)
        self.outcome_encoder = TransformerEncoder(out_dim, hidden_dim, backbone, freeze_layers)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    def encode_context(self, input_ids, attention_mask):
        return self.context_encoder(input_ids, attention_mask)

    def encode_outcome(self, input_ids, attention_mask):
        return self.outcome_encoder(input_ids, attention_mask)

    def compatibility(self, ctx_emb, outcome_emb):
        temp = self.log_temperature.exp().clamp(min=1e-3)
        return ctx_emb @ outcome_emb.t() / temp

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_tokenizer(backbone: str = BACKBONE):
    return AutoTokenizer.from_pretrained(backbone)

"""Dual-encoder JEPA-style compatibility model, v2: pretrained transformer
backbone instead of a from-scratch bag-of-words encoder.

Iteration 1 showed the core idea works (100% on trained descriptions, 75%
on paraphrases) but generalization was capped by a tiny mean-pooled
embedding with no pretrained semantic knowledge. Here both the context
encoder and the outcome encoder start from a pretrained sentence encoder
(all-MiniLM-L6-v2, 6-layer transformer, 384-dim) and are fine-tuned
independently (separate weight copies) with the same contrastive
objective: cosine-similarity compatibility, softmax cross-entropy over the
full candidate outcome bank.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class TransformerEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, backbone: str = BACKBONE):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        proj = self.proj(pooled)
        return nn.functional.normalize(proj, dim=-1)


class JEPADecisionModelV2(nn.Module):
    def __init__(self, out_dim: int = 256, temperature: float = 0.07, backbone: str = BACKBONE):
        super().__init__()
        self.context_encoder = TransformerEncoder(out_dim, backbone)
        self.outcome_encoder = TransformerEncoder(out_dim, backbone)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    def encode_context(self, input_ids, attention_mask):
        return self.context_encoder(input_ids, attention_mask)

    def encode_outcome(self, input_ids, attention_mask):
        return self.outcome_encoder(input_ids, attention_mask)

    def compatibility(self, ctx_emb, outcome_emb):
        temp = self.log_temperature.exp().clamp(min=1e-3)
        return ctx_emb @ outcome_emb.t() / temp


def get_tokenizer(backbone: str = BACKBONE):
    return AutoTokenizer.from_pretrained(backbone)

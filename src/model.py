"""Dual-encoder, JEPA-style compatibility model.

Two encoders map into the same embedding space:
  - context encoder: encodes the situation (the support ticket text)
  - outcome encoder: encodes a candidate outcome's natural-language description

Prediction is NOT generation. It's argmax cosine-similarity between the
context embedding and each candidate outcome embedding -- a single forward
pass per side, then a compatibility (energy) check in representation space.
"""
import re
import torch
import torch.nn as nn


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class Vocab:
    def __init__(self):
        self.tok2idx = {"<pad>": 0, "<unk>": 1}

    def build(self, texts):
        for t in texts:
            for tok in tokenize(t):
                if tok not in self.tok2idx:
                    self.tok2idx[tok] = len(self.tok2idx)
        return self

    def encode(self, text, max_len=32):
        ids = [self.tok2idx.get(tok, 1) for tok in tokenize(text)][:max_len]
        if not ids:
            ids = [1]
        return ids

    def __len__(self):
        return len(self.tok2idx)


class Encoder(nn.Module):
    """Shared architecture for both context and outcome encoders.

    Token embedding -> mean pooling over tokens -> MLP -> L2-normalized
    embedding. Mean pooling keeps this genuinely small and CPU-fast; the
    point of this first iteration is the dual-encoder + contrastive
    objective, not a heavy sequence model.
    """

    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor):
        # token_ids, mask: (batch, seq_len)
        emb = self.embedding(token_ids)  # (B, T, D)
        mask = mask.unsqueeze(-1).float()
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        out = self.mlp(pooled)
        return nn.functional.normalize(out, dim=-1)


class JEPADecisionModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128, out_dim: int = 64,
                 temperature: float = 0.1):
        super().__init__()
        self.context_encoder = Encoder(vocab_size, emb_dim, hidden_dim, out_dim)
        self.outcome_encoder = Encoder(vocab_size, emb_dim, hidden_dim, out_dim)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    def encode_context(self, token_ids, mask):
        return self.context_encoder(token_ids, mask)

    def encode_outcome(self, token_ids, mask):
        return self.outcome_encoder(token_ids, mask)

    def compatibility(self, ctx_emb, outcome_emb):
        """Cosine similarity scaled by learned temperature -> compatibility logits."""
        temp = self.log_temperature.exp().clamp(min=1e-3)
        return ctx_emb @ outcome_emb.t() / temp


def pad_batch(list_of_ids, pad_idx=0):
    max_len = max(len(x) for x in list_of_ids)
    batch = torch.full((len(list_of_ids), max_len), pad_idx, dtype=torch.long)
    mask = torch.zeros((len(list_of_ids), max_len), dtype=torch.long)
    for i, ids in enumerate(list_of_ids):
        batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, :len(ids)] = 1
    return batch, mask

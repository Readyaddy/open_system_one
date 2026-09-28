"""Experiment 8 model: exp7a's System-1 architecture, loaded exactly.

Why a separate model file instead of exp7's model.py: exp7a was trained with
k_max=1 and a head of TransformerEncoderLayers. model.py has since moved on
(TransformerDecoderLayer recurrent block with cross-attention, depth
embeddings, extra LayerNorms, scratchpad, FiLM, LoRA). Loading exp7a into it
silently re-maps norm2 onto the new cross-attention sublayer and random-inits
the rest -- i.e. the head exp8 would start from is NOT the head exp7a trained.
This file rebuilds the exp7a head with matching parameter names, so
`load_state_dict(strict=True)` succeeds and the starting point is exactly
exp7a (verified by reproducing exp7a's recorded zero-shot accuracy, see
verify_exp7a.py).

Pipeline (see the exp8 NOTES.md for the first-principles walkthrough):
  [A] packed sequence  [CLS] type+instr [SEP] [MASK] opt ... [SEP] context [SEP]
  [B] option vectors   same backbone, pooled, projected, gated, added at [MASK]
  [C] backbone         ModernBERT-large, one pass -> H
  [D] readout          [CLS ; 16 context codes ; N option slots]
  [E] head             entry layer -> 2-layer block, ONE pass (System 1)
  [F] scorer           per-option logit -> softmax over options

Packing/tokenization/PackedBatch are reused from exp7's model.py unchanged.
"""
import importlib.util
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

_EXP7_DIR = os.path.join(os.path.dirname(__file__), "..", "exp7_hybrid_decision")


def _load_exp7_model_module():
    # Loaded by path under a unique name: plain `import model` resolves to
    # src/model.py whenever src/ is on sys.path (a known trap in this repo).
    spec = importlib.util.spec_from_file_location("exp7_model", os.path.join(_EXP7_DIR, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    # Registered so pickle can find classes like PackedSequenceBuilder by
    # module name -- DataLoader workers (spawned on Windows) need that.
    sys.modules["exp7_model"] = mod
    spec.loader.exec_module(mod)
    return mod


E7 = _load_exp7_model_module()
BACKBONE = E7.BACKBONE
QUESTION_TYPES = E7.QUESTION_TYPES
QTYPE_IDX = E7.QTYPE_IDX
PackedExample = E7.PackedExample
PackedBatch = E7.PackedBatch
PackedSequenceBuilder = E7.PackedSequenceBuilder
get_tokenizer = E7.get_tokenizer
Projector = E7.Projector
_safe_n_heads = E7._safe_n_heads


class Exp7aBlock(nn.Module):
    """exp7a's 'recurrent' block: 2 pre-norm TransformerEncoderLayers.
    Attribute names (layers.{i}.self_attn/linear1/linear2/norm1/norm2) match
    the exp7a checkpoint exactly."""

    def __init__(self, dim, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=_safe_n_heads(dim), dim_feedforward=dim * 4,
                                       dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(n_layers)])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        return x


class S1DecisionModel(nn.Module):
    """head_input: how the single head pass is fed, since exp7a's forward code
    isn't in git and NOTES.md Sec 2.5's pseudocode `s_k = Block(s_{k-1} + s_0)`
    with s_0 as the starting state reads as Block(2*s0) at k=1. Both are kept
    selectable; verify_exp7a.py picks the one that reproduces exp7a's numbers.
      "s0x2": Block(s0 + s0)     "s0": Block(s0)
    Verified 2026-09-24: "s0" reproduces exp7a's recorded epoch-10 val (0.6433)
    and zero-shot (0.7133) exactly; "s0x2" gives zero-shot 0.7100.
    """

    def __init__(self, backbone: str = BACKBONE, mask_token_id: int = None, n_context_codes: int = 16,
                 head_n_layers: int = 2, dropout: float = 0.1, gradient_checkpointing: bool = True,
                 head_input: str = "s0", maxsim_dim: int = 128):
        super().__init__()
        if mask_token_id is None:
            raise ValueError("mask_token_id is required")
        self.backbone = AutoModel.from_pretrained(backbone)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        hidden = self.backbone.config.hidden_size
        self.hidden = hidden
        self.mask_token_id = mask_token_id
        self.head_input = head_input
        self.n_context_codes = n_context_codes

        self.option_projector = Projector(hidden)
        self.type_embed = nn.Embedding(len(QUESTION_TYPES), hidden)
        self.inject_gate = nn.Parameter(torch.tensor(0.01))
        with torch.no_grad():
            emb_scale = self.backbone.get_input_embeddings().weight.norm(dim=-1).mean().clone()
        self.register_buffer("emb_scale", emb_scale)

        self.context_codes = nn.Parameter(torch.randn(n_context_codes, hidden) * (hidden ** -0.5))
        self.code_attn = nn.MultiheadAttention(hidden, _safe_n_heads(hidden), batch_first=True, dropout=dropout)

        self.entry_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=_safe_n_heads(hidden), dim_feedforward=hidden * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.recurrent_block = Exp7aBlock(hidden, head_n_layers, dropout)
        self.scorer = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, 1))
        # Unused (MaxSim stays off) -- kept only so exp7a checkpoints load strictly.
        self.late_proj = nn.Linear(hidden, maxsim_dim)
        self.maxsim_gate = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    # ---- [B] option vectors ----

    def encode_options(self, opt_ids, opt_mask, chunk_size=256):
        """Mean-pooled backbone states for each DISTINCT option string in the
        batch (dedup happens in the collate function)."""
        pooled = []
        for s in range(0, opt_ids.size(0), chunk_size):
            ids, m = opt_ids[s:s + chunk_size], opt_mask[s:s + chunk_size]
            h = self.backbone(input_ids=ids, attention_mask=m).last_hidden_state
            mf = m.unsqueeze(-1).to(h.dtype)
            pooled.append((h * mf).sum(1) / mf.sum(1).clamp(min=1.0))
        return torch.cat(pooled, 0)

    def forward(self, batch, opt, return_hidden=False):
        """batch: PackedBatch on device. opt: dict of device tensors from the
        collate: ids, mask (unique option strings), index (flat option ->
        unique row), owner (flat option -> example), slot (flat -> slot).
        Returns logits (B, Nmax) with -inf at padding slots."""
        B, Nmax = batch.mask_positions.shape
        pooled = self.encode_options(opt["ids"], opt["mask"])[opt["index"]]
        proj = self.option_projector(pooled)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.emb_scale
        gate = (self.inject_gate * batch.inject_scale.to(proj.dtype))[opt["owner"]].unsqueeze(-1)
        opt_vec = proj.new_zeros(B, Nmax, self.hidden)
        opt_vec[opt["owner"], opt["slot"]] = (proj * gate).to(opt_vec.dtype)

        # [A]+[B] inject at [MASK] positions (padding slots add exactly zero)
        embeds = self.backbone.get_input_embeddings()(batch.input_ids)
        add = (opt_vec + self.type_embed(batch.qtype_idx).unsqueeze(1)) * batch.valid_mask.unsqueeze(-1).to(embeds.dtype)
        idx = batch.mask_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.hidden)
        inj = embeds.new_zeros(embeds.shape).scatter_add_(1, idx, add.to(embeds.dtype))

        # [C] backbone
        H = self.backbone(inputs_embeds=embeds + inj, attention_mask=batch.attention_mask).last_hidden_state

        # [D] readout
        kpm = ~batch.context_token_mask
        kpm = kpm & ~kpm.all(dim=-1, keepdim=True)  # empty context -> attend everywhere, no NaN
        codes, _ = self.code_attn(self.context_codes.unsqueeze(0).expand(B, -1, -1), H, H,
                                  key_padding_mask=kpm, need_weights=False)
        opt_states = torch.gather(H, 1, idx)
        head_seq = torch.cat([H[:, :1], codes, opt_states], 1)
        head_pad = torch.cat([torch.zeros(B, 1 + self.n_context_codes, dtype=torch.bool, device=H.device),
                              ~batch.valid_mask], 1)

        # [E] head: one pass
        s0 = self.entry_layer(head_seq, src_key_padding_mask=head_pad)
        x = s0 + s0 if self.head_input == "s0x2" else s0
        s = self.recurrent_block(x, key_padding_mask=head_pad)

        # [F] scorer
        off = 1 + self.n_context_codes
        logits = self.scorer(s[:, off:off + Nmax]).squeeze(-1)
        logits = logits.masked_fill(~batch.valid_mask, float("-inf"))
        if return_hidden:
            return logits, s
        return logits

    def load_exp7a(self, path, strict=True):
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        sd = ck.get("model_state", ck.get("model", ck))
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        return ck, missing, unexpected

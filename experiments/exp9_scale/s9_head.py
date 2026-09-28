"""Experiment 9 scaled decision head ("rethinking" head).

Everything the exp7 line explored, scaled up and put in one place:

  1. Context resampler (Perceiver-style), replacing 16 codes with one
     cross-attention: R latents (default 128) x L resampler layers, each layer
     = latents cross-attend to the backbone's prefix states, then self-attend,
     then FFN. Compresses an arbitrarily long state into R vectors the loop can
     afford to re-read every pass.

  2. Recurrent "rethinking" block, K passes (default 6), SHARED weights:
         s_0  = EntryLayer([prefix summary ; latents ; option readouts])
         x_k  = LN(s_{k-1} + g0 * s_0) + depth_embed[k]          (re-anchor)
         s_k  = LN(s_{k-1} + Block(x_k, memory_k))
     Block = D decoder layers (self-attn over the head sequence, cross-attn to
     memory, FFN). memory_k = [backbone prefix states ; option readouts ;
     scratchpad thoughts from passes < k] -- every pass can re-read the raw
     evidence AND what earlier passes concluded.

  3. Gated scratchpad: after each pass the head writes S thought vectors
     (attention-pooled from s_k with S learned write-queries) into a memory
     that later passes cross-attend to. A GRU-style gate lets each slot
     keep, overwrite or accumulate:  m_k = z * m_{k-1} + (1 - z) * write_k.
     Both the running memory and each pass's writes stay addressable.

  4. Every pass is scored (per-option MLP), so the forward returns K logit
     tensors -> depth loss + the accuracy-by-depth curve at eval. If depth
     helps at this scale, the curve rises; if not, it's flat and the extra
     passes can be dropped at inference (K is a runtime argument).

Width = backbone hidden size. With d=2560, R=128, resampler 2 layers, block 4
decoder layers, the head is ~0.55B params (cf. exp7a's 45M).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _heads(d, pref=16):
    for h in range(pref, 0, -1):
        if d % h == 0:
            return h
    return 1


class ResamplerLayer(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.xattn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.ns = nn.LayerNorm(d)
        self.sattn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.nf = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, lat, mem, mem_pad):
        k = self.nk(mem)
        lat = lat + self.xattn(self.nq(lat), k, k, key_padding_mask=mem_pad, need_weights=False)[0]
        h = self.ns(lat)
        lat = lat + self.sattn(h, h, h, need_weights=False)[0]
        return lat + self.ff(self.nf(lat))


class Resampler(nn.Module):
    def __init__(self, d, n_latents=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d) * d ** -0.5)
        self.layers = nn.ModuleList([ResamplerLayer(d, _heads(d), dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d)

    def forward(self, H, pad):
        lat = self.latents.unsqueeze(0).expand(H.size(0), -1, -1)
        # an all-padding row would NaN the softmax; unmask it (its output is unused anyway)
        pad = pad & ~pad.all(dim=-1, keepdim=True)
        for layer in self.layers:
            lat = layer(lat, H, pad)
        return self.out_norm(lat)


class Scratchpad(nn.Module):
    """S slots, written after every pass by attention pooling over the head
    state, gated GRU-style against the previous contents."""

    def __init__(self, d, n_slots=8, k_max=6, dropout=0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_slots, d) * d ** -0.5)
        self.norm = nn.LayerNorm(d)
        self.pool = nn.MultiheadAttention(d, _heads(d), dropout=dropout, batch_first=True)
        self.z = nn.Linear(2 * d, d)
        nn.init.constant_(self.z.bias, 1.0)     # start by mostly keeping memory -> stable early training
        self.slot_pos = nn.Embedding(k_max, d)
        nn.init.normal_(self.slot_pos.weight, std=0.02)

    def forward(self, s, pad, prev, depth):
        B = s.size(0)
        h = self.norm(s)
        w = self.pool(self.q.unsqueeze(0).expand(B, -1, -1), h, h, key_padding_mask=pad, need_weights=False)[0]
        w = w + self.slot_pos.weight[depth]
        if prev is None:
            return w
        z = torch.sigmoid(self.z(torch.cat([prev, w], -1)))
        return z * prev + (1 - z) * w


class RethinkHead(nn.Module):
    def __init__(self, d, n_latents=128, resampler_layers=2, block_layers=4, k_max=6, n_scratch=8,
                 dropout=0.1):
        super().__init__()
        h = _heads(d)
        self.k_max = k_max
        self.n_latents = n_latents
        self.ckpt = True   # per-pass activation checkpointing while training
        self.resampler = Resampler(d, n_latents, resampler_layers, dropout)
        self.entry = nn.TransformerEncoderLayer(d, h, 4 * d, dropout, activation="gelu",
                                                batch_first=True, norm_first=True)
        self.block = nn.ModuleList([nn.TransformerDecoderLayer(d, h, 4 * d, dropout, activation="gelu",
                                                               batch_first=True, norm_first=True)
                                    for _ in range(block_layers)])
        self.anchor_norm = nn.LayerNorm(d)
        self.out_norm = nn.LayerNorm(d)
        self.s0_gate = nn.Parameter(torch.tensor(0.5))
        self.depth_embed = nn.Embedding(k_max, d)
        nn.init.normal_(self.depth_embed.weight, std=0.02)
        self.scratch = Scratchpad(d, n_scratch, k_max, dropout) if n_scratch > 0 else None
        self.mem_norm = nn.LayerNorm(d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(d, 1))

    def forward(self, H, prefix_mask, prefix_last, opt_states, valid_mask, k=None):
        """H: (B, L, d) backbone states; prefix_mask: (B, L) bool; prefix_last: (B,);
        opt_states: (B, N, d) option readouts; valid_mask: (B, N).
        Returns a list of K logit tensors (B, N), -inf at padding."""
        k = k or self.k_max
        B, N, d = opt_states.shape
        ctx_pad = ~prefix_mask
        lat = self.resampler(H, ctx_pad)
        last = H.gather(1, prefix_last.view(B, 1, 1).expand(-1, 1, d))
        seq = torch.cat([last, lat, opt_states], 1)
        pad = torch.cat([torch.zeros(B, 1 + self.n_latents, dtype=torch.bool, device=H.device), ~valid_mask], 1)
        s0 = self.entry(seq, src_key_padding_mask=pad)

        # evidence memory: backbone prefix states (full resolution) + option readouts
        base_mem = self.mem_norm(torch.cat([H, opt_states], 1))
        base_pad = torch.cat([ctx_pad & ~ctx_pad.all(dim=-1, keepdim=True), ~valid_mask], 1)

        off = 1 + self.n_latents
        s, scratch, thoughts, out = s0, None, [], []
        for depth in range(k):
            x = s0 if depth == 0 else self.anchor_norm(s + self.s0_gate * s0)
            x = x + self.depth_embed.weight[depth]
            if thoughts:
                tm = torch.cat(thoughts, 1)
                mem = torch.cat([base_mem, tm], 1)
                mpad = torch.cat([base_pad, torch.zeros(B, tm.size(1), dtype=torch.bool, device=H.device)], 1)
            else:
                mem, mpad = base_mem, base_pad
            if self.training and self.ckpt:
                from torch.utils.checkpoint import checkpoint
                y = checkpoint(self._block, x, mem, pad, mpad, use_reentrant=False)
            else:
                y = self._block(x, mem, pad, mpad)
            s = self.out_norm(s + y)
            if self.scratch is not None:
                scratch = self.scratch(s, pad, scratch, depth)
                thoughts.append(scratch)
            logits = self.scorer(s[:, off:off + N]).squeeze(-1).float()
            out.append(logits.masked_fill(~valid_mask, float("-inf")))
        return out

    def _block(self, x, mem, pad, mpad):
        for layer in self.block:
            x = layer(x, mem, tgt_key_padding_mask=pad, memory_key_padding_mask=mpad)
        return x

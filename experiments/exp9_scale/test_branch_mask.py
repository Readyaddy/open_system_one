"""Exact-equivalence test for exp9's branch packing on a real Qwen3 backbone.

Packed pass: [prefix][branch_1][branch_2]...[branch_n], where branch tokens
attend to the whole prefix + causally within their own branch, and every
branch's position ids restart at len(prefix). Reference: each prefix+branch_i
run alone as an ordinary causal sequence. The last-token hidden state of
every branch must match its reference -- if it does, options are scored
independently of each other and of their order, by construction.
"""
import sys
import torch
from transformers import AutoModel, AutoTokenizer

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Embedding-0.6B"
dev = "cuda"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name, dtype=torch.float32).to(dev).eval()

prefix = tok("Instruct: pick the category.\nQuery: my card got stuck in the ATM machine", add_special_tokens=False)["input_ids"]
branches = [tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            for t in ["\nOption: card swallowed", "\nOption: lost or stolen card, please block it",
                      "\nOption: exchange rate"]]
P = len(prefix)
ids = list(prefix)
pos = list(range(P))
seg = [0] * P                      # 0 = prefix, i>0 = branch i
for i, b in enumerate(branches, 1):
    ids += b
    pos += list(range(P, P + len(b)))
    seg += [i] * len(b)
L = len(ids)
seg_t = torch.tensor(seg)
idx = torch.arange(L)
q_seg, k_seg = seg_t[:, None], seg_t[None, :]
causal = idx[None, :] <= idx[:, None]
allowed = causal & ((k_seg == 0) | (k_seg == q_seg))  # prefix visible to all; own branch only
mask4 = allowed[None, None].to(dev)

out = {}
with torch.no_grad():
    for kind, m in (("bool", mask4), ("float", torch.zeros(1, 1, L, L, device=dev).masked_fill(~mask4, torch.finfo(torch.float32).min))):
        try:
            h = model(input_ids=torch.tensor([ids], device=dev), attention_mask=m,
                      position_ids=torch.tensor([pos], device=dev)).last_hidden_state[0]
            out[kind] = h
        except Exception as e:
            print(f"{kind} 4D mask: FAILED {type(e).__name__}: {e}")
    ends, off = [], P
    for b in branches:
        off += len(b)
        ends.append(off - 1)
    for kind, h in out.items():
        worst = 0.0
        for i, b in enumerate(branches):
            ref = model(input_ids=torch.tensor([prefix + b], device=dev)).last_hidden_state[0, -1]
            d = (h[ends[i]] - ref).abs().max().item() / ref.abs().max().item()
            worst = max(worst, d)
        print(f"{kind} 4D mask: max relative diff of branch readouts vs independent runs = {worst:.2e}  "
              f"-> {'EXACT (packing is correct)' if worst < 1e-4 else 'MISMATCH'}")

"""Diagnostic: per-checkpoint accuracy AND predicted-class histogram on
ProofWriter-Long val, restricted to short padding so it runs in about a minute.

Usage: python diag_qdep_ckpt.py <ckpt> [<ckpt> ...]

The histogram is the point. A model that has collapsed to the answer prior
reports ~100% "Unknown" predictions regardless of the true label, and its
accuracy then equals the Unknown base rate of each hop bucket exactly.
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
import train_exp7d_hard_v2 as T  # installs the torchvision stub + fast option-encoding patches
import torch
from torch.amp import autocast

VAL = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                   "exp7_longctx_reasoning_corpus", "val.jsonl")
LABELS = ["True", "False", "Unknown"]


def bucket(q):
    return "0" if q == 0 else ("1" if q == 1 else ("2-3" if q <= 3 else "4+"))


def main(paths, n=400, max_tokens=512):
    dev = torch.device("cuda")
    tok = T.get_tokenizer(T.BACKBONE)
    builder = T.PackedSequenceBuilder(tok, budget_total=2048, l_context=1536,
                                      l_instructions=128, l_max_per_option=64)
    collate = T.PackedBatchCollate(builder, tok)

    rows = [json.loads(l) for l in open(VAL, encoding="utf-8")]
    rows = [r for r in rows if r["source"] == "proofwriter_long"
            and r.get("target_tokens", 99999) <= max_tokens]
    rows = random.Random(0).sample(rows, min(n, len(rows)))
    exs = [T.PackedExample(context=r["context"], instructions=r["instructions"],
                           option_texts=r["option_texts"], qtype="choice",
                           answer_idx=r["answer_idx"]) for r in rows]
    print(f"{len(exs)} short ProofWriter val examples (target_tokens <= {max_tokens})\n", flush=True)

    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck.get("model_state_dict", ck))
        kw = {}
        if any(k.startswith("scratch_proj") for k in sd):
            kw["n_scratch"] = sd["scratch_proj.1.weight"].shape[0] // sd["scratch_pos.weight"].shape[1]
        if "depth_scale.weight" in sd:
            kw["use_film_depth"] = True
        m = T.HybridDecisionModel(backbone=T.BACKBONE, mask_token_id=tok.mask_token_id,
                                  k_max=ck.get("k_max", 6), gradient_checkpointing=False, **kw)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        m.to(dev).eval()

        per = {}
        with torch.no_grad():
            for i in range(0, len(exs), 16):
                chunk = exs[i:i + 16]
                batch = collate(chunk).to_device(dev)
                with autocast("cuda", dtype=torch.bfloat16):
                    lp = m(tok, batch, dev)
                for d in (0, len(lp) - 1):
                    preds = lp[d].argmax(-1).tolist()
                    for r, ex, pr in zip(rows[i:i + 16], chunk, preds):
                        b = bucket(int(r["qdep"]))
                        slot = per.setdefault((d, b), {"ok": 0, "n": 0, "pred": Counter()})
                        slot["n"] += 1
                        slot["ok"] += int(pr == ex.answer_idx)
                        slot["pred"][ex.option_texts[pr]] += 1

        print(f"=== {os.path.basename(p)}  (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
        K = len(lp)
        for d in (0, K - 1):
            print(f"  depth k={d+1}", flush=True)
            for b in ("0", "1", "2-3", "4+"):
                s = per.get((d, b))
                if not s:
                    continue
                hist = " ".join(f"{L[:3]}={s['pred'][L]/s['n']*100:4.0f}%" for L in LABELS)
                print(f"    hops {b:4s} n={s['n']:3d}  acc={s['ok']/s['n']*100:5.1f}%   predicted: {hist}", flush=True)
        print(flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])

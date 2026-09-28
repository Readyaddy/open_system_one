"""Experiment 9 evaluation.

Reuses exp8's whole evaluation (s1_eval.run_suite: per-group acc / blind-hard /
NLL / Brier / ECE / RPS, family/qtype/cardinality aggregates, order flip,
fitted temperatures, decision utility, official-protocol JevBench groups) by
swapping in an exp9 `predict` that returns the identical record format.

Adds the experiment's headline measurement -- does injecting the option vector
at the MARK slot work? -- as a modality x cardinality sweep on held-out data:

    sweep/<set>@N<n>/<mode>/acc     mode in {both, text, vector}
      text   = option text in the branch, vector path gated off
      vector = branch is [MARK][EOS] only: the injected embedding is the ONLY
               thing that says which option this is
      both   = the deployed setting

Sets: Banking77 (never trained on, 71 labels) at N = 4 / 16 / 71, and
zero-shot intents at N = 50. If vector-only sits at chance, the injection
carries nothing; if `both` beats `text` at high N, it is earning its place.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.amp import autocast

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp8_s1_rlcd"))
import s1_data as SD  # noqa: E402
import s1_eval as SE  # noqa: E402
import rlcd  # noqa: E402
from s9_model import Packer9  # noqa: E402

LONG_MAX_TOKENS = 16384


@torch.no_grad()
def predict9(model, packer, pairs, device, max_tokens=24000):
    """Same contract and record format as exp8's s1_eval.predict."""
    model.eval()
    records = [None] * len(pairs)
    tagged = []
    for i, (e, m) in enumerate(pairs):
        m.extra = {**m.extra, "_i": i}
        tagged.append((e, m))
    t_by_bucket = defaultdict(lambda: [0.0, 0])
    for batch, metas in packer.pack(tagged, max_tokens=max_tokens, max_options=4096):
        b = batch.to(device)
        torch.cuda.synchronize()
        t0 = time.time()
        with autocast("cuda", dtype=torch.bfloat16):
            outs = model(b, all_depths=True)
        logits = outs[-1]
        depth_pred = torch.stack([o.float().argmax(-1) for o in outs], 0)   # (K, B)
        torch.cuda.synchronize()
        dt = time.time() - t0
        logits = logits.float()
        logp = F.log_softmax(logits, -1)
        p = logp.exp()
        y = b.answer_idx
        R, comps = rlcd.proper_reward(logits, y, b.qtype_idx)
        pred = p.argmax(-1)
        for j, m in enumerate(metas):
            n = m.n_options
            pj = p[j, :n]
            rec = {"correct": int(pred[j] == y[j]), "conf": float(pj.max()), "p_true": float(pj[y[j]]),
                   "nll": float(-logp[j, y[j]]), "brier": float(((pj - F.one_hot(y[j], n).float()) ** 2).sum()),
                   "n_options": n, "qtype": m.qtype, "blind_easy": m.blind_easy, "family": m.family,
                   "group": m.group, "reward": float(R[j]), "y": int(y[j]), "pred": int(pred[j]),
                   "logits": logits[j, :n].tolist(),
                   "depth_correct": [int(depth_pred[kk, j] == y[j]) for kk in range(depth_pred.size(0))]}
            if m.qtype == "score":
                lv = torch.arange(n, device=pj.device, dtype=pj.dtype)
                rev = m.extra.get("reversed", False)
                to_level = (lambda k: n - 1 - k) if rev else (lambda k: k)
                exp_idx = float((pj * lv).sum())
                rec.update({"rps": float(comps["rps"][j]), "true_level": to_level(int(y[j])),
                            "pred_level": to_level(int(pred[j])),
                            "exp_level": (n - 1 - exp_idx) if rev else exp_idx})
            records[m.extra["_i"]] = rec
            bk = rlcd.card_bucket(n)
            t_by_bucket[bk][0] += dt / len(metas)
            t_by_bucket[bk][1] += 1
    return records, {k: 1000 * t / c for k, (t, c) in t_by_bucket.items() if c}


# exp8's run_suite / order_flip_rate call the module-level `predict`.
SE.predict = predict9


def long_packer9(tok):
    """Official JevBench protocol: full states (hard tier reaches ~3.5k tokens)
    and criteria-length options -- same budgets as exp8's long_context_packer."""
    return Packer9(tok, budget_total=8192, l_context=6400, l_instructions=160, l_max_per_option=128,
                   max_tokens=LONG_MAX_TOKENS, max_options=2048)


def build_sweep(C, cfg, n=300, seed=99):
    rng = random.Random(seed)
    full = SD.DataCfg(**{**cfg.__dict__, "p_vec_only": 0.0, "p_text_only": 0.0})
    it = C["intent"]
    b77 = list(it["banking77"])
    random.Random(seed).shuffle(b77)
    zs = list(it["zero_shot"])
    random.Random(seed).shuffle(zs)
    sets = {f"banking77@N{k}": [(r, C["banking77_labels"], k) for r in b77[:n]] for k in (4, 16, 71)}
    sets["zero_shot@N50"] = [(r, it["all_labels"], 50) for r in zs[:n]]
    G = {}
    for name, rows in sets.items():
        base = [SD.build_intent(r["text"], r["label"], pool, random.Random(seed + i), full, n=k,
                                instructions="Which category should this be routed to?", family="sweep",
                                source=name)
                for i, (r, pool, k) in enumerate(rows)]
        for mode, (ut, uv) in {"both": (True, True), "text": (True, False), "vector": (False, True)}.items():
            xs = []
            for e, m in base:
                e2 = SD.PackedExample(context=e.context, instructions=e.instructions, option_texts=e.option_texts,
                                      qtype=e.qtype, answer_idx=e.answer_idx, use_text=ut, use_vector=uv)
                xs.append((e2, SD.Meta("sweep", name, e.qtype, m.n_options, group=f"sweep/{name}/{mode}")))
            G[f"{name}/{mode}"] = xs
    return G


def run_sweep(model, packer, sweep, device):
    out = {}
    for g, pairs in sweep.items():
        recs, _ = predict9(model, packer, pairs, device)
        out[f"sweep/{g}/acc"] = sum(r["correct"] for r in recs) / len(recs)
        out[f"sweep/{g}/chance"] = sum(1 / r["n_options"] for r in recs) / len(recs)
    return out


def print_depth(flat):
    ks = sorted(int(k.split("/")[1]) for k in flat if k.startswith("depth/") and k.endswith("/acc"))
    if ks:
        print("  depth curve: " + "  ".join(f"k={k} {flat[f'depth/{k}/acc']*100:.1f}" for k in ks)
              + f"   (last-first {flat['depth/gain_last_minus_first']*100:+.1f})", flush=True)


def print_sweep(flat):
    names = sorted({k.split("/")[1] for k in flat if k.startswith("sweep/")})
    print(f"  {'sweep set':18s} {'both':>7s} {'text':>7s} {'vector':>7s} {'chance':>7s}", flush=True)
    for nm in names:
        g = lambda mode: flat.get(f"sweep/{nm}/{mode}/acc", float("nan")) * 100
        print(f"  {nm:18s} {g('both'):7.1f} {g('text'):7.1f} {g('vector'):7.1f} "
              f"{flat.get(f'sweep/{nm}/both/chance', 0) * 100:7.1f}", flush=True)


def evaluate_all(model, packer, suite, sweep, device, tok, with_order=True):
    flat, side = SE.run_suite(model, packer, suite, device, with_order=with_order,
                              long_packer=long_packer9(tok))
    flat.update(run_sweep(model, packer, sweep, device))
    flat.update(depth_curve(model, packer, suite, device))
    return flat, side


def depth_curve(model, packer, suite, device, per_group=60):
    """Accuracy of each rethinking pass on the in-distribution selection groups.
    Rising = the extra passes are doing work; flat = they could be dropped."""
    pairs = [x for g in SD.selection_groups(suite) for x in suite[g][:per_group]]
    recs, _ = predict9(model, packer, pairs, device)
    K = len(recs[0]["depth_correct"])
    out = {f"depth/{k + 1}/acc": sum(r["depth_correct"][k] for r in recs) / len(recs) for k in range(K)}
    out["depth/gain_last_minus_first"] = out[f"depth/{K}/acc"] - out["depth/1/acc"]
    return out


def main():
    from s9_model import S9DecisionModel, get_tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="exp9 checkpoint (trainable-state file)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    tok = get_tokenizer(a["backbone"])
    model = S9DecisionModel(a["backbone"], lora_r=a["lora_r"], lora_alpha=a["lora_alpha"],
                            full_finetune=a["full_finetune"], n_codes=a["n_codes"],
                            gradient_checkpointing=False, head=a.get("head", "simple"),
                            n_latents=a.get("n_latents", 128), resampler_layers=a.get("resampler_layers", 2),
                            block_layers=a.get("block_layers", 4), k_max=a.get("k_max", 6),
                            n_scratch=a.get("n_scratch", 8))
    model.load_trainable_state_dict(ck["model_state"])
    dev = torch.device("cuda")
    model.to(dev)
    C = SD.load_corpora()
    cfg = SD.DataCfg()
    suite = SD.build_val_suite(C, cfg, {"per_source": a.get("val_per_source", 300)}, with_bool_desc=True)
    packer = Packer9(tok, budget_total=a["budget_total"], l_context=a["l_context"])
    flat, side = evaluate_all(model, packer, suite, build_sweep(C, cfg), dev, tok)
    SE.print_report(flat, os.path.basename(args.ckpt))
    print_sweep(flat)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"metrics": flat, **side}, f, indent=1)


if __name__ == "__main__":
    main()

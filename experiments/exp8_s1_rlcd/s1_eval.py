"""Exp8 evaluation: every val group, every metric, one pass.

run_suite() returns a flat dict ready for W&B / JSON:
  val/<group>/{acc, acc_blind_hard, nll, brier, ece, mean_conf, chance, [rps, within1, mae_expected_level]}
  agg/family/<family>/..., agg/qtype/<qtype>/..., agg/card/<bucket>/...
  select/score                    macro acc over selection groups (model selection)
  order/flip_rate                 fraction of predictions that change when options are re-shuffled
  utility/<cost>/...              answer-or-escalate decision utility on the selection groups
  latency/ms_per_example/<bucket>
and a side payload (reliability bins, fitted temperatures) written to JSON only.

Usage (standalone): python s1_eval.py --ckpt <path> [--head_input s0x2]
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

sys.path.insert(0, os.path.dirname(__file__))
from s1_model import S1DecisionModel, PackedSequenceBuilder, PackedExample, get_tokenizer, BACKBONE  # noqa: E402
import s1_data as SD  # noqa: E402
import rlcd  # noqa: E402


@torch.no_grad()
def predict(model, packer, pairs, device, max_tokens=24000):
    """Returns per-example records, in input order."""
    model.eval()
    packer_eval = SD.Packer(packer.builder, packer.tok, max_tokens=max_tokens, max_options=2048)
    order = list(range(len(pairs)))
    records = [None] * len(pairs)
    # pack() sorts by length, so keep the original index in Meta.extra
    tagged = []
    for i, (e, m) in enumerate(pairs):
        m.extra = {**m.extra, "_i": i}
        tagged.append((e, m))
    t_by_bucket = defaultdict(lambda: [0.0, 0])
    for mb in packer_eval.pack(tagged):
        packed, opt = mb.to(device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        with autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(packed, opt)
        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = time.time() - t0
        logits = logits.float()
        logp = F.log_softmax(logits, -1)
        p = logp.exp()
        y = packed.answer_idx
        R, comps = rlcd.proper_reward(logits, y, packed.qtype_idx)
        pred = p.argmax(-1)
        for j, m in enumerate(mb.metas):
            n = m.n_options
            pj = p[j, :n]
            rec = {"correct": int(pred[j] == y[j]), "conf": float(pj.max()), "p_true": float(pj[y[j]]),
                   "nll": float(-logp[j, y[j]]), "brier": float(((pj - F.one_hot(y[j], n).float()) ** 2).sum()),
                   "n_options": n, "qtype": m.qtype, "blind_easy": m.blind_easy, "family": m.family,
                   "group": m.group, "reward": float(R[j]), "y": int(y[j]), "pred": int(pred[j]),
                   "logits": logits[j, :n].tolist()}
            if m.qtype == "score":
                lv = torch.arange(n, device=pj.device, dtype=pj.dtype)
                rev = m.extra.get("reversed", False)
                to_level = (lambda k: n - 1 - k) if rev else (lambda k: k)
                exp_idx = float((pj * lv).sum())
                rec.update({"rps": float(comps["rps"][j]), "true_level": to_level(int(y[j])),
                            "pred_level": to_level(int(pred[j])),
                            "exp_level": (n - 1 - exp_idx) if rev else exp_idx})
            records[m.extra["_i"]] = rec
            b = rlcd.card_bucket(n)
            t_by_bucket[b][0] += dt / len(mb.metas)
            t_by_bucket[b][1] += 1
    lat = {b: 1000 * t / c for b, (t, c) in t_by_bucket.items() if c}
    return records, lat


def order_flip_rate(model, packer, pairs, device, n=300, seed=5):
    """Re-shuffle option order (non-ordinal questions only); fraction of
    examples whose predicted OPTION TEXT changes. A position-biased model
    flips often; an order-invariant one never does."""
    rng = random.Random(seed)
    base = [(e, m) for e, m in pairs if e.qtype != "score"]
    base = rng.sample(base, min(n, len(base)))
    shuffled = []
    for e, m in base:
        perm = list(range(len(e.option_texts)))
        rng.shuffle(perm)
        opts = [e.option_texts[i] for i in perm]
        shuffled.append((PackedExample(context=e.context, instructions=e.instructions, option_texts=opts,
                                       qtype=e.qtype, answer_idx=perm.index(e.answer_idx)),
                         SD.Meta(m.family, m.source, m.qtype, m.n_options, m.blind_easy, m.group)))
    r0, _ = predict(model, packer, base, device)
    r1, _ = predict(model, packer, shuffled, device)
    flips = sum(base[i][0].option_texts[a["pred"]] != shuffled[i][0].option_texts[b["pred"]]
                for i, (a, b) in enumerate(zip(r0, r1)))
    return flips / max(1, len(base))


LONG_CTX_PREFIXES = ("jevbench_",)


def long_context_packer(tok, max_tokens=8192, max_options=2048):
    """Packer for the official JevBench protocol: full state (hard-tier states
    reach ~3.5k tokens; the training packer cuts at 768, which dropped 60-70% of
    every long_policy / multi_hop document) and room for criteria-length options."""
    builder = PackedSequenceBuilder(tok, budget_total=8192, l_context=6400, l_instructions=160,
                                    l_max_per_option=128)
    return SD.Packer(builder, tok, max_tokens=max_tokens, max_options=max_options)


def run_suite(model, packer, suite, device, with_order=True, with_temps=True, long_packer=None):
    flat, side = {}, {"reliability": {}, "temperatures": {}}
    all_recs = []
    t0 = time.time()
    lat_all = defaultdict(list)
    for g, pairs in suite.items():
        if long_packer is not None and g.startswith(LONG_CTX_PREFIXES):
            recs, lat = predict(model, long_packer, pairs, device, max_tokens=long_packer.max_tokens)
        else:
            recs, lat = predict(model, packer, pairs, device)
        for b, v in lat.items():
            lat_all[b].append(v)
        s = rlcd.summarize(recs)
        side["reliability"][g] = s.pop("_reliability")
        for k, v in s.items():
            flat[f"val/{g}/{k}"] = v
        all_recs += recs

    def agg(key_fn, prefix):
        by = defaultdict(list)
        for r in all_recs:
            k = key_fn(r)
            if k:
                by[k].append(r)
        for k, rs in by.items():
            s = rlcd.summarize(rs)
            s.pop("_reliability", None)
            for m in ("acc", "acc_blind_hard", "nll", "brier", "ece", "mean_conf", "chance", "n", "rps", "within1"):
                if m in s:
                    flat[f"agg/{prefix}/{k}/{m}"] = s[m]

    in_dist = lambda r: r["family"] in ("intent", "mcq", "bool", "score", "diversity")
    agg(lambda r: r["family"], "family")
    agg(lambda r: r["qtype"] if in_dist(r) else None, "qtype")
    agg(lambda r: rlcd.card_bucket(r["n_options"]) if in_dist(r) else None, "card")

    sel = SD.selection_groups(suite)
    flat["select/score"] = sum(flat[f"val/{g}/acc"] for g in sel) / len(sel)
    flat["select/score_blind_hard"] = sum(flat.get(f"val/{g}/acc_blind_hard", flat[f"val/{g}/acc"]) for g in sel) / len(sel)
    sel_recs = [r for r in all_recs if r["group"] in sel]
    for c, d in rlcd.utility_curve(sel_recs).items():
        for k, v in d.items():
            flat[f"utility/{c}/{k}"] = v
    for b, v in lat_all.items():
        flat[f"latency/ms_per_example/{b}"] = sum(v) / len(v)

    if with_temps:
        by = defaultdict(list)
        for r in sel_recs:
            by[f"{r['qtype']}:{rlcd.card_bucket(r['n_options'])}"].append(r)
        side["temperatures"] = rlcd.fit_temperatures(by)
    if with_order:
        pool = [x for g in sel if g.split("/")[0] in ("mcq", "diversity", "intent") for x in suite[g]]
        flat["order/flip_rate"] = order_flip_rate(model, packer, pool, device)
    flat["eval/seconds"] = time.time() - t0
    return flat, side


def print_report(flat, title=""):
    print(f"\n=== {title} ===", flush=True)
    groups = sorted({k.split("/")[1] + "/" + k.split("/")[2] for k in flat if k.startswith("val/")})
    print(f"  {'group':34s} {'n':>5s} {'acc':>6s} {'hard':>6s} {'chance':>6s} {'nll':>6s} {'ece':>6s} {'conf':>6s} {'extra':>s}")
    for g in groups:
        f = lambda m: flat.get(f"val/{g}/{m}")
        hard = f("acc_blind_hard")
        extra = f"rps {f('rps'):.3f} w1 {f('within1'):.2f}" if f("rps") is not None else ""
        print(f"  {g:34s} {f('n'):5d} {f('acc')*100:6.1f} {'' if hard is None else f'{hard*100:6.1f}':>6s} "
              f"{f('chance')*100:6.1f} {f('nll'):6.3f} {f('ece')*100:6.1f} {f('mean_conf')*100:6.1f} {extra}", flush=True)
    fam = sorted({k.split("/")[2] for k in flat if k.startswith("agg/family/")})
    print("  families: " + "  ".join(f"{x} {flat[f'agg/family/{x}/acc']*100:.1f}" for x in fam))
    print(f"  SELECT score {flat['select/score']*100:.2f}  (blind-hard {flat['select/score_blind_hard']*100:.2f})"
          + (f"   order flip {flat['order/flip_rate']*100:.1f}%" if "order/flip_rate" in flat else "")
          + f"   eval {flat['eval/seconds']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head_input", default="s0")
    ap.add_argument("--per_source", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    tok = get_tokenizer(BACKBONE)
    model = S1DecisionModel(mask_token_id=tok.mask_token_id, gradient_checkpointing=False, head_input=args.head_input)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(ck.get("model_state", ck.get("model", ck)), strict=True)
    model.to(dev)
    builder = PackedSequenceBuilder(tok, budget_total=2048, l_context=768, l_instructions=96, l_max_per_option=64)
    C = SD.load_corpora()
    suite = SD.build_val_suite(C, SD.DataCfg(), {"per_source": args.per_source})
    flat, side = run_suite(model, SD.Packer(builder, tok), suite, dev, long_packer=long_context_packer(tok))
    print_report(flat, os.path.basename(args.ckpt))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"metrics": flat, **side}, f, indent=1)


if __name__ == "__main__":
    main()

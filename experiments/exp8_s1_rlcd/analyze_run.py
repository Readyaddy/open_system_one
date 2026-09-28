"""Post-run analysis for an exp8 run, from its eval_*.json files (no GPU).

Prints, per eval step:
  select       macro acc over in-distribution groups (what best.pt tracks)
  hard         same, on blind-hard subsets
  heldout      macro acc over held-out groups that never influence selection:
               intent/zero_shot@50, banking77/routing, external/ag_news,
               external/dair_emotion, typed/choice, typed/bool, typed/score
  per-family accuracy, calibration (ECE) on held-out, order flip, utility
Then names the best step by each criterion and which snapshot files exist.

Usage: python analyze_run.py <run_dir> [--baseline <eval_000000.json>] [--out summary.json]
"""
import argparse
import glob
import json
import os

HELDOUT = ["val/intent/zero_shot@50", "val/banking77/routing", "val/external/ag_news",
           "val/external/dair_emotion", "val/typed/choice", "val/typed/bool", "val/typed/score"]
FAMS = ["intent", "mcq", "bool", "score", "diversity"]


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("step", 0), d["metrics"]


def row(step, m):
    r = {"step": step, "select": m["select/score"], "hard": m["select/score_blind_hard"],
         "heldout": sum(m[f"{g}/acc"] for g in HELDOUT) / len(HELDOUT),
         "heldout_ece": sum(m[f"{g}/ece"] for g in HELDOUT) / len(HELDOUT),
         "flip": m.get("order/flip_rate"), "util3": m.get("utility/cw3/best_utility")}
    for f in FAMS:
        r[f] = m.get(f"agg/family/{f}/acc")
    for g in HELDOUT:
        r[g.split("/", 1)[1]] = m[f"{g}/acc"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    evals = sorted(glob.glob(os.path.join(a.run_dir, "eval_*.json")))
    rows = [row(*load(p)) for p in evals]
    if a.baseline and not any(r["step"] == 0 for r in rows):
        rows.insert(0, row(*load(a.baseline)))

    cols = ["select", "hard", "heldout", "heldout_ece"] + FAMS + ["flip", "util3"]
    print(f"{'step':>5s} " + " ".join(f"{c[:8]:>8s}" for c in cols))
    for r in rows:
        print(f"{r['step']:5d} " + " ".join(
            f"{r[c]:8.3f}" if c == "util3" else f"{r[c]*100:8.2f}" for c in cols))

    print("\nheld-out groups by step:")
    hk = [g.split("/", 1)[1] for g in HELDOUT]
    print(f"{'step':>5s} " + " ".join(f"{k.split('/')[-1][:12]:>12s}" for k in hk))
    for r in rows:
        print(f"{r['step']:5d} " + " ".join(f"{r[k]*100:12.2f}" for k in hk))

    trained = [r for r in rows if r["step"] > 0]
    best_sel = max(trained, key=lambda r: r["select"])
    best_ho = max(trained, key=lambda r: r["heldout"])
    base = next((r for r in rows if r["step"] == 0), None)
    print(f"\nbest in-distribution: step {best_sel['step']}  select {best_sel['select']*100:.2f}  "
          f"heldout {best_sel['heldout']*100:.2f}")
    print(f"best held-out:        step {best_ho['step']}  select {best_ho['select']*100:.2f}  "
          f"heldout {best_ho['heldout']*100:.2f}")
    if base:
        print(f"baseline (exp7a):     step 0     select {base['select']*100:.2f}  heldout {base['heldout']*100:.2f}")
    snaps = {int(os.path.basename(p)[9:-3]): p for p in glob.glob(os.path.join(a.run_dir, "snap_step*.pt"))}
    for name, r in (("in-distribution", best_sel), ("held-out", best_ho)):
        where = snaps.get(r["step"]) or ("best.pt" if r is best_sel else "no snapshot for this step")
        print(f"  weights for best {name}: {where}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows, "best_select_step": best_sel["step"], "best_heldout_step": best_ho["step"]},
                      f, indent=1)


if __name__ == "__main__":
    main()

"""Bias reweighting for corpora whose shortcuts survive AFLite filtering.

Measured on exp7_bool_corpus (2026-09-24): after AFLite drove the blind
model to chance INSIDE the filtered pool, a blind model trained on that
filtered train still scored 58.6% (MNLI hypothesis-only), 59/61.5% (BoolQ)
and 66.5% (QQP) on the natural val split. Dropping examples hid the cue
from the filter's own cross-fit; it did not stop the cue from being learnable.

Reweighting instead (Schuster et al. 2019; Utama et al. 2020 "example
reweighting"): keep every example, and give it sampling weight
    w = 1 / p_blind(gold)
from a cross-fitted blind model (averaged over its blind views), clipped
and normalized to mean 1 per source. Under that sampling distribution
the blind features carry no information about the label in expectation,
so there is no gradient toward the shortcut. Verified by
audit_exp7_data.py, which fits its blind baselines WITH these weights.

Writes `bias_w` into each train row (val/test untouched; they keep the
`blind_easy` tags from filter_exp7_shortcuts.py for blind-hard reporting).
data.py samples training rows proportionally to `bias_w`.

Usage: python scripts/weight_exp7_bias.py --corpus bool [--from_unfiltered]
"""
import argparse
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SEED = 20260924
K = 5
CLIP = (0.25, 4.0)

VIEWS = {
    "bool": {"mnli": ["question"], "paws": [], "boolq": ["question", "context"], "qqp": ["question", "context"]},
}
LABEL = {"bool": lambda r: r["answer_idx"]}


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def crossfit_gold_prob(rows, field, label_fn, rng):
    """p(gold) for each row from a blind TF-IDF LR that never saw that row."""
    y = np.array([label_fn(r) for r in rows])
    classes = sorted(set(y.tolist()))
    idx = np.arange(len(rows)); rng.shuffle(idx)
    folds = np.array_split(idx, K)
    p = np.zeros(len(rows))
    for k in range(K):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(K) if j != k])
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform([rows[i][field] for i in tr]), y[tr])
        pr = clf.predict_proba(vec.transform([rows[i][field] for i in te]))
        col = {c: j for j, c in enumerate(clf.classes_)}
        p[te] = [pr[n, col[y[i]]] for n, i in enumerate(te)]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="bool", choices=list(VIEWS))
    ap.add_argument("--from_unfiltered", action="store_true",
                    help="reweight the unfiltered train instead of the AFLite-filtered one")
    args = ap.parse_args()
    d = os.path.join(DATA_DIR, f"exp7_{args.corpus}_corpus")
    src_path = os.path.join(d, "train.unfiltered.jsonl" if args.from_unfiltered else "train.jsonl")
    rows = read(src_path)
    label_fn = LABEL[args.corpus]
    rng = np.random.RandomState(SEED)
    report = {}

    for src in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        sub = [rows[i] for i in idx]
        fields = VIEWS[args.corpus].get(src, [])
        if not fields:
            for i in idx:
                rows[i]["bias_w"] = 1.0
            report[src] = {"n": len(sub), "views": []}
            print(f"  {src:10s} n={len(sub):6d}  no flagged views -> w=1", flush=True)
            continue
        pg = np.mean([crossfit_gold_prob(sub, f, label_fn, rng) for f in fields], axis=0)
        w = np.clip(1.0 / np.maximum(pg, 1e-3), *CLIP)
        w = w / w.mean()
        for i, wi in zip(idx, w):
            rows[i]["bias_w"] = round(float(wi), 4)
        ess = (w.sum() ** 2) / (w ** 2).sum()  # effective sample size under the weights
        report[src] = {"n": len(sub), "views": fields, "mean_p_gold": round(float(pg.mean()), 4),
                       "w_min": round(float(w.min()), 3), "w_max": round(float(w.max()), 3),
                       "effective_n": int(ess)}
        print(f"  {src:10s} n={len(sub):6d}  views={fields}  mean p_blind(gold)={pg.mean():.3f}  "
              f"w in [{w.min():.2f},{w.max():.2f}]  effective n={int(ess)}", flush=True)

    with open(os.path.join(d, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(d, "bias_weights_report.json"), "w", encoding="utf-8") as f:
        json.dump({"source_file": os.path.basename(src_path), **report}, f, indent=2)
    print(f"-> {d}/train.jsonl (bias_w added, {len(rows)} rows)")


if __name__ == "__main__":
    main()

"""AFLite-style adversarial filtering of shortcut-solvable training examples
(Le Bras et al. 2020 / WinoGrande), run after build_exp7_data.py.

audit_exp7_data.py measures whether a BLIND model (one that can't see the
part of the input the task depends on) beats chance. This script removes
what that blind model can exploit:

  train:     repeated K-fold cross-fitting of the blind model. Each example's
             "predictability" = fraction of held-out predictions that were
             correct. Examples above TAU are dropped, a slice per round,
             until held-out blind accuracy is within MARGIN of chance. Labels
             are then re-balanced.
  val/test:  never dropped (that would make val easier to interpret but
             would hide the cue). Each row gets `blind_easy: [views...]`,
             the blind views (fit on the ORIGINAL train) that answer it
             correctly -- training reports accuracy on the blind-hard subset
             separately, so "learned the task" and "learned the cue" can be
             told apart during training.

Blind views per corpus:
  mcq    options_only (option text alone, argmax over options) + option length
  bool   question_only, context_only
  score  length_only
  choice length_only (diversity)

Writes data/exp7_mcq_corpus/ (new; the original data/mcq_corpus is untouched)
and rewrites exp7_bool/score/diversity in place (originals kept as *.unfiltered.jsonl).

Usage: python scripts/filter_exp7_shortcuts.py [--only mcq,bool,score,diversity]
"""
import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SEED = 20260924
K_FOLDS = 5
N_PARTITIONS = 4
TAU = 0.75            # drop if blind model right on >75% of its held-out predictions
MARGIN = 0.02         # stop when held-out blind acc <= chance + 2 points
MAX_ROUNDS = 6
MAX_DROP_FRAC = 0.5   # never drop more than half a source


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write(rows, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Blind views: each is (fit(train_rows) -> model, predict(model, rows) -> correct flags)
# --------------------------------------------------------------------------

def _len_feats(xs):
    return np.array([[math.log1p(len(x)), math.log1p(len(x.split()))] for x in xs])


class TextView:
    """Blind view over one text field -> row label."""

    def __init__(self, field, label_fn, kind="tfidf"):
        self.field, self.label_fn, self.kind = field, label_fn, kind

    def fit(self, rows):
        xs, ys = [r[self.field] for r in rows], [self.label_fn(r) for r in rows]
        if len(set(ys)) < 2:
            return None
        if self.kind == "len":
            return (None, LogisticRegression(max_iter=2000).fit(_len_feats(xs), ys))
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
        return (vec, LogisticRegression(max_iter=2000).fit(vec.fit_transform(xs), ys))

    def correct(self, model, rows):
        if model is None:
            return np.zeros(len(rows), bool)
        vec, clf = model
        xs = [r[self.field] for r in rows]
        X = _len_feats(xs) if vec is None else vec.transform(xs)
        return clf.predict(X) == np.array([self.label_fn(r) for r in rows])


class OptionsView:
    """MCQ: score each option from its own text (+ its length rank) only."""

    def fit(self, rows):
        X, Y, L = [], [], []
        for r in rows:
            lens = [len(o) for o in r["options"]]
            for i, o in enumerate(r["options"]):
                X.append(o); Y.append(int(i == r["answer_idx"]))
                L.append([lens[i] == max(lens), lens[i] == min(lens)])
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
        from scipy.sparse import hstack, csr_matrix
        Xm = hstack([vec.fit_transform(X), csr_matrix(np.array(L, float))]).tocsr()
        return vec, LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xm, Y)

    def correct(self, model, rows):
        # One vectorized transform over every option, then argmax per question.
        from scipy.sparse import hstack, csr_matrix
        vec, clf = model
        texts, L, bounds = [], [], []
        for r in rows:
            lens = [len(o) for o in r["options"]]
            bounds.append((len(texts), len(texts) + len(lens)))
            texts.extend(r["options"])
            L.extend([[l == max(lens), l == min(lens)] for l in lens])
        s = clf.decision_function(hstack([vec.transform(texts), csr_matrix(np.array(L, float))]).tocsr())
        return np.array([int(np.argmax(s[a:b])) == r["answer_idx"] for (a, b), r in zip(bounds, rows)])


# --------------------------------------------------------------------------
# AFLite
# --------------------------------------------------------------------------

def heldout_predictability(rows, views, rng):
    hits = np.zeros(len(rows)); n = np.zeros(len(rows))
    per_view_acc = defaultdict(list)
    for _ in range(N_PARTITIONS):
        idx = np.arange(len(rows)); rng.shuffle(idx)
        folds = np.array_split(idx, K_FOLDS)
        for k in range(K_FOLDS):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(K_FOLDS) if j != k])
            tr_rows = [rows[i] for i in tr]; te_rows = [rows[i] for i in te]
            any_right = np.zeros(len(te), bool)
            for name, v in views.items():
                c = v.correct(v.fit(tr_rows), te_rows)
                per_view_acc[name].append(c.mean())
                any_right |= c
            hits[te] += any_right; n[te] += 1
    return hits / np.maximum(n, 1), {k: float(np.mean(v)) for k, v in per_view_acc.items()}


def rebalance(rows, label_fn, rng):
    by = defaultdict(list)
    for r in rows:
        by[label_fn(r)].append(r)
    cap = min(len(v) for v in by.values())
    cap = max(cap, 50)
    out = []
    for v in by.values():
        rng.shuffle(v); out.extend(v[:cap])
    rng.shuffle(out)
    return out


def aflite(rows, views, chance, label_fn, rng, name, balance=True):
    n0 = len(rows)
    log = []
    prev = None
    for rnd in range(MAX_ROUNDS):
        if len(rows) < 200:
            break
        pred, accs = heldout_predictability(rows, views, rng)
        log.append({"round": rnd, "n": len(rows), **{k: round(v, 4) for k, v in accs.items()}})
        print(f"      {name} round {rnd}: n={len(rows)} blind acc {', '.join(f'{k}={v*100:.1f}' for k, v in accs.items())} (chance {chance*100:.1f})", flush=True)
        # Two-sided: a blind view BELOW chance is also a cue (predict the
        # opposite of it). First run over-filtered MNLI's premise view to
        # 40.2% on 2-way; if a round overshoots, revert it and stop.
        if prev is not None and min(accs.values()) < chance - MARGIN:
            rows = prev
            log.append({"reverted_round": rnd})
            print(f"      {name}: round {rnd} overshot below chance -> reverted", flush=True)
            break
        if max(accs.values()) <= chance + MARGIN or len(rows) <= n0 * (1 - MAX_DROP_FRAC):
            break
        prev = rows
        order = np.argsort(-pred)
        # 5% slices (was 10%) so the last round can't overshoot far past chance.
        n_drop = min(int(0.05 * len(rows)), int((pred > TAU).sum()), len(rows) - int(n0 * (1 - MAX_DROP_FRAC)))
        if n_drop <= 0:
            break
        drop = set(order[:n_drop].tolist())
        rows = [r for i, r in enumerate(rows) if i not in drop]
        if balance:
            rows = rebalance(rows, label_fn, rng)
    return rows, log


def tag_eval(train_rows, eval_rows, views):
    models = {k: v.fit(train_rows) for k, v in views.items()}
    flags = {k: views[k].correct(m, eval_rows) for k, m in models.items()}
    for i, r in enumerate(eval_rows):
        r["blind_easy"] = [k for k in views if flags[k][i]]
    return {k: round(float(f.mean()), 4) for k, f in flags.items()}


# --------------------------------------------------------------------------

def run_corpus(name, splits, source_of, views_fn, chance_fn, label_fn, balance=True):
    rng = np.random.RandomState(SEED)
    prng = random.Random(SEED)
    report = {}
    out = {"train": [], "val": [], "test": []}
    for src in sorted({source_of(r) for r in splits["train"]}):
        tr = [r for r in splits["train"] if source_of(r) == src]
        views = views_fn(src)
        if not views:
            out["train"] += tr
            for s in ("val", "test"):
                out[s] += [r for r in splits[s] if source_of(r) == src]
            continue
        chance = chance_fn(tr)
        kept, log = aflite(tr, views, chance, label_fn, rng, src, balance=balance)
        tags = {}
        for s in ("val", "test"):
            ev = [r for r in splits[s] if source_of(r) == src]
            tags[s] = tag_eval(tr, ev, views) if ev else {}
            out[s] += ev
        out["train"] += kept
        report[src] = {"train_before": len(tr), "train_after": len(kept), "rounds": log,
                       "val_blind_easy_frac": tags.get("val")}
        print(f"    {name}/{src}: train {len(tr)} -> {len(kept)}  "
              f"val blind-easy {tags.get('val')}", flush=True)
    for s in out:
        prng.shuffle(out[s])
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="mcq,bool,score,diversity")
    args = ap.parse_args()
    parts = args.only.split(",")
    full_report = {}

    if "mcq" in parts:
        print("=== mcq (options-only + length) ===", flush=True)
        sp = {s: read(os.path.join(DATA_DIR, "mcq_corpus", f"{s}.jsonl")) for s in ("train", "val", "test")}
        # Only sources the audit flagged; CommonsenseQA's options-only is below chance.
        flagged = {"openbookqa", "sciq", "arc_easy", "arc_challenge", "race", "hellaswag"}
        out, rep = run_corpus(
            "mcq", sp, lambda r: r["source"],
            lambda src: {"options_only": OptionsView()} if src in flagged else {},
            lambda rows: float(np.mean([1 / len(r["options"]) for r in rows])),
            lambda r: r["answer_idx"], balance=False)
        # answer position is re-shuffled at train time, so no positional balance needed
        for s, rows in out.items():
            write(rows, os.path.join(DATA_DIR, "exp7_mcq_corpus", f"{s}.jsonl"))
        full_report["mcq"] = rep

    specs = {
        # Only the views the audit flagged per source. Filtering an unflagged
        # view (MNLI's premise, 48% -> 40% on 2-way) creates an inverse cue.
        "bool": ("exp7_bool_corpus", lambda r: r["answer_idx"], lambda rows: 0.5,
                 lambda src: {
                     "mnli": {"question_only": TextView("question", lambda r: r["answer_idx"])},
                     "paws": {},
                 }.get(src, {"question_only": TextView("question", lambda r: r["answer_idx"]),
                             "context_only": TextView("context", lambda r: r["answer_idx"])})),
        "score": ("exp7_score_corpus", lambda r: r["level"], lambda rows: 0.2,
                  lambda src: {"length_only": TextView("context", lambda r: r["level"], kind="len")}),
        "diversity": ("exp7_diversity_corpus", lambda r: r["label"],
                      lambda rows: 1 / len(rows[0]["labels"]),
                      lambda src: ({"length_only": TextView("text", lambda r: r["label"], kind="len")}
                                   if src in ("dbpedia14", "yahoo_answers", "tweet_offensive") else {})),
    }
    for part, (d, label_fn, chance_fn, views_fn) in specs.items():
        if part not in parts:
            continue
        print(f"=== {part} ===", flush=True)
        base = os.path.join(DATA_DIR, d)
        sp = {}
        for s in ("train", "val", "test"):
            src_path = os.path.join(base, f"{s}.unfiltered.jsonl")
            if not os.path.exists(src_path):
                os.replace(os.path.join(base, f"{s}.jsonl"), src_path)
            sp[s] = read(src_path)
        # Diversity is NOT re-balanced per round: re-balancing to the rarest
        # class after each drop cut DBpedia 21000 -> 9618 over a weak length
        # cue. Its label balance is restored by class-balanced sampling in
        # data.py instead. bool/score keep per-round balancing (2-5 labels).
        out, rep = run_corpus(part, sp, lambda r: r["source"], views_fn, chance_fn, label_fn,
                              balance=(part != "diversity"))
        for s, rows in out.items():
            write(rows, os.path.join(base, f"{s}.jsonl"))
        full_report[part] = rep

    p = os.path.join(DATA_DIR, "exp7_filter_report.json")
    old = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            old = json.load(f)
    old.update(full_report)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=2)
    print(f"-> {p}")


if __name__ == "__main__":
    main()

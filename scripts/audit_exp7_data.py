"""Shortcut audit for every exp7 training corpus.

The question for each source: can a model that is BLIND to the part of the
input the task is supposed to depend on still beat chance? If yes, the data
contains a label-predicting surface cue, and a network will learn that cue
before (or instead of) the task. exp7e's ProofWriter "not -> False" rule
would have shown up here as a question-only baseline of ~94% on True/False.

Per source, on the val split, with models fit on train:
  chance            1 / num_labels (or mean 1/N for variable option sets)
  majority          most frequent train label, applied to val
  <blind>_only      TF-IDF (1-2 gram) + logistic regression over ONLY that
                    field. Which blind views are run depends on the task:
                      bool:  question_only, context_only (need both to solve)
                      score: length_only, position_only
                      choice (diversity/intent): length_only
                      mcq:   options_only (score each option alone, argmax),
                             longest_option, shortest_option
  keyword_match     pick the option sharing the most words with the input --
                    not a bug, a measure of how lexical the task is
  leakage           normalized-text overlap between splits

A blind baseline is FLAGGED when it beats chance by more than
max(5 points, 3 standard errors). Output: printed table + data/exp7_data_audit.json.

Usage: python scripts/audit_exp7_data.py [--max_train 20000]
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to and or in on for is are was were be been it this that with as at by from "
           "what which who how does do did not no yes i you he she they we my your".split())


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def words(t):
    return {w for w in _WORD.findall((t or "").lower()) if w not in STOP}


def norm(t):
    return " ".join(_WORD.findall((t or "").lower()))


def flag(acc, chance, n):
    if n == 0 or acc is None:
        return False
    se = math.sqrt(max(chance * (1 - chance), 1e-6) / n)
    return acc - chance > max(0.05, 3 * se)


def text_clf(train_x, train_y, val_x, val_y, w=None):
    """w: per-row train sampling weights (bias_w from weight_exp7_bias.py) --
    the blind model is fit under the SAME distribution training samples from."""
    if len(set(train_y)) < 2 or not val_x:
        return None, []
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    X = vec.fit_transform(train_x)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, train_y, sample_weight=w)
    acc = float((clf.predict(vec.transform(val_x)) == np.array(val_y)).mean())
    top = []
    if len(clf.classes_) == 2:
        names = vec.get_feature_names_out()
        coef = clf.coef_[0]
        order = np.argsort(coef)
        top = ([f"{names[i]}(-> {clf.classes_[0]})" for i in order[:6]] +
               [f"{names[i]}(-> {clf.classes_[1]})" for i in order[-6:][::-1]])
    return acc, top


def len_clf(train_x, train_y, val_x, val_y):
    if len(set(train_y)) < 2 or not val_x:
        return None
    f = lambda xs: np.array([[math.log1p(len(x)), math.log1p(len(x.split()))] for x in xs])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(f(train_x), train_y)
    return float((clf.predict(f(val_x)) == np.array(val_y)).mean())


def leakage(splits, key):
    keys = {s: {norm(key(r)) for r in rs} for s, rs in splits.items()}
    out = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in keys and b in keys:
            out[f"{a}&{b}"] = len(keys[a] & keys[b])
    return out


def row_report(name, n_val, chance, results, notes=None):
    flags = [k for k, v in results.items() if k.endswith("_only") or k in ("longest_option", "shortest_option", "majority")
             if isinstance(v, float) and flag(v, chance, n_val)]
    return {"source": name, "n_val": n_val, "chance": round(chance, 4),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in results.items()},
            "FLAGS": flags, **(notes or {})}


# --------------------------------------------------------------------------

def audit_bool(max_train):
    d = os.path.join(DATA_DIR, "exp7_bool_corpus")
    sp = {s: read(os.path.join(d, f"{s}.jsonl")) for s in ("train", "val", "test")}
    out = []
    for src in sorted({r["source"] for r in sp["train"]}):
        tr = [r for r in sp["train"] if r["source"] == src][:max_train]
        va = [r for r in sp["val"] if r["source"] == src]
        y_tr, y_va = [r["answer_idx"] for r in tr], [r["answer_idx"] for r in va]
        maj = Counter(y_tr).most_common(1)[0][0]
        w = [r.get("bias_w", 1.0) for r in tr]
        q_acc, q_top = text_clf([r["question"] for r in tr], y_tr, [r["question"] for r in va], y_va, w)
        c_acc, _ = text_clf([r["context"] for r in tr], y_tr, [r["context"] for r in va], y_va, w)
        res = {"label_dist_train": dict(Counter(y_tr)),
               "majority": float(np.mean([y == maj for y in y_va])),
               "question_only": q_acc, "context_only": c_acc}
        out.append(row_report(f"bool/{src}", len(va), 0.5, res,
                              {"question_only_top_features": q_top,
                               "leakage": leakage({s: [r for r in rs if r["source"] == src] for s, rs in sp.items()},
                                                  lambda r: r["context"] + r["question"])}))
    return out


def audit_score(max_train):
    d = os.path.join(DATA_DIR, "exp7_score_corpus")
    sp = {s: read(os.path.join(d, f"{s}.jsonl")) for s in ("train", "val", "test")}
    out = []
    for src in sorted({r["source"] for r in sp["train"]}):
        tr = [r for r in sp["train"] if r["source"] == src][:max_train]
        va = [r for r in sp["val"] if r["source"] == src]
        lv_tr, lv_va = [r["level"] for r in tr], [r["level"] for r in va]
        pos_tr = Counter(r["answer_idx"] for r in tr)
        pos_maj = pos_tr.most_common(1)[0][0]
        res = {"level_dist_train": dict(sorted(Counter(lv_tr).items())),
               "answer_position_dist_train": dict(sorted(pos_tr.items())),
               "reversed_frac": round(np.mean([r["reversed"] for r in tr]), 3),
               "majority": float(np.mean([l == Counter(lv_tr).most_common(1)[0][0] for l in lv_va])),
               "position_only": float(np.mean([r["answer_idx"] == pos_maj for r in va])),
               "length_only": len_clf([r["context"] for r in tr], lv_tr, [r["context"] for r in va], lv_va),
               "question_only": text_clf([r["question"] for r in tr], lv_tr, [r["question"] for r in va], lv_va)[0]}
        # Spearman-ish: correlation of length with level, for insight.
        L = np.array([len(r["context"]) for r in tr]); Y = np.array(lv_tr)
        res["len_level_corr"] = round(float(np.corrcoef(np.argsort(np.argsort(L)), Y)[0, 1]), 3)
        out.append(row_report(f"score/{src}", len(va), 1 / 5, res,
                              {"leakage": leakage({s: [r for r in rs if r["source"] == src] for s, rs in sp.items()},
                                                  lambda r: r["context"] + r["question"])}))
    return out


def keyword_match(text, options):
    tw = words(text)
    scores = [len(tw & words(o)) for o in options]
    best = max(scores)
    if best == 0:
        return None  # no overlap anywhere -> undecided
    return scores.index(best)


def audit_diversity(max_train):
    d = os.path.join(DATA_DIR, "exp7_diversity_corpus")
    sp = {s: read(os.path.join(d, f"{s}.jsonl")) for s in ("train", "val", "test")}
    out = []
    for src in sorted({r["source"] for r in sp["train"]}):
        tr = [r for r in sp["train"] if r["source"] == src][:max_train]
        va = [r for r in sp["val"] if r["source"] == src]
        n_lab = len(tr[0]["labels"])
        y_tr, y_va = [r["label"] for r in tr], [r["label"] for r in va]
        maj = Counter(y_tr).most_common(1)[0][0]
        km = [keyword_match(r["text"], r["labels"]) for r in va]
        km_acc = float(np.mean([(k is not None and r["labels"][k] == r["label"]) or
                                (k is None and False) for k, r in zip(km, va)]))
        res = {"n_labels": n_lab, "imbalance_train": round(max(Counter(y_tr).values()) / min(Counter(y_tr).values()), 2),
               "majority": float(np.mean([y == maj for y in y_va])),
               "length_only": len_clf([r["text"] for r in tr], y_tr, [r["text"] for r in va], y_va),
               "keyword_match": km_acc,
               "keyword_coverage": round(float(np.mean([k is not None for k in km])), 3)}
        out.append(row_report(f"diversity/{src}", len(va), 1 / n_lab, res,
                              {"leakage": leakage({s: [r for r in rs if r["source"] == src] for s, rs in sp.items()},
                                                  lambda r: r["text"])}))
    return out


def audit_intent(max_train):
    d = os.path.join(DATA_DIR, "intent_corpus")
    tr = [r for r in read(os.path.join(d, "train.jsonl")) if not r["label"].startswith("banking::")]
    va = [r for r in read(os.path.join(d, "val.jsonl")) if not r["label"].startswith("banking::")]
    c = Counter(r["label"] for r in tr)
    by_src = Counter(r["label"].split("::")[0] for r in tr)
    res = {"n_labels": len(c), "imbalance_train": round(max(c.values()) / min(c.values()), 1),
           "per_source_train": dict(by_src),
           "top_labels": c.most_common(5),
           "majority": float(np.mean([r["label"] == c.most_common(1)[0][0] for r in va]))}
    note = {"note": "options are sampled at train time (data.py) -- label frequency imbalance becomes a "
                    "prior toward frequent labels whenever they are among the options; capped in data.py"}
    return [row_report("intent/all(minus banking77)", len(va), 1 / len(c), res, note)]


def audit_mcq(max_train):
    # Filtered corpus when it exists (filter_exp7_shortcuts.py), original otherwise.
    d = os.path.join(DATA_DIR, "exp7_mcq_corpus")
    if not os.path.exists(os.path.join(d, "train.jsonl")):
        d = os.path.join(DATA_DIR, "mcq_corpus")
    sp = {s: read(os.path.join(d, f"{s}.jsonl")) for s in ("train", "val")}
    out = []
    for src in sorted({r["source"] for r in sp["train"]}):
        tr = [r for r in sp["train"] if r["source"] == src][:max_train]
        va = [r for r in sp["val"] if r["source"] == src]
        if not va:
            continue
        # options-only: binary classifier on option text alone, argmax per question.
        X, Y = [], []
        for r in tr:
            for i, o in enumerate(r["options"]):
                X.append(o); Y.append(int(i == r["answer_idx"]))
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(vec.fit_transform(X), Y)
        ok = []
        for r in va:
            p = clf.decision_function(vec.transform(r["options"]))
            ok.append(int(np.argmax(p)) == r["answer_idx"])
        chance = float(np.mean([1 / len(r["options"]) for r in va]))
        longest = float(np.mean([max(range(len(r["options"])), key=lambda i: len(r["options"][i])) == r["answer_idx"] for r in va]))
        shortest = float(np.mean([min(range(len(r["options"])), key=lambda i: len(r["options"][i])) == r["answer_idx"] for r in va]))
        km = [keyword_match((r.get("context") or "") + " " + r["question"], r["options"]) for r in va]
        km_acc = float(np.mean([k is not None and k == r["answer_idx"] for k, r in zip(km, va)]))
        res = {"options_only": float(np.mean(ok)), "longest_option": longest, "shortest_option": shortest,
               "keyword_match": km_acc}
        out.append(row_report(f"mcq/{src}", len(va), chance, res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--only", default="bool,score,diversity,intent,mcq")
    args = ap.parse_args()
    parts = args.only.split(",")
    fns = {"bool": audit_bool, "score": audit_score, "diversity": audit_diversity,
           "intent": audit_intent, "mcq": audit_mcq}
    report = []
    for p in parts:
        print(f"auditing {p}...", flush=True)
        report += fns[p](args.max_train)

    path = os.path.join(DATA_DIR, "exp7_data_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cols = ["chance", "majority", "question_only", "context_only", "length_only", "position_only",
            "options_only", "longest_option", "keyword_match"]
    print(f"\n{'source':34s} {'n_val':>6s} " + " ".join(f"{c[:10]:>10s}" for c in cols) + "  FLAGS")
    for r in report:
        vals = " ".join(f"{r[c]*100:10.1f}" if isinstance(r.get(c), float) else f"{'':>10s}" for c in cols)
        print(f"{r['source']:34s} {r['n_val']:6d} {vals}  {','.join(r['FLAGS'])}")
    for r in report:
        if r.get("question_only_top_features") and "question_only" in r["FLAGS"]:
            print(f"\n{r['source']} question-only cues: {r['question_only_top_features']}")
        lk = r.get("leakage")
        if lk and any(lk.values()):
            print(f"{r['source']} LEAKAGE {lk}")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()

"""Builds the System-1 corpora experiment 7 trains on (NOTES.md Sec 5.2),
rebuilt 2026-09-24 with shortcut removal as the first design constraint.

Why the rebuild: exp7e learned "question contains 'not' -> False" on
ProofWriter and scored exactly that heuristic's accuracy for three epochs.
A corpus with a label-predicting surface cue gets that cue learned instead
of the task, so every source here is built to leave no such cue in its
label distribution, and scripts/audit_exp7_data.py measures what's left.

Rules applied to EVERY source:
  1. Labels balanced within each (source, split) by downsampling to the
     rarest class (with a floor for tiny classes -- reported, not hidden).
     A label prior then carries no information, and val accuracy can be
     read directly against 1/num_labels.
  2. Deduplicated on a normalized text key, and no key is shared across
     splits (test wins over val wins over train).
  3. Any text matching a held-out benchmark is removed: AG News, DAIR
     Emotion, Banking77, JevBench Hard-111, typed-decisions. Those stay
     zero-shot.
  4. Every decision (counts before/after, dropped rows, per-class caps)
     goes into data/exp7_manifest.json.

Outputs (formats consumed by experiments/exp7_hybrid_decision/data.py):
  data/exp7_bool_corpus/{train,val,test}.jsonl
      {context, question, answer_idx (1=yes), source}
      boolq, mnli (entailment vs neutral+contradiction), paws, qqp
  data/exp7_score_corpus/{train,val,test}.jsonl
      {context, question, options (ordered), num_levels, answer_idx, source}
      yelp_stars, amazon_stars, sst5, stsb -- options are descriptive level
      texts, sometimes listed high->low so direction must be read, not assumed.
  data/exp7_diversity_corpus/{train,val,test}.jsonl + instructions.json
      {text, label, labels (native label set), source}
      dbpedia14, yahoo_answers, 20_newsgroups, trec_coarse, trec_fine,
      goemotions, tweet_hate, tweet_irony, tweet_offensive, tweet_sentiment,
      tweet_stance_<target>
  data/exp7_typed_decisions/test.jsonl   eval-only, never trained on.

Deliberately excluded (held-out benchmarks or near-copies of their label
vocabulary): AG News, BBC News, dair-ai/emotion (= SetFit/emotion),
tweet_eval/emotion, Banking77.

Usage: python scripts/build_exp7_data.py [--only bool,score,diversity,typed]
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SEED = 20260924

TRAIN_PER_CLASS = {"default": 2500}
VAL_PER_SOURCE = 1000
TEST_PER_SOURCE = 2000
MIN_CLASS_FLOOR = 50   # val/test: rarest-class floor
TRAIN_BALANCE_FLOOR = 400  # train: per-class cap never drops below this (see finalize)

MANIFEST = {"seed": SEED, "sources": {}, "blocklist": {}}


# --------------------------------------------------------------------------
# Normalization, dedup, blocklist
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9 ]")


def norm_key(text) -> str:
    if not isinstance(text, str):
        text = json.dumps(text, sort_keys=True)  # Hard-111 / typed-decisions states can be dicts
    t = _NONWORD.sub(" ", (text or "").lower())
    t = _WS.sub(" ", t).strip()
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def clean(text: str, max_chars: int = 0) -> str:
    t = _WS.sub(" ", (text or "").replace("\\n", " ")).strip()
    if max_chars and len(t) > max_chars:
        cut = t[:max_chars]
        t = cut[:cut.rfind(" ")] if " " in cut else cut
    return t


def build_blocklist():
    """Normalized keys of every held-out benchmark text."""
    keys = set()
    counts = {}

    def add(name, texts):
        n0 = len(keys)
        for t in texts:
            if t:
                keys.add(norm_key(t))
        counts[name] = len(keys) - n0

    print("  blocklist: AG News...", flush=True)
    ag = load_dataset("fancyzhx/ag_news")
    add("ag_news", [r["text"] for s in ag.values() for r in s])
    print("  blocklist: DAIR emotion...", flush=True)
    emo = load_dataset("dair-ai/emotion")
    add("dair_emotion", [r["text"] for s in emo.values() for r in s])
    print("  blocklist: Banking77 (from intent_corpus)...", flush=True)
    b77 = []
    for split in ("train", "val", "test"):
        with open(os.path.join(DATA_DIR, "intent_corpus", f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["label"].startswith("banking::"):
                    b77.append(r["text"])
    add("banking77", b77)
    hard = os.path.join(DATA_DIR, "hard_111.jsonl")
    if os.path.exists(hard):
        with open(hard, encoding="utf-8") as f:
            add("hard_111", [json.loads(l).get("state", "") for l in f if l.strip()])
    MANIFEST["blocklist"] = counts
    print(f"  blocklist: {len(keys)} keys {counts}", flush=True)
    return keys


# --------------------------------------------------------------------------
# Generic finalize: dedup -> cross-split leakage -> blocklist -> balance
# --------------------------------------------------------------------------

def finalize(source, splits, text_of, label_of, blocklist, rng,
             train_per_class=None, val_n=VAL_PER_SOURCE, test_n=TEST_PER_SOURCE):
    """splits: {"train": rows, "val": rows, "test": rows} (val may be empty ->
    carved from train). Returns balanced, leak-free splits + manifest entry."""
    info = {"raw": {s: len(r) for s, r in splits.items()}}

    if not splits.get("val"):
        tr = list(splits["train"])
        rng.shuffle(tr)
        n_val = min(max(val_n * 2, len(tr) // 10), len(tr) // 5)
        splits["val"], splits["train"] = tr[:n_val], tr[n_val:]
    if not splits.get("test"):
        va = list(splits["val"])
        rng.shuffle(va)
        half = len(va) // 2
        splits["test"], splits["val"] = va[:half], va[half:]

    seen = set()
    dropped = Counter()
    out = {}
    for s in ("test", "val", "train"):
        kept = []
        for r in splits[s]:
            k = norm_key(text_of(r))
            raw = r.get("context") or r.get("text") or ""
            if k in blocklist or norm_key(raw) in blocklist:
                dropped[f"{s}:blocklist"] += 1
                continue
            if k in seen:
                dropped[f"{s}:dup_or_cross_split"] += 1
                continue
            seen.add(k)
            kept.append(r)
        out[s] = kept

    per_class_cap = {"train": train_per_class or TRAIN_PER_CLASS["default"]}
    for s in ("train", "val", "test"):
        by = defaultdict(list)
        for r in out[s]:
            by[label_of(r)].append(r)
        for rs in by.values():
            rng.shuffle(rs)
        n_cls = len(by)
        if s == "train":
            counts = sorted(len(v) for v in by.values())
            # Balance to the rarest class, but never below TRAIN_BALANCE_FLOOR:
            # balancing GoEmotions to "grief" (~50) or TREC to "abbreviation"
            # (~68) threw away ~90% of those sources. Classes under the floor
            # keep everything they have; the residual imbalance is removed at
            # train time by class-balanced sampling in data.py, so it never
            # reaches the model as a label prior. val/test stay strictly balanced.
            rare = counts[0]
            cap = min(per_class_cap["train"], max(rare, TRAIN_BALANCE_FLOOR))
        else:
            budget = val_n if s == "val" else test_n
            cap = max(1, budget // max(1, n_cls))
            counts = sorted(len(v) for v in by.values())
            cap = min(cap, max(counts[0], MIN_CLASS_FLOOR // 5))
        rows = []
        for rs in by.values():
            rows.extend(rs[:cap])
        rng.shuffle(rows)
        out[s] = rows
        lab = Counter(label_of(r) for r in rows)
        info[s] = {"n": len(rows), "n_labels": len(lab), "per_class_cap": cap,
                   "max_min_ratio": round(max(lab.values()) / max(1, min(lab.values())), 2) if lab else None}
    info["dropped"] = dict(dropped)
    MANIFEST["sources"][source] = info
    print(f"    {source:24s} train {info['train']['n']:6d}  val {info['val']['n']:5d}  "
          f"test {info['test']['n']:5d}  labels {info['train']['n_labels']:3d}  "
          f"imbalance {info['train']['max_min_ratio']}  dropped {dict(dropped)}", flush=True)
    return out


def write_splits(corpus_dir, per_source, extra_files=None):
    rows = {"train": [], "val": [], "test": []}
    for splits in per_source:
        for s in rows:
            rows[s].extend(splits[s])
    rng = random.Random(SEED + 1)
    d = os.path.join(DATA_DIR, corpus_dir)
    os.makedirs(d, exist_ok=True)
    for s, rs in rows.items():
        rng.shuffle(rs)
        with open(os.path.join(d, f"{s}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")},
                                   ensure_ascii=False) + "\n")
        print(f"  wrote {len(rs):6d} -> {corpus_dir}/{s}.jsonl", flush=True)
    for name, obj in (extra_files or {}).items():
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)


def hf_rows(ds, n_max, rng):
    idx = list(range(len(ds)))
    if len(idx) > n_max:
        idx = rng.sample(idx, n_max)
    return [ds[i] for i in idx]


# --------------------------------------------------------------------------
# bool
# --------------------------------------------------------------------------

MNLI_Q = [
    'Based on this, is it true that "{h}"?',
    'Does the text support the claim: "{h}"?',
    'Can we conclude that "{h}"?',
    'Is the statement "{h}" entailed by the text?',
]
PARA_Q = [
    'Does this mean the same thing as: "{s}"?',
    'Is this a paraphrase of: "{s}"?',
    'Do these two sentences say the same thing? Other sentence: "{s}"',
]


def build_bool(blocklist):
    print("=== exp7_bool_corpus ===", flush=True)
    rng = random.Random(SEED)
    out = []

    # Dedup/leak key is the underlying content (premise+hypothesis, passage+
    # question, sentence pair), never the randomly chosen question template.
    text_of = lambda r: r["_key"]
    label_of = lambda r: r["answer_idx"]

    # BoolQ: native yes/no QA. 62% yes upstream -> balanced here.
    bq = load_dataset("google/boolq")
    mk = lambda ex: {"context": clean(ex["passage"], 2000), "question": ex["question"].strip().rstrip("?") + "?",
                     "answer_idx": int(bool(ex["answer"])), "source": "boolq",
                     "_key": ex["passage"] + " || " + ex["question"]}
    out.append(finalize("boolq", {"train": [mk(e) for e in bq["train"]],
                                  "val": [mk(e) for e in bq["validation"]], "test": []},
                        text_of, label_of, blocklist, rng))

    # MNLI: yes = entailment; no = neutral OR contradiction, in equal parts.
    # Using only contradiction as "no" makes negation words in the hypothesis
    # a label cue (the known hypothesis-only artifact); mixing in neutral
    # dilutes it. The audit's question-only baseline measures what remains.
    mn = load_dataset("nyu-mll/multi_nli")

    def mnli_rows(ds, n):
        rows = []
        for ex in hf_rows(ds, n, rng):
            if ex["label"] not in (0, 1, 2):
                continue
            rows.append({"context": clean(ex["premise"]),
                         "question": rng.choice(MNLI_Q).format(h=ex["hypothesis"].strip()),
                         "answer_idx": int(ex["label"] == 0), "source": "mnli",
                         "_sub": ex["label"], "_key": ex["premise"] + " || " + ex["hypothesis"]})
        # equal neutral / contradiction inside "no"
        yes = [r for r in rows if r["_sub"] == 0]
        neu = [r for r in rows if r["_sub"] == 1]
        con = [r for r in rows if r["_sub"] == 2]
        k = min(len(neu), len(con), len(yes) // 2)
        rows = yes + neu[:k] + con[:k]
        for r in rows:
            r["nli_label"] = ["entailment", "neutral", "contradiction"][r.pop("_sub")]
        return rows

    out.append(finalize("mnli", {"train": mnli_rows(mn["train"], 60000),
                                 "val": mnli_rows(mn["validation_matched"], 10000),
                                 "test": mnli_rows(mn["validation_mismatched"], 10000)},
                        text_of, label_of, blocklist, rng, train_per_class=10000))

    # PAWS: adversarial paraphrases with high word overlap either way, so
    # "shares most words -> paraphrase" (QQP's shortcut) stops working.
    pw = load_dataset("google-research-datasets/paws", "labeled_final")

    def para_rows(ds, n, src, a="sentence1", b="sentence2", lab="label"):
        rows = []
        for ex in hf_rows(ds, n, rng):
            s1, s2 = ex[a], ex[b]
            if rng.random() < 0.5:
                s1, s2 = s2, s1
            rows.append({"context": clean(s1), "question": rng.choice(PARA_Q).format(s=clean(s2)),
                         "answer_idx": int(ex[lab]), "source": src,
                         "_key": " || ".join(sorted([ex[a], ex[b]]))})
        return rows

    out.append(finalize("paws", {"train": para_rows(pw["train"], 49401, "paws"),
                                 "val": para_rows(pw["validation"], 8000, "paws"),
                                 "test": para_rows(pw["test"], 8000, "paws")},
                        text_of, label_of, blocklist, rng, train_per_class=8000))

    # QQP (from the local pairs exp6/exp7a used), 63% negative -> balanced.
    qqp = {}
    for s in ("train", "val"):
        with open(os.path.join(DATA_DIR, "qqp_paraphrase_pairs", f"{s}.jsonl"), encoding="utf-8") as f:
            raw = [json.loads(l) for l in f if l.strip()]
        qqp[s] = para_rows(raw, 60000, "qqp", "text1", "text2", "is_paraphrase")
    out.append(finalize("qqp", {"train": qqp["train"], "val": qqp["val"], "test": []},
                        text_of, label_of, blocklist, rng, train_per_class=8000))

    write_splits("exp7_bool_corpus", out)


# --------------------------------------------------------------------------
# score (ordinal)
# --------------------------------------------------------------------------

STAR_RENDERINGS = [
    ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
    ["1 star - terrible", "2 stars - poor", "3 stars - average", "4 stars - good", "5 stars - excellent"],
    ["very negative", "negative", "mixed or neutral", "positive", "very positive"],
    ["hated it", "disliked it", "it was okay", "liked it", "loved it"],
]
SENTIMENT_RENDERINGS = [
    ["very negative", "negative", "neutral", "positive", "very positive"],
    ["strongly negative", "somewhat negative", "neither positive nor negative",
     "somewhat positive", "strongly positive"],
    ["1 - very negative", "2 - negative", "3 - neutral", "4 - positive", "5 - very positive"],
]
SIMILARITY_RENDERINGS = [
    ["completely different meaning", "mostly different, a few shared details",
     "partly the same, important differences", "mostly the same, minor differences",
     "same meaning"],
    ["0 - unrelated", "1 - loosely related", "2 - partially equivalent",
     "3 - roughly equivalent", "4 - fully equivalent"],
]
SCORE_Q = {
    "yelp_stars": ["How many stars would this review give?", "Rate this review on the scale.",
                   "How satisfied is the reviewer?"],
    "amazon_stars": ["How many stars did this product review give?", "How satisfied is this customer?",
                     "Rate this review on the scale."],
    "sst5": ["How positive is the sentiment of this text?", "Rate the sentiment of this sentence.",
             "How does this review feel about the movie?"],
    "stsb": ['How similar in meaning is this to: "{s}"?', 'Rate the semantic similarity to: "{s}".'],
}
P_REVERSED = 0.3


def build_score(blocklist):
    print("=== exp7_score_corpus ===", flush=True)
    rng = random.Random(SEED + 2)
    out = []

    def mk(context, level, source, renderings, question, key=None):
        opts = list(rng.choice(renderings))
        ans = level
        reversed_ = rng.random() < P_REVERSED
        if reversed_:
            opts = opts[::-1]
            ans = len(opts) - 1 - level
        return {"context": context, "question": question, "options": opts,
                "num_levels": len(opts), "answer_idx": ans, "level": level,
                "reversed": reversed_, "source": source, "_key": key or context}

    text_of = lambda r: r["_key"]
    label_of = lambda r: r["level"]

    yelp = load_dataset("Yelp/yelp_review_full")
    y = lambda ds, n: [mk(clean(e["text"], 1500), int(e["label"]), "yelp_stars", STAR_RENDERINGS,
                          rng.choice(SCORE_Q["yelp_stars"])) for e in hf_rows(ds, n, rng)]
    out.append(finalize("yelp_stars", {"train": y(yelp["train"], 30000), "val": [],
                                       "test": y(yelp["test"], 6000)},
                        text_of, label_of, blocklist, rng))

    amz = load_dataset("SetFit/amazon_reviews_multi_en")
    a = lambda ds, n: [mk(clean(e["text"], 1500), int(e["label"]), "amazon_stars", STAR_RENDERINGS,
                          rng.choice(SCORE_Q["amazon_stars"])) for e in hf_rows(ds, n, rng)]
    out.append(finalize("amazon_stars", {"train": a(amz["train"], 30000), "val": a(amz["validation"], 5000),
                                         "test": a(amz["test"], 5000)},
                        text_of, label_of, blocklist, rng))

    sst = load_dataset("SetFit/sst5")
    s5 = lambda ds: [mk(clean(e["text"]), int(e["label"]), "sst5", SENTIMENT_RENDERINGS,
                        rng.choice(SCORE_Q["sst5"])) for e in ds]
    out.append(finalize("sst5", {"train": s5(sst["train"]), "val": s5(sst["validation"]),
                                 "test": s5(sst["test"])},
                        text_of, label_of, blocklist, rng))

    # STS-B: similarity is ordinal but has nothing to do with sentiment, so
    # the score head can't reduce to "how positive is this".
    sts = load_dataset("sentence-transformers/stsb")

    def st(ds):
        rows = []
        for e in ds:
            lvl = min(4, int(round(float(e["score"]) * 4)))
            s1, s2 = e["sentence1"], e["sentence2"]
            if rng.random() < 0.5:
                s1, s2 = s2, s1
            rows.append(mk(clean(s1), lvl, "stsb", SIMILARITY_RENDERINGS,
                           rng.choice(SCORE_Q["stsb"]).format(s=clean(s2)),
                           key=" || ".join(sorted([e["sentence1"], e["sentence2"]]))))
        return rows

    out.append(finalize("stsb", {"train": st(sts["train"]), "val": st(sts["validation"]),
                                 "test": st(sts["test"])},
                        text_of, label_of, blocklist, rng))

    write_splits("exp7_score_corpus", out)


# --------------------------------------------------------------------------
# diversity (label-vocabulary diversity, choice)
# --------------------------------------------------------------------------

DBPEDIA_NAMES = ["company", "educational institution", "artist", "athlete", "office holder",
                 "means of transportation", "building", "natural place", "village", "animal",
                 "plant", "album", "film", "written work"]
NEWSGROUP_NAMES = {
    "alt.atheism": "atheism and religion debate",
    "comp.graphics": "computer graphics",
    "comp.os.ms-windows.misc": "Microsoft Windows",
    "comp.sys.ibm.pc.hardware": "PC hardware",
    "comp.sys.mac.hardware": "Mac hardware",
    "comp.windows.x": "X Window System",
    "misc.forsale": "items for sale",
    "rec.autos": "cars",
    "rec.motorcycles": "motorcycles",
    "rec.sport.baseball": "baseball",
    "rec.sport.hockey": "hockey",
    "sci.crypt": "cryptography",
    "sci.electronics": "electronics",
    "sci.med": "medicine",
    "sci.space": "space",
    "soc.religion.christian": "Christianity",
    "talk.politics.guns": "gun politics",
    "talk.politics.mideast": "Middle East politics",
    "talk.politics.misc": "general politics",
    "talk.religion.misc": "religion",
}
STANCE_TARGETS = {"abortion": "abortion", "atheism": "atheism", "climate": "climate change action",
                  "feminist": "feminism", "hillary": "Hillary Clinton"}

INSTRUCTIONS = {
    "dbpedia14": ["What type of entity is this text about?", "Which category of entity does this describe?",
                  "What kind of thing is being described here?", ""],
    "yahoo_answers": ["Which topic does this question belong to?", "What subject area is this about?",
                      "Which category best fits this question?", ""],
    "20_newsgroups": ["Which discussion group does this post belong to?", "What subject is this post about?",
                      "Which topic best fits this message?", ""],
    "trec_coarse": ["What type of answer does this question expect?",
                    "What kind of information is this question asking for?", ""],
    "trec_fine": ["What specific type of answer does this question expect?",
                  "What exactly is this question asking for?", ""],
    "goemotions": ["What emotion does this comment express?", "How does the writer feel?",
                   "Which emotion best fits this comment?", ""],
    "tweet_hate": ["Is this tweet hateful?", "Does this tweet contain hate speech?", ""],
    "tweet_irony": ["Is this tweet ironic?", "Is the writer being ironic?", ""],
    "tweet_offensive": ["Is this tweet offensive?", "Would this tweet offend people?", ""],
    "tweet_sentiment": ["What is the sentiment of this tweet?", "How does this tweet feel?", ""],
    "tweet_stance": ["What stance does this tweet take on {target}?",
                     "Is this tweet for or against {target}?", "How does the author view {target}?"],
}


def build_diversity(blocklist):
    print("=== exp7_diversity_corpus ===", flush=True)
    rng = random.Random(SEED + 3)
    out = []
    text_of = lambda r: r["text"]
    label_of = lambda r: r["label"]

    def rows_from(ds, n, text_fn, label_fn, labels, source, min_chars=15, extra=None):
        rows = []
        for e in hf_rows(ds, n, rng):
            t = text_fn(e)
            if not t or len(t) < min_chars:
                continue
            r = {"text": t, "label": label_fn(e), "labels": list(labels), "source": source}
            if extra:
                r.update(extra)
            rows.append(r)
        return rows

    db = load_dataset("fancyzhx/dbpedia_14")
    f = lambda ds, n: rows_from(ds, n, lambda e: clean(e["title"] + ". " + e["content"], 800),
                                lambda e: DBPEDIA_NAMES[e["label"]], DBPEDIA_NAMES, "dbpedia14")
    out.append(finalize("dbpedia14", {"train": f(db["train"], 40000), "val": [], "test": f(db["test"], 8000)},
                        text_of, label_of, blocklist, rng, train_per_class=1500))

    ya = load_dataset("community-datasets/yahoo_answers_topics")
    ynames = ya["train"].features["topic"].names
    f = lambda ds, n: rows_from(ds, n, lambda e: clean(e["question_title"] + " " + (e["question_content"] or ""), 800),
                                lambda e: ynames[e["topic"]], ynames, "yahoo_answers")
    out.append(finalize("yahoo_answers", {"train": f(ya["train"], 40000), "val": [], "test": f(ya["test"], 8000)},
                        text_of, label_of, blocklist, rng, train_per_class=1500))

    ng = load_dataset("SetFit/20_newsgroups")
    nnames = [NEWSGROUP_NAMES[k] for k in sorted(NEWSGROUP_NAMES)]
    f = lambda ds, n: rows_from(ds, n, lambda e: clean(e["text"], 1200),
                                lambda e: NEWSGROUP_NAMES[e["label_text"]], nnames, "20_newsgroups",
                                min_chars=60)
    out.append(finalize("20_newsgroups", {"train": f(ng["train"], 20000), "val": [], "test": f(ng["test"], 8000)},
                        text_of, label_of, blocklist, rng, train_per_class=600))

    # TREC twice, with two different label vocabularies over the same
    # questions -- the answer depends on which label set is offered, which
    # is exactly the conditioning a System-1 decision model needs.
    tr = load_dataset("SetFit/TREC-QC")
    coarse = sorted(set(tr["train"]["label_coarse_text"]))
    fine_counts = Counter(tr["train"]["label_text"])
    fine = sorted(k for k, c in fine_counts.items() if c >= 20)
    f = lambda ds: rows_from(ds, 10000, lambda e: clean(e["text"]), lambda e: e["label_coarse_text"],
                             coarse, "trec_coarse", min_chars=5)
    out.append(finalize("trec_coarse", {"train": f(tr["train"]), "val": [], "test": f(tr["test"])},
                        text_of, label_of, blocklist, rng, train_per_class=800))
    f = lambda ds: [r for r in rows_from(ds, 10000, lambda e: clean(e["text"]), lambda e: e["label_text"],
                                         fine, "trec_fine", min_chars=5) if r["label"] in fine]
    out.append(finalize("trec_fine", {"train": f(tr["train"]), "val": [], "test": f(tr["test"])},
                        text_of, label_of, blocklist, rng, train_per_class=300))

    ge = load_dataset("google-research-datasets/go_emotions", "simplified")
    gnames = ge["train"].features["labels"].feature.names

    def ge_rows(ds):
        return [{"text": clean(e["text"]), "label": gnames[e["labels"][0]], "labels": list(gnames),
                 "source": "goemotions"} for e in ds if len(e["labels"]) == 1 and len(e["text"]) >= 10]

    out.append(finalize("goemotions", {"train": ge_rows(ge["train"]), "val": ge_rows(ge["validation"]),
                                       "test": ge_rows(ge["test"])},
                        text_of, label_of, blocklist, rng, train_per_class=800))

    tweet_labels = {
        "hate": ["not hateful", "hateful"],
        "irony": ["not ironic", "ironic"],
        "offensive": ["not offensive", "offensive"],
        "sentiment": ["negative", "neutral", "positive"],
    }
    for cfg, names in tweet_labels.items():
        tw = load_dataset("cardiffnlp/tweet_eval", cfg)
        f = lambda ds, names=names, cfg=cfg: rows_from(ds, 50000, lambda e: clean(e["text"]),
                                                       lambda e: names[e["label"]], names, f"tweet_{cfg}")
        out.append(finalize(f"tweet_{cfg}", {"train": f(tw["train"]), "val": f(tw["validation"]),
                                             "test": f(tw["test"])},
                            text_of, label_of, blocklist, rng, train_per_class=2000))

    for tgt, name in STANCE_TARGETS.items():
        tw = load_dataset("cardiffnlp/tweet_eval", f"stance_{tgt}")
        names = [f"neutral or unrelated to {name}", f"against {name}", f"in favor of {name}"]
        f = lambda ds, names=names, name=name, tgt=tgt: rows_from(
            ds, 5000, lambda e: clean(e["text"].replace("#SemST", "")), lambda e: names[e["label"]], names,
            f"tweet_stance_{tgt}", extra={"target": name})
        out.append(finalize(f"tweet_stance_{tgt}", {"train": f(tw["train"]), "val": f(tw["validation"]),
                                                    "test": f(tw["test"])},
                            text_of, label_of, blocklist, rng, train_per_class=400))

    write_splits("exp7_diversity_corpus", out, extra_files={"instructions.json": INSTRUCTIONS})


# --------------------------------------------------------------------------
# typed_decisions (eval-only)
# --------------------------------------------------------------------------

def build_typed_decisions():
    print("=== exp7_typed_decisions (eval-only) ===", flush=True)
    try:
        ds = load_dataset("LocalLLaMA/typed-decisions", "all")
    except Exception as e:
        print(f"  FAILED to load LocalLLaMA/typed-decisions ({e}) -- skipping.")
        return
    QTYPE_MAP = {"choice": "choice", "score": "score", "noul": "bool"}
    rows, skipped = [], 0
    for split_name, d in ds.items():
        for ex in d:
            try:
                state = json.loads(ex["state"])
                questions = json.loads(ex["questions"])
                gold = json.loads(ex["gold"])
            except (json.JSONDecodeError, KeyError, TypeError):
                skipped += 1
                continue
            for q_name, q in questions.items():
                crit, qt = q.get("criteria"), q.get("type")
                if not crit or qt not in QTYPE_MAP:
                    skipped += 1
                    continue
                if isinstance(crit, dict):
                    keys, texts = list(crit.keys()), list(crit.values())
                elif isinstance(crit, list):
                    keys, texts = [str(i) for i in range(len(crit))], list(crit)
                else:
                    skipped += 1
                    continue
                g = gold.get(q_name)
                lab = g.get("label") if isinstance(g, dict) else None
                if lab not in keys:
                    skipped += 1
                    continue
                rows.append({"context": json.dumps(state), "question": q.get("instructions", ""),
                             "options": texts, "answer_idx": keys.index(lab), "qtype": QTYPE_MAP[qt],
                             "workflow": ex.get("workflow", "")})
    d = os.path.join(DATA_DIR, "exp7_typed_decisions")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "test.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    MANIFEST["sources"]["typed_decisions"] = {"eval_only": True, "n": len(rows), "skipped": skipped}
    print(f"  wrote {len(rows)} rows ({skipped} skipped)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="bool,score,diversity,typed")
    args = ap.parse_args()
    parts = set(args.only.split(","))

    mpath = os.path.join(DATA_DIR, "exp7_manifest.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            old = json.load(f)
        MANIFEST["sources"].update(old.get("sources", {}))

    print("=== blocklist (held-out benchmarks) ===", flush=True)
    blocklist = build_blocklist()
    if "typed" in parts:
        td = build_typed_decisions() or []
        for r in td:
            blocklist.add(norm_key(r["context"]))
    else:
        tp = os.path.join(DATA_DIR, "exp7_typed_decisions", "test.jsonl")
        if os.path.exists(tp):
            with open(tp, encoding="utf-8") as f:
                for l in f:
                    blocklist.add(norm_key(json.loads(l)["context"]))

    if "bool" in parts:
        build_bool(blocklist)
    if "score" in parts:
        build_score(blocklist)
    if "diversity" in parts:
        build_diversity(blocklist)

    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(MANIFEST, f, indent=2)
    print(f"\nDone. Manifest -> {mpath}")


if __name__ == "__main__":
    main()

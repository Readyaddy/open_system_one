"""Multiple-choice QA corpus: "given a question (and, when available, a
passage of CONTEXT it depends on) and a small set of candidate answers,
pick the correct one" -- structurally identical to the poly-encoder's
compatibility-scoring mechanism (model.compatibility), just with a
per-EXAMPLE candidate bank instead of intent_corpus's one GLOBAL bank of
255 intents shared across all examples. This is the actual target
capability (answer questions from given options, grounded in whatever
context is provided), not a proxy task like intent classification --
see data/README.md.

REVISION: the first version of this corpus deliberately excluded every
context/passage field (RACE entirely, SciQ's `support`) to keep a single
"bare question" task shape. That was backwards for the actual goal --
the target capability is specifically "use the CONTEXT you're given to
pick the right answer," not "answer trivia from parametric knowledge
alone." This version restores context wherever a source naturally has
it, and makes context-grounded examples the majority of the corpus
(RACE + SciQ's support >> the context-free sources combined), while
still keeping some context-free examples (CommonsenseQA, OpenBookQA,
ARC, HellaSwag) since "no extra context was given, use what you know" is
a legitimate case the same skill should also cover.

Seven independent, real MCQ datasets, all HuggingFace `datasets`:
  - RACE           (ehovy/race, "all")       4 options, real passage + question
                                              (English exams for Chinese students,
                                              middle + high school) -- the largest
                                              and most genuinely context-grounded
                                              source here
  - SciQ           (allenai/sciq)            4 options, science, WITH its
                                              supporting passage as context
  - CommonsenseQA  (tau/commonsense_qa)      5 options, commonsense, no context
  - OpenBookQA     (allenai/openbookqa)      4 options, science facts, no context
  - ARC-Easy       (allenai/ai2_arc)         3-5 options, grade-school science, no context
  - ARC-Challenge  (allenai/ai2_arc)         3-5 options, harder science, no context
  - HellaSwag      (Rowan/hellaswag)         4 options, commonsense "what
                                              happens next", no context

Every record carries a `context` field (empty string when the source has
none) so training/eval code can uniformly build "context + question" as
the input, with context-free sources naturally degrading to "just the
question."

Some official test splits are unlabeled (answer hidden for a leaderboard
-- true of CommonsenseQA and HellaSwag). To get a usable, labeled corpus
from every source uniformly, this pools every split that DOES have a
real answer label, then makes our own deterministic 80/10/10 train/val/test
split per source (fixed seed), rather than relying on each dataset's own,
inconsistent split scheme.
"""
import random

from datasets import load_dataset

SOURCES = ["race", "sciq", "commonsense_qa", "openbookqa", "arc_easy", "arc_challenge", "hellaswag"]


def _split_pool(records, seed):
    """Deterministic 80/10/10 split of a list of (context, question, options, answer_idx) tuples."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = int(n * 0.1)
    n_test = int(n * 0.1)
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test:n_test + n_val],
        "train": shuffled[n_test + n_val:],
    }


def _labeled_splits(name, cfg, answer_key):
    """Loads every split of a dataset and keeps only rows where `answer_key`
    is a non-empty label -- drops hidden-label test sets automatically."""
    all_rows = []
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
        except (ValueError, FileNotFoundError):
            continue
        for ex in ds:
            if str(ex.get(answer_key, "")).strip():
                all_rows.append(ex)
    return all_rows


def _load_race():
    """The main context-grounded source: a real passage (`article`) every
    question depends on -- this is the shape the actual target capability
    needs ("given context, choose the right option"), not a bare-trivia
    question. English-exam reading comprehension, middle + high school
    difficulty pooled together (the "all" config)."""
    rows = _labeled_splits("ehovy/race", "all", "answer")
    out = []
    letters = ["A", "B", "C", "D"]
    for ex in rows:
        if ex["answer"] not in letters or len(ex["options"]) != 4:
            continue
        answer_idx = letters.index(ex["answer"])
        out.append((ex["article"], ex["question"], ex["options"], answer_idx))
    return out


def _load_commonsense_qa():
    rows = _labeled_splits("tau/commonsense_qa", None, "answerKey")
    out = []
    for ex in rows:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        if ex["answerKey"] not in labels:
            continue
        answer_idx = labels.index(ex["answerKey"])
        out.append(("", ex["question"], texts, answer_idx))
    return out


def _load_openbookqa():
    rows = _labeled_splits("allenai/openbookqa", "main", "answerKey")
    out = []
    for ex in rows:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        if ex["answerKey"] not in labels:
            continue
        answer_idx = labels.index(ex["answerKey"])
        out.append(("", ex["question_stem"], texts, answer_idx))
    return out


def _load_arc(cfg):
    rows = _labeled_splits("allenai/ai2_arc", cfg, "answerKey")
    out = []
    for ex in rows:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        if ex["answerKey"] not in labels or len(texts) < 2:
            continue
        answer_idx = labels.index(ex["answerKey"])
        out.append(("", ex["question"], texts, answer_idx))
    return out


def _load_sciq():
    """Restores the `support` passage as context -- SciQ's questions were
    generated FROM that passage, so it's exactly the "given context,
    choose the right option" shape, not a bare-trivia question."""
    rows = _labeled_splits("allenai/sciq", None, "correct_answer")
    out = []
    rng = random.Random(42)
    for ex in rows:
        options = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled_options = [options[i] for i in order]
        answer_idx = order.index(0)
        out.append((ex.get("support", "") or "", ex["question"], shuffled_options, answer_idx))
    return out


def _load_hellaswag():
    """No separate context field -- `ctx` IS the question (the setup text
    to complete), there's no extra passage beyond it."""
    rows = _labeled_splits("Rowan/hellaswag", None, "label")
    out = []
    for ex in rows:
        try:
            answer_idx = int(ex["label"])
        except (ValueError, TypeError):
            continue
        if not (0 <= answer_idx < len(ex["endings"])):
            continue
        out.append(("", ex["ctx"], ex["endings"], answer_idx))
    return out


_LOADERS = {
    "race": (_load_race, 1000),
    "sciq": (_load_sciq, 1005),
    "commonsense_qa": (_load_commonsense_qa, 1001),
    "openbookqa": (_load_openbookqa, 1002),
    "arc_easy": (lambda: _load_arc("ARC-Easy"), 1003),
    "arc_challenge": (lambda: _load_arc("ARC-Challenge"), 1004),
    "hellaswag": (_load_hellaswag, 1006),
}


def _clean(result):
    """Two real data-quality problems, same category as intent_corpus's
    (see data/README.md):

    1. Duplicate option text within one example (~0.2% of rows) -- an
       upstream source-dataset defect (mostly CommonsenseQA/OpenBookQA).
       If two options are the literal same string, a text-only embedding
       model cannot possibly distinguish "the correct one" from "the
       identical-text wrong one" -- these examples are unanswerable by
       construction, not just hard. Dropped outright.

    2. Cross-split ambiguity from bare, context-free questions (~0.3% of
       the context-free sources): some short questions ("which is true?",
       "more sunlight will be absorbed by") appear verbatim across
       multiple different context-free source items that had DIFFERENT
       correct answers depending on information that source didn't carry
       forward. Now that context-bearing sources (RACE, SciQ's support)
       keep their disambiguating passage, this collision is keyed on
       (context, question) together -- two examples only collide if
       BOTH the passage and the question text are identical, so a real
       passage automatically resolves what a bare question couldn't.
       Unlike intent_corpus's leakage (same text, same true label, just
       needs priority-based split assignment), a genuine (context,
       question) collision with different answers is unresolvable given
       the fields kept, so every example under it is dropped everywhere.

    Remaining EXACT duplicates (same context+question, same correct-answer
    text, genuinely just repeated) are then collapsed by priority
    test > val > train, same convention as intent_corpus.
    """
    from collections import defaultdict

    def key_of(r):
        return (r.get("context", "").strip().lower(), r["question"].strip().lower())

    answer_by_q = defaultdict(set)
    for rows in result.values():
        for r in rows:
            answer_by_q[key_of(r)].add(r["options"][r["answer_idx"]].strip().lower())
    ambiguous = {q for q, answers in answer_by_q.items() if len(answers) > 1}

    cleaned = {}
    for split_name, rows in result.items():
        kept = []
        for r in rows:
            if len(set(r["options"])) != len(r["options"]):
                continue
            if key_of(r) in ambiguous:
                continue
            kept.append(r)
        cleaned[split_name] = kept

    seen_keys = set()
    final = {"train": [], "val": [], "test": []}
    for split_name in ["test", "val", "train"]:  # priority order
        for r in cleaned[split_name]:
            key = key_of(r) + (r["options"][r["answer_idx"]].strip().lower(),)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            final[split_name].append(r)
    return final


def build_mcq_dataset():
    """Returns {"train": [...], "val": [...], "test": [...]}, each a list
    of dicts {"context": str (empty if the source has none), "question":
    str, "options": [str, ...], "answer_idx": int, "source": str}."""
    result = {"train": [], "val": [], "test": []}
    for source in SOURCES:
        loader, seed = _LOADERS[source]
        records = loader()
        splits = _split_pool(records, seed)
        for split_name, items in splits.items():
            for context, question, options, answer_idx in items:
                result[split_name].append({
                    "context": context,
                    "question": question,
                    "options": options,
                    "answer_idx": answer_idx,
                    "source": source,
                })
    return _clean(result)

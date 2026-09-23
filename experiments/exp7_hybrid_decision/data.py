"""Experiment 7 data pipeline: turns raw rows from every source corpus into
`model.PackedExample` objects, applying the Sec 4.1 augmentation table:
  - option count N (log-uniform 2-128, 5% at 255)
  - modality dropout (20% vector-only, 20% text-only, 60% both)
  - option order shuffling
  - option rendering variation (intent task only -- other sources already
    carry natural-language option text)
  - instruction paraphrasing
  - distractor sampling, with a lexical positive-aware filter

Also implements the two dataset-level fixes from NOTES.md Sec 5:
  - Banking77 held out of `intent_corpus` entirely (§5.3) -- done here at
    load time, non-destructively (data/intent_corpus/*.jsonl is untouched).
  - HellaSwag downweighted from ~57% to ~15% of the MCQ pool (§5.1).

New corpora (bool/score/label-diversity/typed-decisions) are read from
data/exp7_*/*.jsonl, built by scripts/build_exp7_data.py -- see that file
for provenance. Loading here stays disk-only, same convention as
src/local_data.py.
"""
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from local_data import load_intent_corpus, load_mcq_corpus, load_qqp_pairs  # noqa: E402
from dataset_v7 import _humanize, raw_intent_name, sample_outcome_description  # noqa: E402

from model import PackedExample  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# --------------------------------------------------------------------------
# Instruction paraphrase bank (Sec 4.1) -- one entry can be "" (empty
# instructions), so the model also learns to work from context+options alone.
# --------------------------------------------------------------------------

INSTRUCTION_BANK = {
    "choice": [
        "Which category best describes what this is about, or where should it be routed?",
        "Pick the option that best matches this request.",
        "Which of the following applies here?",
        "Choose the single best-fitting category.",
        "Where should this be routed?",
        "",
    ],
    "bool": [
        "Does this apply? Answer yes or no.",
        "Is the following true?",
        "Answer yes or no.",
        "",
    ],
    "score": [
        "Rate this on the given scale, choosing the closest level.",
        "Pick the ordinal level that best fits.",
        "How would you rate this?",
        "",
    ],
}


def sample_instructions(qtype: str, rng: random.Random) -> str:
    return rng.choice(INSTRUCTION_BANK[qtype])


# --------------------------------------------------------------------------
# Per-source instruction bank for diversity_corpus -- fixes a second,
# previously-unflagged format-shift confound, the same category of bug as
# OPTION_RENDER_MODES below but on the INSTRUCTIONS field instead of the
# option text. Found by direct comparison after the first external-
# benchmark run: build_diversity_example was calling sample_instructions
# ("choice", ...) -- the SAME generic, routing-flavored bank used for
# every domain (intent/banking included) -- for every diversity source,
# regardless of what kind of text it actually is. At eval time,
# eval.py's run_external_benchmarks asks genuinely task-specific questions
# ("Which category best describes this news article?", "What emotion does
# this text express?") that the model had literally never seen phrased
# that way during training. Measured effect of the OPTION-side version of
# this same bug: AG News 50.4% (exp6, no option-style variation) vs. 74.4%
# (exp7, with it) -- the instruction-side version is the same kind of gap,
# unmeasured until now but mechanistically identical, and DAIR Emotion's
# comparatively strong 59.0% (vs. Laya 59.5%) despite ALSO never seeing
# "What emotion does this text express?" during training is best explained
# by GoEmotions' option labels happening to already match EMOTION_OPTIONS
# verbatim ("joy", "anger", "fear", ...) -- carrying the transfer on the
# option side alone. Fixing the instruction side too should help across
# the board, AG News most of all since it had neither match.
# --------------------------------------------------------------------------

DIVERSITY_INSTRUCTION_BANK = {
    "dbpedia14": [
        "What type of entity is this text about?",
        "Which category of entity does this describe?",
        "What kind of thing is being described here?",
        "",
    ],
    "trec": [
        "What type of answer does this question expect?",
        "Which category best classifies this question?",
        "What kind of information is this question asking for?",
        "",
    ],
    "yahoo_answers": [
        "Which topic does this question belong to?",
        "What subject area is this about?",
        "Which category best fits this question?",
        "",
    ],
    "20_newsgroups": [
        "Which topic or discussion group does this post belong to?",
        "What subject is this text about?",
        "Which category best fits this post?",
        "",
    ],
    "goemotions": [
        "What emotion does this text express?",
        "How does the writer feel?",
        "Which emotion best fits this text?",
        "",
    ],
}


def sample_diversity_instructions(source: str, rng: random.Random) -> str:
    """Falls back to the generic 'choice' bank for any source not listed
    above (defensive -- e.g. a future diversity source added without also
    updating this dict shouldn't crash, just lose this specific fix for
    that one source until it's added here too)."""
    bank = DIVERSITY_INSTRUCTION_BANK.get(source, INSTRUCTION_BANK["choice"])
    return rng.choice(bank)


# --------------------------------------------------------------------------
# Option rendering (intent task only) -- fixes the exact format-shift bug
# found in exp6's eval_external_benchmarks.py: training only ever showed 8
# templates, external eval fed bare label strings. Randomizing rendering
# mode per example removes that confound permanently (Sec 4.1).
# --------------------------------------------------------------------------

OPTION_RENDER_MODES = ["bare", "template", "description", "label_description"]


def render_intent_option(raw_name: str, mode: str, rng: random.Random) -> str:
    phrase = _humanize(raw_name)
    if mode == "bare":
        return phrase
    if mode == "template":
        # dataset_v7.sample_outcome_description does template + domain-
        # synonym substitution together (card <-> bank card, etc.) --
        # reused rather than reinventing that diversity mechanism here.
        return sample_outcome_description(raw_name, rng)
    if mode == "description":
        return f"The user wants help with: {phrase}."
    if mode == "label_description":
        return f"{phrase} -- {sample_outcome_description(raw_name, rng)}"
    raise ValueError(mode)


# --------------------------------------------------------------------------
# Modality dropout (Sec 4.1) -- the single detail NOTES.md flags as least
# safe to skip. Decided once per example, applied consistently to every
# option in that example (not per-option -- a real request either has full
# option text or it doesn't).
# --------------------------------------------------------------------------

def sample_modality(rng: random.Random, p_vector_only: float = 0.2,
                     p_text_only: float = 0.2) -> Tuple[bool, bool]:
    r = rng.random()
    if r < p_vector_only:
        return False, True   # use_text=False, use_vector=True
    if r < p_vector_only + p_text_only:
        return True, False   # use_text=True, use_vector=False
    return True, True


# --------------------------------------------------------------------------
# Option-count sampling (Sec 4.1) -- log-uniform 2-128, 5% at 255.
# --------------------------------------------------------------------------

def sample_n_options(rng: random.Random, lo: int = 2, hi: int = 128, p_max: float = 0.05,
                      max_n: int = 255) -> int:
    if rng.random() < p_max:
        return max_n
    import math
    log_lo, log_hi = math.log(lo), math.log(hi)
    return int(round(math.exp(rng.uniform(log_lo, log_hi))))


# --------------------------------------------------------------------------
# Positive-aware-ish distractor filtering -- a lexical approximation of
# NV-Retriever's positive-aware hard-negative filtering (Sec 3 Tier 3 idea
# from the earlier ideation discussion). We don't have a trained similarity
# model available to mine "secretly-correct" distractors properly, so this
# uses token-set Jaccard overlap between humanized label names as a cheap
# stand-in, and is documented here as exactly that -- a heuristic, not the
# real thing. Drops candidates that look suspiciously close to the true
# label; falls back to keeping the least-similar available ones if that
# would leave too few candidates.
# --------------------------------------------------------------------------

def _token_set(name: str) -> set:
    return set(_humanize(name).lower().replace("_", " ").split())


def positive_aware_distractors(true_label: str, pool: List[str], k: int, rng: random.Random,
                                jaccard_threshold: float = 0.5) -> List[str]:
    true_tokens = _token_set(true_label)
    scored = []
    for cand in pool:
        if cand == true_label:
            continue
        cand_tokens = _token_set(cand)
        union = true_tokens | cand_tokens
        jaccard = len(true_tokens & cand_tokens) / len(union) if union else 0.0
        scored.append((jaccard, cand))
    safe = [c for j, c in scored if j < jaccard_threshold]
    if len(safe) >= k:
        return rng.sample(safe, k)
    # Not enough "safe" candidates (a genuinely small or tightly-clustered
    # label pool) -- fall back to the least-similar ones available so
    # sampling never crashes, rather than silently relaxing the filter.
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:k]]


# --------------------------------------------------------------------------
# Intent task (Choice)
# --------------------------------------------------------------------------

@dataclass
class IntentCorpus:
    train: List[Tuple[str, str]]
    val: List[Tuple[str, str]]
    test: List[Tuple[str, str]]
    test_oos: List[Tuple[str, str]]
    test_zero_shot: List[Tuple[str, str]]
    seen_labels: List[str]
    all_labels: List[str]
    zero_shot_labels: List[str]
    banking77_holdout: List[Tuple[str, str]]   # held out entirely, Sec 5.3
    banking77_labels: List[str]


def load_intent_corpus_minus_banking77() -> IntentCorpus:
    """Loads intent_corpus and removes every Banking77 example AND label
    from the trainable pool -- non-destructive (data/intent_corpus/*.jsonl
    is never modified), so the exclusion lives entirely in this loader.
    The held-out Banking77 rows become the TRUE zero-shot high-cardinality
    eval set this project's headline claim rests on (NOTES.md Sec 5.3):
    exp6 scored 89.35% there having trained on it, which proved nothing;
    Laya scores 42.5% there zero-shot."""
    d = load_intent_corpus()

    def is_banking(label):
        return label.startswith("banking::")

    banking_labels = sorted(l for l in d["seen_labels"] if is_banking(l))
    banking_holdout = [(t, l) for t, l in (d["train"] + d["val"] + d["test"]) if is_banking(l)]

    train = [(t, l) for t, l in d["train"] if not is_banking(l)]
    val = [(t, l) for t, l in d["val"] if not is_banking(l)]
    test = [(t, l) for t, l in d["test"] if not is_banking(l)]
    seen_labels = [l for l in d["seen_labels"] if not is_banking(l)]
    all_labels = [l for l in d["all_labels"] if not is_banking(l)]

    return IntentCorpus(
        train=train, val=val, test=test, test_oos=d["test_oos"], test_zero_shot=d["test_zero_shot"],
        seen_labels=seen_labels, all_labels=all_labels, zero_shot_labels=d["zero_shot_labels"],
        banking77_holdout=banking_holdout, banking77_labels=banking_labels,
    )


def build_intent_example(text: str, true_label: str, label_pool: List[str], rng: random.Random,
                          n_target: Optional[int] = None) -> PackedExample:
    n = n_target or sample_n_options(rng)
    n = min(n, len(label_pool))
    distractor_labels = positive_aware_distractors(true_label, [l for l in label_pool if l != true_label],
                                                     n - 1, rng)
    option_labels = distractor_labels + [true_label]
    rng.shuffle(option_labels)
    answer_idx = option_labels.index(true_label)

    mode = rng.choice(OPTION_RENDER_MODES)
    option_texts = [render_intent_option(raw_intent_name(l), mode, rng) for l in option_labels]

    use_text, use_vector = sample_modality(rng)
    return PackedExample(
        context=text, instructions=sample_instructions("choice", rng), option_texts=option_texts,
        qtype="choice", answer_idx=answer_idx, use_text=use_text, use_vector=use_vector,
        source="intent",
    )


# --------------------------------------------------------------------------
# MCQ task (Choice, context-grounded)
# --------------------------------------------------------------------------

HELLASWAG_TARGET_FRACTION = 0.15  # was ~57% of mcq_corpus train (Sec 5.1)


def load_mcq_corpus_reweighted(seed: int = 4242) -> Dict[str, List[dict]]:
    """Downsamples HellaSwag (adversarially constructed against sub-1B
    models, per exp6's own per-source breakdown: 38.9%, on-trend for this
    size class before any training benefit) from ~57% to ~15% of the MCQ
    training pool, so most of the MCQ gradient isn't spent on something
    this backbone provably can't learn. val/test are left untouched --
    reweighting only changes the TRAINING distribution, not what's measured."""
    d = load_mcq_corpus()
    rng = random.Random(seed)
    train = d["train"]
    hella = [ex for ex in train if ex["source"] == "hellaswag"]
    other = [ex for ex in train if ex["source"] != "hellaswag"]
    if not other:
        return d
    # Solve for the hella count that makes it HELLASWAG_TARGET_FRACTION of
    # the new total, given `other` stays fixed size:
    #   target = h / (h + len(other))  =>  h = target * len(other) / (1 - target)
    target_hella_n = int(HELLASWAG_TARGET_FRACTION * len(other) / (1 - HELLASWAG_TARGET_FRACTION))
    target_hella_n = min(target_hella_n, len(hella))
    hella_kept = rng.sample(hella, target_hella_n)
    new_train = other + hella_kept
    rng.shuffle(new_train)
    return {"train": new_train, "val": d["val"], "test": d["test"]}


MCQ_FOREIGN_DISTRACTOR_MAX_N = 20  # cap on how far we pad small native option
# sets with foreign distractors -- kept modest (native counts are 3-5) so
# padding doesn't dominate the signal or blow up compute; NOTES.md Sec 5.1
# documents this as a known limitation vs. the intent task's full 2-128 range.


def build_mcq_example(ex: dict, foreign_pool_texts: List[str], rng: random.Random) -> PackedExample:
    context = ex["context"] or ""
    options = list(ex["options"])
    true_text = options[ex["answer_idx"]]

    n_target = min(sample_n_options(rng, hi=MCQ_FOREIGN_DISTRACTOR_MAX_N), len(foreign_pool_texts) + len(options))
    if n_target > len(options) and foreign_pool_texts:
        n_extra = min(n_target - len(options), len(foreign_pool_texts))
        extra = rng.sample(foreign_pool_texts, n_extra)
        options = options + extra

    rng.shuffle(options)
    answer_idx = options.index(true_text)

    use_text, use_vector = sample_modality(rng)
    return PackedExample(
        context=context, instructions=f"{sample_instructions('choice', rng)} {ex['question']}".strip(),
        option_texts=options, qtype="choice", answer_idx=answer_idx,
        use_text=use_text, use_vector=use_vector, source=f"mcq:{ex['source']}",
    )


# --------------------------------------------------------------------------
# QQP paraphrase pairs, reframed as a Bool question (Sec 3.3 note in
# NOTES.md carries the auxiliary loss forward; the unified single-sequence
# architecture has no separate compatibility_pairwise method the way exp6's
# dual encoder did, so QQP is reframed to fit the same packed-sequence
# mechanism everything else uses, instead of getting a bespoke code path).
# --------------------------------------------------------------------------

def build_qqp_example(text1: str, text2: str, is_paraphrase: bool, rng: random.Random) -> PackedExample:
    use_text, use_vector = sample_modality(rng)
    return PackedExample(
        context=text1, instructions=f"Is this a paraphrase of: \"{text2}\"?",
        option_texts=["no", "yes"], qtype="bool", answer_idx=int(is_paraphrase),
        use_text=use_text, use_vector=use_vector, source="qqp",
    )


# --------------------------------------------------------------------------
# New corpora: bool / score / label-diversity / typed-decisions
# (built by scripts/build_exp7_data.py; disk-only loaders, same convention
# as src/local_data.py)
# --------------------------------------------------------------------------

def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bool_corpus() -> Dict[str, List[dict]]:
    d = os.path.join(DATA_DIR, "exp7_bool_corpus")
    return {s: _read_jsonl(os.path.join(d, f"{s}.jsonl")) for s in ("train", "val", "test")}


def load_score_corpus() -> Dict[str, List[dict]]:
    d = os.path.join(DATA_DIR, "exp7_score_corpus")
    return {s: _read_jsonl(os.path.join(d, f"{s}.jsonl")) for s in ("train", "val", "test")}


def load_diversity_corpus() -> Dict[str, List[dict]]:
    d = os.path.join(DATA_DIR, "exp7_diversity_corpus")
    return {s: _read_jsonl(os.path.join(d, f"{s}.jsonl")) for s in ("train", "val", "test")}


def load_typed_decisions() -> List[dict]:
    """Eval-only (NOTES.md Sec 5.3) -- never mixed into training."""
    path = os.path.join(DATA_DIR, "exp7_typed_decisions", "test.jsonl")
    return _read_jsonl(path)


def build_bool_example(row: dict, rng: random.Random) -> PackedExample:
    use_text, use_vector = sample_modality(rng)
    return PackedExample(
        context=row["context"], instructions=row.get("question") or sample_instructions("bool", rng),
        option_texts=["no", "yes"], qtype="bool", answer_idx=int(row["answer_idx"]),
        use_text=use_text, use_vector=use_vector, source=f"bool:{row.get('source', '')}",
    )


def build_score_example(row: dict, rng: random.Random) -> PackedExample:
    n = row["num_levels"]
    option_texts = [f"level {i + 1} of {n}" for i in range(n)]
    use_text, use_vector = sample_modality(rng)
    return PackedExample(
        context=row["context"], instructions=row.get("question") or sample_instructions("score", rng),
        option_texts=option_texts, qtype="score", answer_idx=int(row["answer_idx"]),
        use_text=use_text, use_vector=use_vector, source=f"score:{row.get('source', '')}",
    )


def build_diversity_example(row: dict, other_label_pool: List[str], rng: random.Random) -> PackedExample:
    """row: {"text", "label", "labels" (this dataset's full label set)}.
    Padded toward a sampled target N with labels borrowed from OTHER
    diversity datasets when the native label set is small -- a legitimate
    (if easier) high-cardinality regime, documented in NOTES.md Sec 5.2 as
    complementary to, not a substitute for, the Banking77 same-domain
    high-cardinality eval."""
    options = list(row["labels"])
    true_text = row["label"]
    n_target = sample_n_options(rng, hi=64)
    if n_target > len(options) and other_label_pool:
        pool = [l for l in other_label_pool if l not in options]
        n_extra = min(n_target - len(options), len(pool))
        if n_extra > 0:
            options = options + rng.sample(pool, n_extra)
    rng.shuffle(options)
    answer_idx = options.index(true_text)
    use_text, use_vector = sample_modality(rng)
    return PackedExample(
        context=row["text"], instructions=sample_diversity_instructions(row.get("source", ""), rng),
        option_texts=options, qtype="choice", answer_idx=answer_idx,
        use_text=use_text, use_vector=use_vector, source=f"diversity:{row.get('source', '')}",
    )


def typed_decision_to_packed(row: dict) -> PackedExample:
    """typed-decisions rows are already single-question, fixed-option --
    no augmentation applied (this is an EVAL set, Sec 5.3)."""
    return PackedExample(
        context=row["context"], instructions=row["question"], option_texts=row["options"],
        qtype=row.get("qtype", "choice"), answer_idx=int(row["answer_idx"]),
        use_text=True, use_vector=True, source="typed_decisions",
    )

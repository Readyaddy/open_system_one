"""Iteration/experiment dataset v7: a genuinely more diverse, larger
combined intent corpus, built specifically in response to a real gap
found in every prior dataset in this project.

THE PROBLEM THIS FIXES: every dataset used so far (CLINC150, Banking77,
SNIPS) is short, voice-assistant-style, crowdsourced in a similar way --
one register of text. And the "paraphrase diversity" the model ever saw
during training came from 8 fixed template sentences plus a small
hand-built synonym dictionary (see sample_outcome_description below) --
not real diverse writing. Full-fine-tuning a large pretrained model
(e.g. an LLM-derived encoder) on data this narrow risks the model learning
to be invariant to OUR 8 templates specifically, not genuine paraphrase
invariance -- and risks catastrophic forgetting of whatever broad
pretrained knowledge motivated using a bigger backbone in the first
place, since narrow data gives gradient descent every opportunity to
overwrite it.

THE FIX, two parts:
  1. MORE SOURCES, not just more examples of the same three datasets:
     - CLINC150 (150 intents, 10 broad domains)
     - Banking77 (77 fine-grained banking intents)
     - SNIPS (7 assistant-command intents)
     - HWU64 (64 intents, ~11k examples, different crowdsourcing
       methodology -- alarms, calendar, cooking, email, IoT, music, etc.)
     - Amazon MASSIVE en-US (60 intents, ~16.5k examples, 18 domains --
       Amazon's own crowdsourced multilingual assistant corpus, a
       genuinely independent data collection effort from the other four)
     Combined: 358 intents (before the zero-shot holdout), ~53k+ training
     examples -- both more diverse AND more numerous, addressing both
     complaints (data too narrow, too few points) at once, not trading
     one for the other.
  2. A real paraphrase-pair dataset (QQP) as an AUXILIARY training
     signal, kept in a separate module (see paraphrase_aux.py) --
     genuine human-written paraphrases, independent of anything in this
     file, specifically to teach paraphrase invariance directly rather
     than hoping template augmentation is enough.
"""
import random
import re
from datasets import load_dataset

OOS_LABEL = "oos"
ZERO_SHOT_FRACTION = 0.15
ZERO_SHOT_SEED = 77

TEMPLATES = [
    "The user wants help with: {x}.",
    "This request is about {x}.",
    "Please route this to: {x}.",
    "This ticket concerns {x}.",
    "The customer is asking about {x}.",
    "This is related to {x}.",
    "Category: {x}.",
    "This message should be handled as: {x}.",
]

SYNONYMS = {
    "help": ["help", "assistance", "support"], "card": ["card", "bank card"],
    "account": ["account", "bank account"], "payment": ["payment", "charge", "transaction"],
    "transfer": ["transfer", "money transfer", "wire"], "balance": ["balance", "account balance"],
    "refund": ["refund", "money back"], "cash": ["cash", "money"],
    "phone": ["phone", "mobile", "cellphone"], "verify": ["verify", "confirm", "check"],
    "fee": ["fee", "charge"], "limit": ["limit", "cap"], "not": ["not", "isn't"],
    "lost": ["lost", "missing"], "stolen": ["stolen", "taken"],
    "declined": ["declined", "rejected", "denied"], "pending": ["pending", "in progress"],
    "top": ["top", "load"], "withdrawal": ["withdrawal", "cash withdrawal"],
    "exchange": ["exchange", "conversion"], "playlist": ["playlist", "music queue"],
    "weather": ["weather", "forecast"], "restaurant": ["restaurant", "place to eat"],
    "book": ["book", "reserve"], "movie": ["movie", "film"], "song": ["song", "track"],
    "alarm": ["alarm", "wake-up reminder"], "calendar": ["calendar", "schedule"],
    "recipe": ["recipe", "cooking instructions"], "email": ["email", "message"],
    "volume": ["volume", "sound level"], "news": ["news", "current events"],
    "light": ["light", "lighting"], "question": ["question", "query"],
}


def _humanize(name: str) -> str:
    name = re.sub(r"[_?]", " ", name)
    name = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def sample_outcome_description(intent_name: str, rng: random.Random) -> str:
    if intent_name == OOS_LABEL:
        return rng.choice([
            "This request does not match any known category, it is out of scope.",
            "This message is unrelated to any of the known request types.",
            "This does not belong to any recognized category.",
        ])
    phrase = _humanize(intent_name)
    words = phrase.split()
    out = []
    for w in words:
        if w in SYNONYMS and rng.random() < 0.4:
            out.append(rng.choice(SYNONYMS[w]))
        else:
            out.append(w)
    phrase = " ".join(out)
    return rng.choice(TEMPLATES).format(x=phrase)


def base_description(intent_name: str) -> str:
    if intent_name == OOS_LABEL:
        return "This request does not match any known category, it is out of scope."
    return f"The user wants help with: {_humanize(intent_name)}."


def load_clinc150():
    ds = load_dataset("clinc_oos", "plus")
    label_names = ds["train"].features["intent"].names
    train = [(ex["text"], label_names[ex["intent"]]) for ex in ds["train"]]
    val = [(ex["text"], label_names[ex["intent"]]) for ex in ds["validation"]]
    test = [(ex["text"], label_names[ex["intent"]]) for ex in ds["test"]]
    return train, val, test, label_names


def load_banking77():
    ds = load_dataset("mteb/banking77")
    all_train = list(ds["train"])
    rng = random.Random(42)
    rng.shuffle(all_train)
    n_val = 1000
    val_rows, train_rows = all_train[:n_val], all_train[n_val:]
    test_rows = list(ds["test"])
    to_ex = lambda rows: [(r["text"], r["label_text"]) for r in rows]
    label_names = sorted(set(r["label_text"] for r in all_train))
    return to_ex(train_rows), to_ex(val_rows), to_ex(test_rows), label_names


def load_snips():
    ds = load_dataset("benayas/snips")
    all_train = list(ds["train"])
    rng = random.Random(43)
    rng.shuffle(all_train)
    n_val = 700
    val_rows, train_rows = all_train[:n_val], all_train[n_val:]
    test_rows = list(ds["test"])
    to_ex = lambda rows: [(r["text"], r["category"]) for r in rows]
    label_names = sorted(set(r["category"] for r in all_train))
    return to_ex(train_rows), to_ex(val_rows), to_ex(test_rows), label_names


def load_hwu64():
    """FastFit/hwu_64 -- ships its own train/validation/test split with
    readable string labels (e.g. 'alarm query'), unlike DeepPavlov/hwu64
    which only has integer labels with no name mapping."""
    ds = load_dataset("FastFit/hwu_64")
    to_ex = lambda split: [(r["text"], r["label"]) for r in ds[split]]
    label_names = sorted(set(ds["train"]["label"]))
    return to_ex("train"), to_ex("validation"), to_ex("test"), label_names


def load_massive():
    """SetFit/amazon_massive_intent_en-US -- Amazon's own crowdsourced
    multilingual assistant corpus (English subset here), 60 intents
    across 18 domains, collected independently of every other dataset in
    this pool."""
    ds = load_dataset("SetFit/amazon_massive_intent_en-US")
    to_ex = lambda split: [(r["text"], r["label_text"]) for r in ds[split]]
    label_names = sorted(set(ds["train"]["label_text"]))
    return to_ex("train"), to_ex("validation"), to_ex("test"), label_names


def _normalize_name(composite_label: str) -> str:
    name = composite_label.split("::", 1)[1]
    return name.lower().replace("_", " ").replace("-", " ")


def _build_merge_map(all_labels):
    """HWU64 and Amazon MASSIVE were built on the same underlying intent
    taxonomy (Amazon MASSIVE's schema explicitly extends the earlier NLU
    evaluation data HWU64 also derives from) -- checking after combining
    found 57 cases where two or three source datasets used the literal
    same intent name (e.g. 'hwu::alarm set' / 'massive::alarm_set'), 55 of
    them HWU/MASSIVE pairs. Treating these as separate classes would be a
    real labeling bug, not a diversity feature: the model would be taught
    a false distinction between two names for the same concept, and if
    one sibling landed in the zero-shot holdout while the other stayed in
    training, the "zero-shot" test would be contaminated -- the concept
    wouldn't really be unseen. Fix: merge same-named intents across
    sources into ONE canonical label, pooling their examples. This is a
    genuine improvement, not just a correction -- HWU64 and MASSIVE were
    collected by different crowdworkers, so pooling their phrasings of
    the same concept adds real surface-form diversity under one correct
    label, rather than fake conceptual diversity under two wrong ones.
    Returns {original_label: canonical_label}."""
    by_name = {}
    for l in all_labels:
        by_name.setdefault(_normalize_name(l), []).append(l)
    merge_map = {}
    for name, labels in by_name.items():
        canonical = sorted(labels)[0]  # deterministic choice
        for l in labels:
            merge_map[l] = canonical
    return merge_map


def build_combined_dataset():
    c_train, c_val, c_test, c_labels = load_clinc150()
    b_train, b_val, b_test, b_labels = load_banking77()
    s_train, s_val, s_test, s_labels = load_snips()
    h_train, h_val, h_test, h_labels = load_hwu64()
    m_train, m_val, m_test, m_labels = load_massive()

    tag = lambda examples, ds_name: [(t, f"{ds_name}::{l}") for t, l in examples]

    train = (tag(c_train, "clinc") + tag(b_train, "banking") + tag(s_train, "snips") +
             tag(h_train, "hwu") + tag(m_train, "massive"))
    val = (tag(c_val, "clinc") + tag(b_val, "banking") + tag(s_val, "snips") +
           tag(h_val, "hwu") + tag(m_val, "massive"))
    test = (tag(c_test, "clinc") + tag(b_test, "banking") + tag(s_test, "snips") +
            tag(h_test, "hwu") + tag(m_test, "massive"))

    raw_all_labels = (
        [f"clinc::{l}" for l in c_labels if l != OOS_LABEL] +
        [f"banking::{l}" for l in b_labels] +
        [f"snips::{l}" for l in s_labels] +
        [f"hwu::{l}" for l in h_labels] +
        [f"massive::{l}" for l in m_labels]
    )
    merge_map = _build_merge_map(raw_all_labels)
    remap = lambda examples: [(t, merge_map.get(l, l)) for t, l in examples]
    train, val, test = remap(train), remap(val), remap(test)
    all_labels = sorted(set(merge_map.values()))

    train = [(t, l) for t, l in train if not l.endswith(f"::{OOS_LABEL}")]
    val = [(t, l) for t, l in val if not l.endswith(f"::{OOS_LABEL}")]
    test_oos = [(t, l) for t, l in test if l.endswith(f"::{OOS_LABEL}")]
    test = [(t, l) for t, l in test if not l.endswith(f"::{OOS_LABEL}")]

    # Cross-source text deduplication. Checking after combining five
    # independently-collected datasets found the exact same utterance
    # text appearing in more than one split -- 1220 cases with matching
    # labels (straightforward train/test leakage: the model could just
    # memorize these instead of generalizing) and 40 cases with
    # CONFLICTING labels (deeper issue: exact-name merging above caught
    # same-named intents like 'alarm_set'/'alarm set', but not
    # semantically-identical, differently-named ones, e.g.
    # 'clinc::weather' vs 'hwu::weather query' for the literal same
    # phrase "what is the weather like"). Fix: keep each exact text in
    # only the highest-priority split it appears in (test > zero-shot
    # eval > oos > val > train), dropping it from every lower-priority
    # one -- this removes both problems at once, since after dedup no
    # text has a chance to carry two different labels across splits.
    def dedupe_by_priority(*splits):
        seen = set()
        out = []
        for split in splits:
            kept = [(t, l) for t, l in split if t not in seen]
            seen.update(t for t, _ in kept)
            out.append(kept)
        return out

    test, test_oos, val, train = dedupe_by_priority(test, test_oos, val, train)

    rng = random.Random(ZERO_SHOT_SEED)
    shuffled_labels = all_labels[:]
    rng.shuffle(shuffled_labels)
    n_zero_shot = int(len(shuffled_labels) * ZERO_SHOT_FRACTION)
    zero_shot_labels = set(shuffled_labels[:n_zero_shot])

    train_seen = [(t, l) for t, l in train if l not in zero_shot_labels]
    val_seen = [(t, l) for t, l in val if l not in zero_shot_labels]
    test_seen = [(t, l) for t, l in test if l not in zero_shot_labels]
    test_zero_shot = [(t, l) for t, l in test if l in zero_shot_labels]
    seen_labels = [l for l in all_labels if l not in zero_shot_labels]

    return {
        "train": train_seen, "val": val_seen, "test": test_seen,
        "test_oos": test_oos, "test_zero_shot": test_zero_shot,
        "seen_labels": seen_labels, "all_labels": all_labels,
        "zero_shot_labels": sorted(zero_shot_labels),
    }


def raw_intent_name(composite_label: str) -> str:
    return composite_label.split("::", 1)[1]

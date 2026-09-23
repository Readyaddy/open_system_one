"""Iteration 4 dataset: CLINC150 + Banking77 + SNIPS combined, with a real
ZERO-SHOT split.

Everything before this iteration tested generalization to unseen PHRASING
of a description for an intent that was still trained on (paraphrase
eval). That is a real but limited test. This iteration adds the harder,
more honest test explicitly requested: hold out a whole slice of intents
--- both their example questions AND their candidate descriptions ---
entirely out of training, and only introduce them at test time, mixed in
with the trained intents as distractors. Correctly routing to a class the
model has NEVER seen an example of, and never saw described in any form
during training, is the real test of "this is compatibility scoring, not a
classifier in disguise."

Three real datasets, for a genuinely more general model:
  - CLINC150 (150 intents, 10 broad domains: banking, travel, kitchen,
    auto, work, small talk, credit cards, home, utility, meta)
  - Banking77 (77 fine-grained banking intents, hard negatives)
  - SNIPS (7 intents: smart-home / media / booking commands, a distinct
    style from the other two -- assistant commands, not just questions)
234 intents combined, ~38k in-scope training examples before the
zero-shot holdout is removed.
"""
import random
import re
from datasets import load_dataset

OOS_LABEL = "oos"
ZERO_SHOT_FRACTION = 0.15
ZERO_SHOT_SEED = 7


def _humanize(name: str) -> str:
    # snake_case -> spaced, and CamelCase (SNIPS style) -> spaced.
    name = re.sub(r"[_?]", " ", name)
    name = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


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
    "help": ["help", "assistance", "support"],
    "card": ["card", "bank card"],
    "account": ["account", "bank account"],
    "payment": ["payment", "charge", "transaction"],
    "transfer": ["transfer", "money transfer", "wire"],
    "balance": ["balance", "account balance"],
    "refund": ["refund", "money back"],
    "cash": ["cash", "money"],
    "phone": ["phone", "mobile", "cellphone"],
    "verify": ["verify", "confirm", "check"],
    "fee": ["fee", "charge"],
    "limit": ["limit", "cap"],
    "not": ["not", "isn't"],
    "lost": ["lost", "missing"],
    "stolen": ["stolen", "taken"],
    "declined": ["declined", "rejected", "denied"],
    "pending": ["pending", "in progress"],
    "top": ["top", "load"],
    "withdrawal": ["withdrawal", "cash withdrawal"],
    "exchange": ["exchange", "conversion"],
    "playlist": ["playlist", "music queue"],
    "weather": ["weather", "forecast"],
    "restaurant": ["restaurant", "place to eat"],
    "book": ["book", "reserve"],
    "movie": ["movie", "film"],
    "song": ["song", "track"],
}


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
    template = rng.choice(TEMPLATES)
    return template.format(x=phrase)


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
    val_rows = all_train[:n_val]
    train_rows = all_train[n_val:]
    test_rows = list(ds["test"])

    def to_ex(rows):
        return [(r["text"], r["label_text"]) for r in rows]

    label_names = sorted(set(r["label_text"] for r in all_train))
    return to_ex(train_rows), to_ex(val_rows), to_ex(test_rows), label_names


def load_snips():
    ds = load_dataset("benayas/snips")
    all_train = list(ds["train"])
    rng = random.Random(43)
    rng.shuffle(all_train)
    n_val = 700
    val_rows = all_train[:n_val]
    train_rows = all_train[n_val:]
    test_rows = list(ds["test"])

    def to_ex(rows):
        return [(r["text"], r["category"]) for r in rows]

    label_names = sorted(set(r["category"] for r in all_train))
    return to_ex(train_rows), to_ex(val_rows), to_ex(test_rows), label_names


def build_combined_dataset():
    c_train, c_val, c_test, c_labels = load_clinc150()
    b_train, b_val, b_test, b_labels = load_banking77()
    s_train, s_val, s_test, s_labels = load_snips()

    def tag(examples, ds_name):
        return [(t, f"{ds_name}::{l}") for t, l in examples]

    train = tag(c_train, "clinc") + tag(b_train, "banking") + tag(s_train, "snips")
    val = tag(c_val, "clinc") + tag(b_val, "banking") + tag(s_val, "snips")
    test = tag(c_test, "clinc") + tag(b_test, "banking") + tag(s_test, "snips")

    all_labels = (
        [f"clinc::{l}" for l in c_labels if l != OOS_LABEL] +
        [f"banking::{l}" for l in b_labels] +
        [f"snips::{l}" for l in s_labels]
    )

    train = [(t, l) for t, l in train if not l.endswith(f"::{OOS_LABEL}")]
    val = [(t, l) for t, l in val if not l.endswith(f"::{OOS_LABEL}")]
    test_oos = [(t, l) for t, l in test if l.endswith(f"::{OOS_LABEL}")]
    test = [(t, l) for t, l in test if not l.endswith(f"::{OOS_LABEL}")]

    # --- Zero-shot split: hold out a fraction of intents ENTIRELY. ---
    rng = random.Random(ZERO_SHOT_SEED)
    shuffled_labels = all_labels[:]
    rng.shuffle(shuffled_labels)
    n_zero_shot = int(len(shuffled_labels) * ZERO_SHOT_FRACTION)
    zero_shot_labels = set(shuffled_labels[:n_zero_shot])
    seen_labels = [l for l in all_labels if l not in zero_shot_labels]

    train_seen = [(t, l) for t, l in train if l not in zero_shot_labels]
    val_seen = [(t, l) for t, l in val if l not in zero_shot_labels]
    test_seen = [(t, l) for t, l in test if l not in zero_shot_labels]
    test_zero_shot = [(t, l) for t, l in test if l in zero_shot_labels]

    return {
        "train": train_seen,
        "val": val_seen,
        "test": test_seen,
        "test_oos": test_oos,
        "test_zero_shot": test_zero_shot,
        "seen_labels": seen_labels,            # candidate pool used during training
        "all_labels": all_labels,               # seen + zero-shot, for the zero-shot eval candidate pool
        "zero_shot_labels": sorted(zero_shot_labels),
    }


def raw_intent_name(composite_label: str) -> str:
    return composite_label.split("::", 1)[1]

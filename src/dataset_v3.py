"""Iteration 3 dataset: CLINC150 + Banking77 combined, with outcome-side
description AUGMENTATION.

The iteration-2 finding: a pretrained backbone did not improve paraphrase
generalization (0.75 -> 0.73) because the outcome encoder only ever saw one
fixed description string per class. Cross-entropy over fixed anchor points
can be solved by memorizing distinguishable points, not by understanding
meaning. The fix implemented here: build several template phrasings +
lexical variation per class, and sample a different one every training
step, so the encoder is never allowed to lock onto one exact string.

Combining two real datasets also raises the difficulty honestly:
  - CLINC150: 150 intents, 10 broad domains, plus an out-of-scope class.
  - Banking77: 77 FINE-GRAINED banking intents that are semantically very
    close to each other (card_arrival vs card_delivery_estimate vs
    lost_or_stolen_card vs card_swallowed vs compromised_card ...) -- a
    much harder discrimination test than CLINC150 alone.
"""
import random
import re
from datasets import load_dataset

OOS_LABEL = "oos"

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

# Lightweight domain-synonym substitution applied to the humanized intent
# phrase before it's dropped into a template, to add lexical variety beyond
# just re-wrapping the same words in different templates.
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
}


def _humanize(name: str) -> str:
    return re.sub(r"[_?]", " ", name).strip()


def _apply_synonyms(phrase: str, rng: random.Random) -> str:
    words = phrase.split()
    out = []
    for w in words:
        key = w.lower()
        if key in SYNONYMS and rng.random() < 0.4:
            out.append(rng.choice(SYNONYMS[key]))
        else:
            out.append(w)
    return " ".join(out)


def sample_outcome_description(intent_name: str, rng: random.Random) -> str:
    if intent_name == OOS_LABEL:
        return rng.choice([
            "This request does not match any known category, it is out of scope.",
            "This message is unrelated to any of the known request types.",
            "This does not belong to any recognized category.",
        ])
    phrase = _humanize(intent_name)
    phrase = _apply_synonyms(phrase, rng)
    template = rng.choice(TEMPLATES)
    return template.format(x=phrase)


def base_description(intent_name: str) -> str:
    """Deterministic, non-augmented description -- used for eval so results
    are reproducible (no randomness at test time)."""
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


def build_combined_dataset():
    """Returns train/val/test example lists (text, "<dataset>::<label>") and
    the full list of composite label keys (excluding oos, which is tracked
    separately for the rejection metric)."""
    c_train, c_val, c_test, c_labels = load_clinc150()
    b_train, b_val, b_test, b_labels = load_banking77()

    def tag(examples, ds_name):
        return [(t, f"{ds_name}::{l}") for t, l in examples]

    train = tag(c_train, "clinc") + tag(b_train, "banking")
    val = tag(c_val, "clinc") + tag(b_val, "banking")
    test = tag(c_test, "clinc") + tag(b_test, "banking")

    in_scope_labels = (
        [f"clinc::{l}" for l in c_labels if l != OOS_LABEL] +
        [f"banking::{l}" for l in b_labels]
    )

    train_in = [(t, l) for t, l in train if not l.endswith(f"::{OOS_LABEL}")]
    val_in = [(t, l) for t, l in val if not l.endswith(f"::{OOS_LABEL}")]
    test_in = [(t, l) for t, l in test if not l.endswith(f"::{OOS_LABEL}")]
    test_oos = [(t, l) for t, l in test if l.endswith(f"::{OOS_LABEL}")]

    return {
        "train": train_in,
        "val": val_in,
        "test": test_in,
        "test_oos": test_oos,
        "labels": in_scope_labels,  # composite keys like "clinc::weather" / "banking::card_arrival"
    }


def raw_intent_name(composite_label: str) -> str:
    return composite_label.split("::", 1)[1]

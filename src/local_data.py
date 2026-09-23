"""Loads training data directly from data/*.jsonl -- no network calls, no
HuggingFace `datasets` cache dependency. This is the single source of
truth for training data from here on; dataset_v7.py / paraphrase_aux.py
(which hit the network) are what BUILT data/ once via
scripts/build_data_v7.py and are not needed again unless data/ is being
regenerated from scratch.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_intent_corpus():
    """Returns a dict with the same shape dataset_v7.build_combined_dataset()
    returned, but read straight from disk."""
    d = os.path.join(DATA_DIR, "intent_corpus")

    def split(name):
        rows = _read_jsonl(os.path.join(d, f"{name}.jsonl"))
        return [(r["text"], r["label"]) for r in rows]

    with open(os.path.join(d, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)

    return {
        "train": split("train"),
        "val": split("val"),
        "test": split("test"),
        "test_oos": split("test_oos"),
        "test_zero_shot": split("test_zero_shot"),
        "seen_labels": labels["seen_labels"],
        "all_labels": labels["all_labels"],
        "zero_shot_labels": labels["zero_shot_labels"],
    }


def load_mcq_corpus():
    """Returns a dict {"train": [...], "val": [...], "test": [...]}, each
    a list of dicts {"question", "options" (list[str]), "answer_idx"
    (int), "source"}, read straight from disk."""
    d = os.path.join(DATA_DIR, "mcq_corpus")

    def split(name):
        return _read_jsonl(os.path.join(d, f"{name}.jsonl"))

    return {"train": split("train"), "val": split("val"), "test": split("test")}


def load_qqp_pairs():
    """Returns (train_pairs, val_pairs), each a list of
    (text1, text2, is_paraphrase: bool), read straight from disk."""
    d = os.path.join(DATA_DIR, "qqp_paraphrase_pairs")

    def split(name):
        rows = _read_jsonl(os.path.join(d, f"{name}.jsonl"))
        return [(r["text1"], r["text2"], bool(r["is_paraphrase"])) for r in rows]

    return split("train"), split("val")


# Re-exported so callers of the old dataset_v7 module's helper functions
# (sample_outcome_description, base_description, raw_intent_name) don't
# need two import lines -- these are pure functions with no network
# dependency, so importing them from dataset_v7 is fine; only the
# network-hitting loaders above needed a local replacement.
#
# NOTE: does NOT touch sys.path here -- callers (e.g. an experiment's
# train.py) are responsible for putting src/ on sys.path themselves,
# since this module being importABLE already means src/ is on the path;
# inserting it again here previously re-prioritized src/ over a caller's
# own local directory, causing e.g. `from model import ...` in an
# experiment folder to resolve to src/model.py (a much older, unrelated
# file) instead of that experiment's own model.py.
from dataset_v7 import sample_outcome_description, base_description, raw_intent_name  # noqa: F401

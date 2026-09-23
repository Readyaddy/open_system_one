"""Experiment 3 dataset: identical small-option-set request framing to
experiment 2, PLUS the needs_human (Noul) and urgency (Score) questions
from iteration 5 -- packed into ONE request per example, matching Jev's
actual request shape (a state plus several heterogeneous questions
answered together). This is what the joint-sequence model (model.py)
needs as input: one state, one Choice question with a handful of options,
one Noul question, one Score question with ordered levels.

Reuses dataset_v5's heuristic needs_human/urgency labels (see that file
and ANALYSIS.md for the explicit caveat: these are keyword-rule
heuristics, not gold data) and dataset_v4's intent pool + augmentation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random

from dataset_v5 import build_multi_question_dataset
from dataset_v4 import base_description, raw_intent_name, sample_outcome_description

URGENCY_LEVEL_TEXTS = [
    "Very low urgency: informational or casual, no action needed soon.",
    "Low urgency: routine request, can be handled in normal course of business.",
    "Medium urgency: needs attention but not immediately time-critical.",
    "High urgency: time-sensitive, should be handled soon.",
    "Critical urgency: needs immediate attention, likely a security or safety issue.",
]
NEEDS_HUMAN_INSTRUCTION = "This request is sensitive or complex enough that it should be escalated to a human."
URGENCY_INSTRUCTION = "How urgent is this request?"
CHOICE_INSTRUCTION = "Which of the following best describes this request?"


def _dataset_of(composite_label: str) -> str:
    return composite_label.split("::", 1)[0]


def make_request(example, label_pool, label_by_dataset, rng: random.Random,
                  n_choice_options: int = 6, augment: bool = True):
    correct_label = example["intent_label"]
    correct_ds = _dataset_of(correct_label)

    n_hard = min(n_choice_options - 1, max(1, (n_choice_options - 1) * 2 // 3))
    same_ds_pool = [l for l in label_by_dataset.get(correct_ds, []) if l != correct_label]
    hard_negatives = rng.sample(same_ds_pool, min(n_hard, len(same_ds_pool)))

    remaining = n_choice_options - 1 - len(hard_negatives)
    other_pool = [l for l in label_pool if l != correct_label and l not in hard_negatives]
    random_negatives = rng.sample(other_pool, min(remaining, len(other_pool)))

    option_labels = hard_negatives + random_negatives + [correct_label]
    rng.shuffle(option_labels)
    correct_idx = option_labels.index(correct_label)

    def desc(label):
        name = raw_intent_name(label)
        return sample_outcome_description(name, rng) if augment else base_description(name)

    return {
        "state": example["text"],
        "choice_instructions": CHOICE_INSTRUCTION,
        "choice_options": [desc(l) for l in option_labels],
        "choice_correct_idx": correct_idx,
        "choice_correct_label": correct_label,
        "noul_instructions": NEEDS_HUMAN_INSTRUCTION,
        "noul_label": float(example["needs_human_label"]),
        "score_instructions": URGENCY_INSTRUCTION,
        "score_levels": URGENCY_LEVEL_TEXTS,
        "score_correct_idx": example["urgency_label"] - 1,
    }


def build_request_dataset(n_choice_options: int = 6, seed: int = 456):
    data = build_multi_question_dataset()
    seen_labels, all_labels = data["seen_labels"], data["all_labels"]
    zero_shot_labels = set(data["zero_shot_labels"])

    by_ds_seen, by_ds_all = {}, {}
    for l in seen_labels:
        by_ds_seen.setdefault(_dataset_of(l), []).append(l)
    for l in all_labels:
        by_ds_all.setdefault(_dataset_of(l), []).append(l)

    rng = random.Random(seed)

    def build(examples, label_pool, by_ds, augment):
        return [make_request(e, label_pool, by_ds, rng, n_choice_options, augment) for e in examples]

    train_req = build(data["train"], seen_labels, by_ds_seen, augment=True)
    val_req = build(data["val"], seen_labels, by_ds_seen, augment=False)
    test_req = build(data["test"], seen_labels, by_ds_seen, augment=False)
    zero_shot_req = build(data["test_zero_shot"], all_labels, by_ds_all, augment=False)

    return {
        "train": train_req, "val": val_req, "test": test_req, "test_zero_shot": zero_shot_req,
        "zero_shot_labels": zero_shot_labels, "n_choice_options": n_choice_options,
    }

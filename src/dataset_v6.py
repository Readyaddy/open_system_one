"""Iteration 6 dataset: reformats the same CLINC150+Banking77+SNIPS pool
(with dataset_v5's heuristic needs_human/urgency labels) into Jev-shaped
REQUESTS -- a state plus a small, per-request set of Choice options, a
Noul question, and a Score question -- instead of one giant fixed
234-way classification.

This matters architecturally, not just cosmetically: every Choice example
in TypeSafe's own docs has 3-6 options, and the 255-option figure is a
documented ceiling, not typical usage. Training against a small sampled
candidate set every time is a materially different (and more realistic)
task than "distinguish this example from all 233 other classes at once,"
and it's what the joint-sequence architecture (model_v6.py) is actually
built to process cheaply.

Distractor sampling mixes:
  - "hard" distractors: other intents from the SAME source dataset (e.g.
    other Banking77 intents), which tend to be semantically close --
    forces real discrimination, not just topic-level separation.
  - random distractors: intents from anywhere in the combined pool, for
    variety.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random

from dataset_v5 import build_multi_question_dataset, needs_human_label, urgency_label
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


def _dataset_of(composite_label: str) -> str:
    return composite_label.split("::", 1)[0]


def make_request(example, all_labels, label_by_dataset, rng: random.Random, n_choice_options: int = 6,
                  augment_choice_descriptions: bool = True):
    """Builds one Jev-shaped request dict from a raw (text, intent_label,
    needs_human_label, urgency_label) example."""
    correct_label = example["intent_label"]
    correct_ds = _dataset_of(correct_label)

    n_hard = min(n_choice_options - 1, max(1, (n_choice_options - 1) * 2 // 3))
    same_ds_pool = [l for l in label_by_dataset[correct_ds] if l != correct_label]
    hard_negatives = rng.sample(same_ds_pool, min(n_hard, len(same_ds_pool)))

    remaining_slots = n_choice_options - 1 - len(hard_negatives)
    other_pool = [l for l in all_labels if l != correct_label and l not in hard_negatives]
    random_negatives = rng.sample(other_pool, min(remaining_slots, len(other_pool)))

    distractors = hard_negatives + random_negatives
    option_labels = distractors + [correct_label]
    rng.shuffle(option_labels)
    correct_idx = option_labels.index(correct_label)

    def desc(label):
        name = raw_intent_name(label)
        if augment_choice_descriptions:
            return sample_outcome_description(name, rng)
        return base_description(name)

    option_texts = [desc(l) for l in option_labels]

    return {
        "state": example["text"],
        "choice_instructions": "Which of the following best describes this request?",
        "choice_options": option_texts,
        "choice_correct_idx": correct_idx,
        "noul_instructions": NEEDS_HUMAN_INSTRUCTION,
        "noul_label": float(example["needs_human_label"]),
        "score_instructions": URGENCY_INSTRUCTION,
        "score_levels": URGENCY_LEVEL_TEXTS,
        "score_correct_idx": example["urgency_label"] - 1,  # 0-4
    }


def build_request_dataset(n_choice_options: int = 6, seed: int = 123):
    data = build_multi_question_dataset()
    all_labels = data["all_labels"]
    seen_labels = data["seen_labels"]
    zero_shot_labels = set(data["zero_shot_labels"])

    label_by_dataset_all = {}
    for l in all_labels:
        label_by_dataset_all.setdefault(_dataset_of(l), []).append(l)
    label_by_dataset_seen = {}
    for l in seen_labels:
        label_by_dataset_seen.setdefault(_dataset_of(l), []).append(l)

    rng = random.Random(seed)

    def build_split(examples, label_pool, label_by_ds, augment):
        return [make_request(e, label_pool, label_by_ds, rng, n_choice_options, augment) for e in examples]

    train_requests = build_split(data["train"], seen_labels, label_by_dataset_seen, augment=True)
    val_requests = build_split(data["val"], seen_labels, label_by_dataset_seen, augment=False)
    test_requests = build_split(data["test"], seen_labels, label_by_dataset_seen, augment=False)
    # Zero-shot requests: distractor pool includes ALL labels (seen + unseen),
    # so the never-trained intent must be picked out from a realistic mixed set.
    zero_shot_requests = build_split(data["test_zero_shot"], all_labels, label_by_dataset_all, augment=False)

    return {
        "train": train_requests,
        "val": val_requests,
        "test": test_requests,
        "test_zero_shot": zero_shot_requests,
        "zero_shot_labels": zero_shot_labels,
    }

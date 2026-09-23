"""Iteration 5 dataset: same CLINC150 + Banking77 + SNIPS combined corpus
and zero-shot split as dataset_v4, PLUS two synthetic auxiliary questions
per example so a single context has multiple DIFFERENT typed questions to
answer at once -- this is what the multi-question architecture needs to
train on.

IMPORTANT, stated plainly: `needs_human` and `urgency` are NOT gold labels
from any dataset. They are simple keyword-rule heuristics derived from the
intent name, built only to give the multi-question architecture something
concrete and checkable to train and evaluate on. Treat any accuracy number
on these two heads as "did the model learn to reproduce our heuristic
rule", not "did the model learn true urgency" -- that distinction is kept
explicit in the eval output and the README, not glossed over.

Question types:
  - "intent"      : Choice, one of the (up to 234) combined intents. Same
                     task as v4's main task, same outcome descriptions.
  - "needs_human"  : Bool. Heuristic: True if the intent name contains a
                     keyword associated with fraud/security/loss/dispute/
                     account-termination, i.e. the kinds of things a real
                     triage system would actually escalate.
  - "urgency"      : Score, 5 discrete bins (1-5). Heuristic bucket by the
                     same keyword categories: security/fraud/loss highest,
                     informational/small-talk lowest.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataset_v4 import build_combined_dataset, raw_intent_name, base_description, sample_outcome_description

NEEDS_HUMAN_KEYWORDS = [
    "fraud", "stolen", "lost", "compromised", "dispute", "unauthorized",
    "blocked", "declined", "terminate", "complaint", "emergency", "swallowed",
    "damaged_card", "report", "not_working", "not_received", "not_recognised",
    "not_recognized", "wrong_", "failed", "reverted",
]

URGENCY_HIGH_KEYWORDS = [
    "fraud", "stolen", "compromised", "unauthorized", "swallowed",
    "declined", "blocked", "emergency", "lost_or_stolen",
]
URGENCY_LOW_KEYWORDS = [
    "greeting", "goodbye", "thank_you", "tell_joke", "meaning_of_life",
    "who_made_you", "weather", "date", "time", "fun_fact", "what_song",
    "what_is_your_name", "are_you_a_bot", "how_old_are_you",
]


def needs_human_label(intent_name: str) -> bool:
    name = intent_name.lower()
    return any(k in name for k in NEEDS_HUMAN_KEYWORDS)


def urgency_label(intent_name: str) -> int:
    """Returns an integer 1-5 (1 = lowest urgency, 5 = highest)."""
    name = intent_name.lower()
    if any(k in name for k in URGENCY_HIGH_KEYWORDS):
        return 5
    if needs_human_label(intent_name):
        return 4
    if any(k in name for k in URGENCY_LOW_KEYWORDS):
        return 1
    if "pending" in name or "balance" in name or "limit" in name or "fee" in name:
        return 3
    return 2


def build_multi_question_dataset():
    """Same splits as v4, with each example carrying all three labels."""
    data = build_combined_dataset()

    def annotate(examples):
        out = []
        for text, label in examples:
            intent = raw_intent_name(label)
            out.append({
                "text": text,
                "intent_label": label,
                "needs_human_label": needs_human_label(intent),
                "urgency_label": urgency_label(intent),
            })
        return out

    return {
        "train": annotate(data["train"]),
        "val": annotate(data["val"]),
        "test": annotate(data["test"]),
        "test_oos": annotate(data["test_oos"]),
        "test_zero_shot": annotate(data["test_zero_shot"]),
        "seen_labels": data["seen_labels"],
        "all_labels": data["all_labels"],
        "zero_shot_labels": data["zero_shot_labels"],
    }

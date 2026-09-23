"""The other side of the demo: exp7 doing the tasks it was actually trained
on, not the out-of-domain stretch that Snake is (see snake_laya/README.md
for why Snake needed the fatal-move filter and graded descriptors just to
be playable at all).

Ten text tasks, each pulled straight from exp7's own training corpora and
built with exp7's own training-time example builders (experiments/
exp7_hybrid_decision/data.py) -- not reimplemented here, so what's on
screen is provably the same distribution the checkpoint was scored on. Kept
fine-grained (a separate kind per MCQ source, rather than one blended "mcq")
because the whole point of adding more than one or two is to show a real
capability profile -- including where it's weak -- not just a highlight
reel. Measured per-source accuracy on the live exp7_latest checkpoint
(n=40-60/kind, one-off eval -- re-check if the checkpoint changes):

  intent          0.97   clinc150, zero-shot labels (unseen at training)
  banking77       0.50   entirely separate label domain, held out of
                         training in full (data.py Sec 5.3) -- this is the
                         checkpoint's own tracked banking77_holdout_acc
  oos             --     in-scope vs. out-of-scope request detection
  paraphrase      0.82   QQP pair -> yes/no
  mcq_sciq        0.90
  mcq_race        0.70
  mcq_hellaswag   0.55
  mcq_commonsense 0.55
  mcq_openbookqa  0.47
  mcq_arc         ~0.40  arc_easy + arc_challenge -- the measured weak spot;
                         kept in (not hidden) since a diagnostic dashboard
                         showing only wins isn't showing whether it works.

Modality dropout (data.py's sample_modality) is pinned to full text+vector
here -- that augmentation exists to make training robust to missing
modalities, not something a demo should show off; a live quiz should
always run the checkpoint's best-case path.
"""
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXP7_DIR = os.path.join(ROOT, "experiments", "exp7_hybrid_decision")
SRC_DIR = os.path.join(ROOT, "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))
sys.path.insert(0, os.path.abspath(EXP7_DIR))  # same ordering rationale as
# open_one_agent.py: exp7's own model.py must win over src/model.py.

import model as _exp7_model  # noqa: E402,F401 -- see open_one_agent.py's comment;
# must be imported (and cached under the bare name "model") before data.py.

from data import (  # noqa: E402
    build_intent_example, build_mcq_example, build_qqp_example,
    load_intent_corpus_minus_banking77, raw_intent_name, sample_instructions,
)
from local_data import load_mcq_corpus, load_qqp_pairs  # noqa: E402
from model import PackedExample  # noqa: E402

# Which MCQ corpus source feeds which quiz kind. arc_easy/arc_challenge are
# merged into one "mcq_arc" bucket -- both ARC, both weak, splitting them
# further wouldn't teach anything the merged number doesn't already show.
MCQ_KIND_SOURCES = {
    "mcq_sciq": {"sciq"}, "mcq_race": {"race"}, "mcq_hellaswag": {"hellaswag"},
    "mcq_commonsense": {"commonsense_qa"}, "mcq_openbookqa": {"openbookqa"},
    "mcq_arc": {"arc_easy", "arc_challenge"},
}

QUIZ_TYPES = ["intent", "banking77", "oos", "paraphrase", *MCQ_KIND_SOURCES.keys()]

# Demo-sized option counts -- data.py's real training range (2-128, 5% at
# 255) is right for gradient diversity, not for a screen a person reads.
INTENT_N_OPTIONS = 5
BANKING77_N_OPTIONS = 5


def _pin_full_modality(ex):
    """Undoes data.py's random modality dropout for one built example --
    see module docstring."""
    ex.use_text = True
    ex.use_vector = True
    return ex


class QuizBank:
    """Loads each corpus once, holds it in memory, and hands out one fresh
    PackedExample + display metadata per call. No model code -- mirrors the
    snake_env.py split (rules/data vs. inference)."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

        intent = load_intent_corpus_minus_banking77()
        self._intent_pool = intent.zero_shot_labels
        self._intent_rows = [(t, l) for t, l in intent.test_zero_shot if l in self._intent_pool]

        self._banking_pool = intent.banking77_labels
        self._banking_rows = intent.banking77_holdout

        # oos: 50/50 mix of genuinely out-of-scope requests (test_oos, all
        # labeled clinc::oos) and genuinely in-scope ones (test_zero_shot,
        # reusing the same rows as the intent task so this isn't a second
        # copy of the same data under a different name -- it's a different
        # QUESTION about the same rows: "is this answerable at all?" vs.
        # "which category?").
        self._oos_rows = [t for t, _ in intent.test_oos]
        self._inscope_rows = [t for t, _ in self._intent_rows]

        mcq = load_mcq_corpus()
        self._mcq_rows = {
            kind: [r for r in mcq["test"] if r.get("source") in sources]
            for kind, sources in MCQ_KIND_SOURCES.items()
        }

        _, qqp_val = load_qqp_pairs()
        self._qqp_rows = qqp_val

    def sample(self, qtype):
        if qtype == "intent":
            return self._sample_intent()
        if qtype == "banking77":
            return self._sample_banking77()
        if qtype == "oos":
            return self._sample_oos()
        if qtype == "paraphrase":
            return self._sample_paraphrase()
        if qtype in MCQ_KIND_SOURCES:
            return self._sample_mcq(qtype)
        raise ValueError(qtype)

    def _sample_intent(self):
        text, true_label = self.rng.choice(self._intent_rows)
        ex = build_intent_example(text, true_label, self._intent_pool, self.rng,
                                   n_target=INTENT_N_OPTIONS)
        _pin_full_modality(ex)
        return ex, {
            "kind": "intent", "source": "clinc150 (zero-shot labels)",
            "prompt": text, "gold_text": raw_intent_name(true_label),
        }

    def _sample_banking77(self):
        text, true_label = self.rng.choice(self._banking_rows)
        ex = build_intent_example(text, true_label, self._banking_pool, self.rng,
                                   n_target=BANKING77_N_OPTIONS)
        _pin_full_modality(ex)
        return ex, {
            "kind": "banking77", "source": "banking77 (entire domain held out of training)",
            "prompt": text, "gold_text": raw_intent_name(true_label),
        }

    def _sample_oos(self):
        is_oos = self.rng.random() < 0.5
        text = self.rng.choice(self._oos_rows if is_oos else self._inscope_rows)
        ex = PackedExample(
            context=text,
            instructions="Is this a real request that fits a known support category, or is it out of scope?",
            option_texts=["out of scope", "fits a known category"],
            qtype="bool", answer_idx=int(not is_oos),
        )
        _pin_full_modality(ex)
        return ex, {
            "kind": "oos", "source": "clinc150 (in-scope vs. out-of-scope)",
            "prompt": text, "gold_text": "out of scope" if is_oos else "fits a known category",
        }

    def _sample_mcq(self, kind):
        row = self.rng.choice(self._mcq_rows[kind])
        ex = build_mcq_example(row, foreign_pool_texts=[], rng=self.rng)
        _pin_full_modality(ex)
        return ex, {
            "kind": kind, "source": row.get("source", kind),
            "prompt": row["context"], "question": row["question"],
            "gold_text": ex.option_texts[ex.answer_idx],
        }

    def _sample_paraphrase(self):
        text1, text2, is_para = self.rng.choice(self._qqp_rows)
        ex = build_qqp_example(text1, text2, is_para, self.rng)
        _pin_full_modality(ex)
        return ex, {
            "kind": "paraphrase", "source": "qqp",
            "prompt": text1, "question": ex.instructions,
            "gold_text": "yes" if is_para else "no",
        }

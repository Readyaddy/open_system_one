"""Builds the new corpora experiment 7 needs that don't exist yet (see
NOTES.md Sec 5.2):

  data/exp7_bool_corpus/       {context, question, answer_idx(0/1), source}
                                from MNLI (entailment -> bool) + BoolQ (native
                                yes/no QA).
  data/exp7_score_corpus/      {context, question, num_levels, answer_idx, source}
                                from Yelp Review Full + SST-5 -- both genuinely
                                5-level ORDINAL labels, which is what the RPS
                                loss term (NOTES.md Sec 3.1) needs to mean
                                anything.
  data/exp7_diversity_corpus/  {text, label, labels (this example's full
                                label set), source} from DBpedia-14, TREC,
                                Yahoo Answers Topics, GoEmotions -- each a
                                DIFFERENT label vocabulary, which is the
                                single biggest lever for zero-shot
                                generalization per NOTES.md Sec 5.2 (exp6 had
                                seen exactly one vocabulary, 255 intents, in
                                its entire life).
  data/exp7_typed_decisions/   eval-only, from LocalLLaMA/typed-decisions
                                (Apache-2.0) -- see LAYA_COMPARISON_REPORT.md
                                Sec 6. Never mixed into training.

Deliberately EXCLUDED from exp7_diversity_corpus: AG News and dair-ai/emotion
(both are the Laya/Jev comparison benchmarks -- eval.py downloads them
directly for that; including them here would make "zero-shot" a lie) and
Banking77 (already excluded from intent_corpus at load time, see data.py's
load_intent_corpus_minus_banking77 -- NOT touched by this script since it's
non-destructive filtering, not a data-build step).

This is a best-effort builder, not dataset_v7.py's full rigor (no
cross-source dedup pass, no leakage check across these NEW corpora) --
stated plainly rather than silently assumed. Each source gets a fixed-seed
deterministic split when the upstream dataset doesn't already have one.

Usage: python scripts/build_exp7_data.py
"""
import json
import os
import random

from datasets import load_dataset

random.seed(20260921)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows -> {path}")


def _split(rows, train_frac=0.8, val_frac=0.1, seed=1234):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:]


# --------------------------------------------------------------------------
# bool_corpus
# --------------------------------------------------------------------------

def build_bool_corpus(max_per_source=15000):
    print("=== exp7_bool_corpus ===")
    rows = {"train": [], "val": [], "test": []}

    print("  loading MNLI (entailment -> bool)...")
    mnli = load_dataset("nyu-mll/multi_nli")
    for split_name, hf_split in (("train", "train"), ("val", "validation_matched")):
        ds = mnli[hf_split]
        idx = random.sample(range(len(ds)), min(max_per_source, len(ds)))
        for i in idx:
            ex = ds[i]
            if ex["label"] not in (0, 2):  # keep entailment (0) / contradiction (2), drop neutral (1)
                continue
            rows[split_name].append({
                "context": ex["premise"],
                "question": f"Does this entail: \"{ex['hypothesis']}\"?",
                "answer_idx": int(ex["label"] == 0),
                "source": "mnli",
            })

    print("  loading BoolQ (native yes/no QA)...")
    boolq = load_dataset("google/boolq")
    for split_name, hf_split in (("train", "train"), ("val", "validation")):
        ds = boolq[hf_split]
        idx = random.sample(range(len(ds)), min(max_per_source, len(ds)))
        for i in idx:
            ex = ds[i]
            rows[split_name].append({
                "context": ex["passage"], "question": ex["question"],
                "answer_idx": int(bool(ex["answer"])), "source": "boolq",
            })

    # BoolQ has no test split of its own -- carve one out of val's tail
    # (deterministic, fixed seed) rather than leaving `test` empty.
    _, val_kept, val_carved_test = _split(rows["val"], train_frac=0.0, val_frac=0.7, seed=55)
    rows["val"], rows["test"] = val_kept, val_carved_test

    for name, split_rows in rows.items():
        random.Random(9).shuffle(split_rows)
        _write_jsonl(split_rows, os.path.join(DATA_DIR, "exp7_bool_corpus", f"{name}.jsonl"))


# --------------------------------------------------------------------------
# score_corpus (ordinal)
# --------------------------------------------------------------------------

def build_score_corpus(max_per_source=15000):
    print("=== exp7_score_corpus ===")
    rows = {"train": [], "val": [], "test": []}

    print("  loading Yelp Review Full (5-star, genuinely ordinal)...")
    yelp = load_dataset("Yelp/yelp_review_full")
    yelp_train_all, yelp_val_all, _ = _split(yelp["train"].select(
        random.sample(range(len(yelp["train"])), min(max_per_source * 2, len(yelp["train"])))
    ), seed=77)
    for split_name, ds in (("train", yelp_train_all), ("val", yelp_val_all),
                            ("test", yelp["test"].select(range(min(max_per_source, len(yelp["test"]))))
                             )):
        for ex in ds:
            text = ex["text"]
            if len(text) > 2000:
                text = text[:2000]
            rows[split_name].append({
                "context": text, "question": "How many stars would this review receive?",
                "num_levels": 5, "answer_idx": int(ex["label"]), "source": "yelp_stars",
            })

    print("  loading SST-5 (5-way ordinal sentiment)...")
    sst5 = load_dataset("SetFit/sst5")
    for split_name, hf_split in (("train", "train"), ("val", "validation"), ("test", "test")):
        if hf_split not in sst5:
            continue
        ds = sst5[hf_split]
        for ex in ds:
            rows[split_name].append({
                "context": ex["text"], "question": "How positive is the sentiment, on a 5-point scale?",
                "num_levels": 5, "answer_idx": int(ex["label"]), "source": "sst5",
            })

    for name, split_rows in rows.items():
        random.Random(9).shuffle(split_rows)
        _write_jsonl(split_rows, os.path.join(DATA_DIR, "exp7_score_corpus", f"{name}.jsonl"))


# --------------------------------------------------------------------------
# diversity_corpus (label-vocabulary diversity -- see NOTES.md Sec 5.2)
# --------------------------------------------------------------------------

def build_diversity_corpus(max_per_source=15000):
    print("=== exp7_diversity_corpus ===")
    print("  (AG News, dair-ai/emotion, and Banking77 are DELIBERATELY excluded -- "
          "they're held-out comparison benchmarks, see module docstring)")
    rows = {"train": [], "val": [], "test": []}

    def add_source(name, hf_dataset_name, text_field, label_field, splits_map, label_names=None,
                   config=None, text_label_field=None):
        """text_label_field: for sources whose label is already a plain
        string column (SetFit/TREC-QC's `label_coarse_text`,
        SetFit/20_newsgroups' `label_text`) instead of an int + ClassLabel
        `.names` lookup -- pass that field name directly instead of
        label_field/label_names."""
        print(f"  loading {name}...")
        ds = load_dataset(hf_dataset_name, config) if config else load_dataset(hf_dataset_name)
        train_split = splits_map["train"]
        if text_label_field:
            names = sorted(set(ds[train_split].unique(text_label_field)))
        else:
            names = label_names or ds[train_split].features[label_field].names
        for out_split, hf_split in splits_map.items():
            if hf_split not in ds:
                continue
            d = ds[hf_split]
            idx = random.sample(range(len(d)), min(max_per_source, len(d)))
            for i in idx:
                ex = d[i]
                label_text = ex[text_label_field] if text_label_field else names[ex[label_field]]
                rows[out_split].append({
                    "text": ex[text_field], "label": label_text,
                    "labels": list(names), "source": name,
                })

    add_source("dbpedia14", "fancyzhx/dbpedia_14", "content", "label",
               {"train": "train", "test": "test"})
    # CogComp/trec ships a legacy loading SCRIPT that current `datasets`
    # versions refuse to run at all ("Dataset scripts are no longer
    # supported") -- confirmed live, not a schema guess. SetFit/TREC-QC is
    # a parquet-format repackaging of the same underlying TREC data, with
    # the coarse label already given as a string (`label_coarse_text`).
    add_source("trec", "SetFit/TREC-QC", "text", None, {"train": "train", "test": "test"},
               text_label_field="label_coarse_text")
    add_source("yahoo_answers", "community-datasets/yahoo_answers_topics", "question_title", "topic",
               {"train": "train", "test": "test"})
    # Bonus source, added once TREC's replacement was already being wired
    # up the same way -- another wholly distinct label vocabulary (20
    # newsgroup topics), which is directly the highest-priority lever this
    # corpus exists for (NOTES.md Sec 5.2).
    add_source("20_newsgroups", "SetFit/20_newsgroups", "text", None, {"train": "train", "test": "test"},
               text_label_field="label_text")

    print("  loading GoEmotions (single-label subset only -- it's natively multi-label;"
          " examples with more than one annotated emotion are dropped, a real simplification)...")
    goemo = load_dataset("google-research-datasets/go_emotions", "simplified")
    emo_names = goemo["train"].features["labels"].feature.names
    for out_split, hf_split in {"train": "train", "val": "validation", "test": "test"}.items():
        if hf_split not in goemo:
            continue
        d = goemo[hf_split]
        idx = random.sample(range(len(d)), min(max_per_source, len(d)))
        for i in idx:
            ex = d[i]
            if len(ex["labels"]) != 1:
                continue
            rows[out_split].append({
                "text": ex["text"], "label": emo_names[ex["labels"][0]],
                "labels": list(emo_names), "source": "goemotions",
            })

    # dbpedia14/trec/yahoo have no val split -- carve one from train's tail.
    train_only, val_carved, _ = _split(rows["train"], train_frac=0.9, val_frac=0.1, seed=321)
    rows["train"], existing_val = train_only, rows["val"]
    rows["val"] = existing_val + val_carved

    for name, split_rows in rows.items():
        random.Random(9).shuffle(split_rows)
        _write_jsonl(split_rows, os.path.join(DATA_DIR, "exp7_diversity_corpus", f"{name}.jsonl"))


# --------------------------------------------------------------------------
# typed_decisions (eval-only, Apache-2.0 -- see LAYA_COMPARISON_REPORT.md Sec 6)
# --------------------------------------------------------------------------

def build_typed_decisions():
    print("=== exp7_typed_decisions (eval-only) ===")
    print("  loading LocalLLaMA/typed-decisions...")
    try:
        # Confirmed live: this dataset requires an explicit config name
        # (['agent_trace_observability', 'customer_service',
        # 'invoice_processing', 'security_incidents', 'all']) -- 'all'
        # pools every workflow, matching LAYA_COMPARISON_REPORT.md's
        # description of it as ~2,000 decisions across four business
        # workflows.
        ds = load_dataset("LocalLLaMA/typed-decisions", "all")
    except Exception as e:
        print(f"  FAILED to load LocalLLaMA/typed-decisions ({e}) -- skipping. "
              f"This eval set is optional; training/eval both degrade gracefully without it.")
        return

    print(f"  {sum(len(s) for s in ds.values())} rows across splits {list(ds.keys())}. "
          f"Confirmed live schema: `state`/`questions`/`gold` are each a JSON-ENCODED STRING "
          f"(not a nested struct) -- `questions` keyed by question name, each value "
          f"{{criteria: {{option_key: option_description}}, instructions, type}}; `gold` keyed "
          f"the same way, each value {{label: <the correct option_key>, ...}}. Laya's own three "
          f"question types (LAYA_COMPARISON_REPORT.md Sec 3) appear verbatim here: 'choice', "
          f"'score', and 'noul' -- mapped to this project's 'bool' below.")

    QTYPE_MAP = {"choice": "choice", "score": "score", "noul": "bool"}

    rows = []
    n_skipped_rows, n_skipped_questions = 0, 0
    for split_name, d in ds.items():
        for ex in d:
            try:
                state = json.loads(ex["state"])
                questions = json.loads(ex["questions"])
                gold = json.loads(ex["gold"])
            except (json.JSONDecodeError, KeyError, TypeError):
                n_skipped_rows += 1
                continue
            state_text = json.dumps(state)  # structured data rendered as text context
            for q_name, q_spec in questions.items():
                criteria = q_spec.get("criteria")
                qtype_raw = q_spec.get("type")
                if not criteria or qtype_raw not in QTYPE_MAP:
                    n_skipped_questions += 1
                    continue
                # `criteria` is a {option_key: description} dict for choice/
                # noul questions, but a plain ORDERED LIST of level
                # DESCRIPTIONS for score (ordinal) questions -- and for
                # score questions `gold.label` is the STRINGIFIED INDEX
                # ("0", "1", ...) into that list, not the description text
                # itself (confirmed live by inspecting a real row -- e.g.
                # gold={"label": "1", "probabilities": {"0":.., "1":..}}
                # against criteria=["No time pressure...", "Routine...", ...]).
                if isinstance(criteria, dict):
                    option_keys = list(criteria.keys())       # dict order = authored ordinal order
                    option_texts = [criteria[k] for k in option_keys]
                elif isinstance(criteria, list):
                    option_keys = [str(i) for i in range(len(criteria))]
                    option_texts = list(criteria)
                else:
                    n_skipped_questions += 1
                    continue
                gold_entry = gold.get(q_name)
                gold_label = gold_entry.get("label") if isinstance(gold_entry, dict) else None
                if gold_label not in option_keys:
                    n_skipped_questions += 1
                    continue
                rows.append({
                    "context": state_text, "question": q_spec.get("instructions", ""),
                    "options": option_texts, "answer_idx": option_keys.index(gold_label),
                    "qtype": QTYPE_MAP[qtype_raw], "workflow": ex.get("workflow", ""),
                })
    print(f"  normalized {len(rows)} question rows ({n_skipped_rows} source rows failed to parse, "
          f"{n_skipped_questions} individual questions skipped -- missing criteria/gold/unknown type).")
    if rows:
        print(f"  sample row: {json.dumps(rows[0])[:400]}")
    _write_jsonl(rows, os.path.join(DATA_DIR, "exp7_typed_decisions", "test.jsonl"))


if __name__ == "__main__":
    build_bool_corpus()
    build_score_corpus()
    build_diversity_corpus()
    build_typed_decisions()
    print("\nDone. See data/exp7_*/ for output.")

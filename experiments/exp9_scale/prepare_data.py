"""Rebuilds every exp8/exp9 corpus from its public sources, in the exact order
it was built and audited locally (2026-09-24/25). Meant for a cheap Colab CPU
session writing to Drive; every stage leaves a marker so a disconnect resumes
where it stopped instead of starting over.

  stage            produces                                         script
  intent_qqp       data/intent_corpus, data/qqp_paraphrase_pairs    scripts/build_data_v7.py
  mcq              data/mcq_corpus                                  scripts/build_mcq_data.py
  exp7             data/exp7_{bool,score,diversity}_corpus,         scripts/build_exp7_data.py
                   data/exp7_typed_decisions, exp7_manifest.json
  filter           AFLite on mcq -> data/exp7_mcq_corpus;           scripts/filter_exp7_shortcuts.py
                   bool/diversity filtered + val `blind_easy` tags
  bias_weights     bool train = unfiltered + bias_w                 scripts/weight_exp7_bias.py
  jevbench         data/jevbench (official public tiers)            git clone
  audit            data/exp7_data_audit.json                        scripts/audit_exp7_data.py

Usage: python prepare_data.py [--root /content/JEPA] [--only stage,stage] [--force]
"""
import argparse
import json
import os
import subprocess
import sys
import time

STAGES = ["intent_qqp", "mcq", "exp7", "filter", "bias_weights", "jevbench", "audit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--only", default=",".join(STAGES))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    data = os.path.join(a.root, "data")
    os.makedirs(os.path.join(data, "_stages"), exist_ok=True)
    py = sys.executable
    S = lambda *p: os.path.join(a.root, "scripts", *p)
    cmds = {
        "intent_qqp": [py, S("build_data_v7.py")],
        "mcq": [py, S("build_mcq_data.py")],
        "exp7": [py, S("build_exp7_data.py")],
        "filter": [py, S("filter_exp7_shortcuts.py"), "--only", "mcq,bool,diversity"],
        "bias_weights": [py, S("weight_exp7_bias.py"), "--corpus", "bool", "--from_unfiltered"],
        "jevbench": ["git", "clone", "--depth", "1", "https://github.com/fstandhartinger/jevbench.git",
                     os.path.join(data, "jevbench")],
        "audit": [py, S("audit_exp7_data.py")],
    }
    log = open(os.path.join(data, "_stages", "prepare.log"), "a", encoding="utf-8")

    def say(m):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    for st in [s for s in STAGES if s in a.only.split(",")]:
        marker = os.path.join(data, "_stages", f"{st}.done")
        if os.path.exists(marker) and not a.force:
            say(f"[{st}] already done -- skipping")
            continue
        if st == "jevbench" and os.path.exists(os.path.join(data, "jevbench", "datasets", "public")):
            open(marker, "w").write("present\n")
            say(f"[{st}] already present")
            continue
        say(f"[{st}] start: {' '.join(cmds[st])}")
        t = time.time()
        with open(os.path.join(data, "_stages", f"{st}.log"), "w", encoding="utf-8") as out:
            r = subprocess.run(cmds[st], cwd=a.root, stdout=out, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode != 0:
            say(f"[{st}] FAILED (exit {r.returncode}) after {time.time()-t:.0f}s -- see data/_stages/{st}.log")
            sys.exit(1)
        open(marker, "w").write(f"{time.time()-t:.0f}s\n")
        say(f"[{st}] done in {(time.time()-t)/60:.1f} min")

    # summary of what training will read
    summary = {}
    for d in ("intent_corpus", "exp7_mcq_corpus", "exp7_bool_corpus", "exp7_score_corpus", "exp7_diversity_corpus"):
        p = os.path.join(data, d, "train.jsonl")
        summary[d] = sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else None
    say("train rows: " + json.dumps(summary))
    open(os.path.join(data, "_stages", "ALL_DONE"), "w").write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()

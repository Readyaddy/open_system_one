#!/bin/bash
# Runs eval.py --mode external against a checkpoint -- AG News / DAIR
# Emotion (genuine zero-shot) + Banking77 holdout (true zero-shot here,
# unlike exp6's in-distribution number) -- and prints the direct comparison
# against Laya/Jev's published numbers (baked into eval.py's own print
# statements, see run_external_benchmarks).
#
# Uses a CPU session by default -- eval is forward-only (no gradients, no
# optimizer), so it doesn't need a GPU's speed for a one-off check, and
# CPU sessions are cheaper. Needs its OWN Drive mount (the just-stopped
# jepa-exp7-train session's mount doesn't carry over to a new session) --
# `colab drivemount`'s OAuth approval needs a real browser click, which
# can't be done headlessly (see COLAB_SKILL.md / this project's own
# SESSION_LOG.md) -- run this yourself in an interactive terminal, approve
# the prompt when it appears, and the rest is unattended.
#
# Usage:
#   wsl -d kali-linux
#   bash /mnt/d/projects/JEPA/scripts/colab_exp7_eval_external.sh [checkpoint_name]
#
# checkpoint_name defaults to exp7_best_zeroshot.pt (epoch 10, the current
# best-zero-shot checkpoint from the run stopped 2026-09-21). Pass
# exp7_latest.pt or exp7_best_val.pt to eval a different one instead.

set -e
source ~/colab-cli-env/bin/activate

SESSION=jepa-exp7-eval
LOCAL_ROOT=/mnt/d/projects/JEPA
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_NAME="${1:-exp7_best_zeroshot.pt}"

echo "=== 1. Creating Colab session ($SESSION, CPU -- eval is forward-only, no GPU needed) ==="
colab --auth=adc new -s "$SESSION"

echo "=== 2. Mounting Google Drive (approve the browser prompt when it appears) ==="
colab --auth=adc drivemount -s "$SESSION"

echo "=== 3. Creating remote directories ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 60
import os
for d in ['/content/experiments/exp7_hybrid_decision', '/content/src',
          '/content/data/intent_corpus']:
    os.makedirs(d, exist_ok=True)
print('dirs ready')
PYEOF

echo "=== 4. Uploading code ==="
for f in model.py data.py train.py eval.py; do
  colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/experiments/exp7_hybrid_decision/$f" "/content/experiments/exp7_hybrid_decision/$f"
done
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/local_data.py" /content/src/local_data.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/dataset_v7.py" /content/src/dataset_v7.py

echo "=== 5. Uploading intent_corpus data (needed for Banking77 holdout + zero-shot pool) ==="
for f in labels.json test.jsonl test_oos.jsonl test_zero_shot.jsonl train.jsonl val.jsonl; do
  colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/data/intent_corpus/$f" "/content/data/intent_corpus/$f"
done

echo "=== 6. Reusing cached HF backbone + installing packages ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 120
import shutil, subprocess, os
cache_backup = '/content/drive/MyDrive/jepa_checkpoints/hf_cache'
if os.path.exists(cache_backup):
    shutil.copytree(cache_backup, os.path.expanduser('~/.cache/huggingface'), dirs_exist_ok=True)
    print('reused cached HF backbone from Drive')
r = subprocess.run(['pip', 'install', '-q', '-U', 'transformers', 'datasets'], capture_output=True, text=True)
print('deps installed, returncode', r.returncode)
PYEOF

echo "=== 7. Verifying the checkpoint exists on Drive ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 30
import os
p = '/content/drive/MyDrive/jepa_checkpoints/exp7/$CKPT_NAME'
if not os.path.exists(p):
    print(f'ERROR: {p} not found.')
    raise SystemExit(1)
print(f'Found: {p}  ({os.path.getsize(p)/1e6:.1f} MB)')
PYEOF

echo "=== 8. Running eval.py --mode external (AG News, DAIR Emotion, Banking77 holdout) ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 600
import subprocess
r = subprocess.run(
    ['python3', '/content/experiments/exp7_hybrid_decision/eval.py',
     '--ckpt', '/content/drive/MyDrive/jepa_checkpoints/exp7/$CKPT_NAME',
     '--mode', 'external'],
    cwd='/content/experiments/exp7_hybrid_decision', capture_output=True, text=True)
print(r.stdout[-6000:])
print('--- stderr tail ---')
print(r.stderr[-2000:])
print('returncode', r.returncode)
PYEOF

echo ""
echo "=== Done. Stop the session when finished to avoid burning compute units: ==="
echo "===   colab --auth=adc stop -s $SESSION ==="

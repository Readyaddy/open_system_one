#!/bin/bash
# CPU-only Colab session: uploads exp7 code, downloads the REAL
# ModernBERT-large backbone + tokenizer (no GPU needed for this -- it's
# just a weight download + a correctness/memory check), runs
# smoke_test.py --full on CPU, and caches the downloaded model to Drive so
# the later A100 session (colab_exp7_full_setup_and_launch.sh) doesn't pay
# for that download again. This is the step NOTES.md's plan calls for
# BEFORE ever paying for GPU time: verify the real weights load and the
# mechanism runs end to end.
#
# Usage:
#   wsl -d kali-linux
#   bash /mnt/d/projects/JEPA/scripts/colab_exp7_cpu_probe.sh

set -e
source ~/colab-cli-env/bin/activate

SESSION=jepa-exp7-cpu
LOCAL_ROOT=/mnt/d/projects/JEPA

echo "=== 1. Creating Colab session ($SESSION, CPU-only -- no GPU needed for this probe) ==="
colab --auth=adc new -s "$SESSION"

echo "=== 2. Mounting Google Drive (approve the browser prompt when it appears) ==="
colab --auth=adc drivemount -s "$SESSION"

echo "=== 3. Creating remote directories ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 60
import os
for d in ['/content/experiments/exp7_hybrid_decision', '/content/src']:
    os.makedirs(d, exist_ok=True)
print('dirs ready')
PYEOF

echo "=== 4. Uploading exp7 code ==="
for f in model.py data.py train.py smoke_test.py requirements.txt; do
  colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/experiments/exp7_hybrid_decision/$f" "/content/experiments/exp7_hybrid_decision/$f"
done
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/local_data.py" /content/src/local_data.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/dataset_v7.py" /content/src/dataset_v7.py

echo "=== 5. Installing dependencies (transformers>=4.48 for ModernBERT) ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 120
import subprocess
r = subprocess.run(['pip', 'install', '-q', '-U', 'transformers', 'datasets'], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
print('installed, returncode', r.returncode)
PYEOF

echo "=== 6. Downloading the real backbone + running smoke_test.py --full on CPU ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 900
import subprocess, os
env = os.environ.copy()
r = subprocess.run(
    ['python3', '/content/experiments/exp7_hybrid_decision/smoke_test.py', '--full',
     '--backbone', 'answerdotai/ModernBERT-large', '--batch_size', '2', '--n_options', '20',
     '--budget_total', '1024'],
    capture_output=True, text=True, env=env, cwd='/content/experiments/exp7_hybrid_decision')
print(r.stdout[-6000:])
print('--- stderr ---')
print(r.stderr[-3000:])
print('returncode', r.returncode)
PYEOF

echo "=== 7. Caching the downloaded backbone to Drive (so the A100 GPU session reuses it) ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 120
import shutil, os
src = os.path.expanduser('~/.cache/huggingface')
dst = '/content/drive/MyDrive/jepa_checkpoints/hf_cache'
os.makedirs(dst, exist_ok=True)
shutil.copytree(src, dst, dirs_exist_ok=True)
print('cached HF weights to', dst)
PYEOF

echo ""
echo "=== Done. Review the smoke_test output above BEFORE launching the A100 GPU run. ==="
echo "=== Terminate this CPU session yourself once satisfied (website: Runtime > Manage sessions). ==="

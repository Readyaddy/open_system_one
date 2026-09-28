#!/bin/bash
# exp9 step 1 of 2 -- CHEAP CPU session: rebuild all training data from public
# sources and download the backbone weights, both straight into Google Drive.
# No GPU is rented. Run this first; the GPU script refuses to start until it
# finds data/_stages/ALL_DONE and the cached weights on Drive.
#
#   wsl -d kali-linux
#   bash /mnt/d/projects/JEPA/scripts/colab_exp9_prep_cpu.sh
#
# Drive layout written (MyDrive/openjev/):
#   data/          every corpus + _stages/ markers and logs
#   hf/            Qwen3-Embedding-4B (+0.6B) snapshot
# The data build is resumable: re-running skips finished stages. When it
# finishes, the VM releases itself (runtime.unassign), so it can't idle.

set -e
source ~/colab-cli-env/bin/activate
SESSION=${SESSION:-openjev-exp9-prep}
LOCAL_ROOT=/mnt/d/projects/JEPA
C="colab --auth=adc"

# RESUME=1: session already exists and Drive is already mounted -- skip steps 1-2
# (re-running `colab new -s` would allocate a NEW VM rather than reuse this one).
if [ -z "$RESUME" ]; then
  echo "=== 1. CPU session ($SESSION) ==="
  $C new -s "$SESSION"
  echo "=== 2. Mount Drive (approve the browser prompt) ==="
  $C drivemount -s "$SESSION"
fi

echo "=== 3. Upload code (one tarball) ==="
TAR=/tmp/openjev_code.tgz
# GNU tar: --exclude must come BEFORE the paths it applies to
tar -czf "$TAR" --exclude='__pycache__' --exclude='*.pt' --exclude='wandb' -C "$LOCAL_ROOT" \
    src scripts experiments/exp7_hybrid_decision/model.py \
    experiments/exp8_s1_rlcd experiments/exp9_scale experiments/jevbench_protocol.py
$C upload -s "$SESSION" "$TAR" /content/openjev_code.tgz

echo "=== 4. Unpack, link data dir to Drive, install deps, start detached prep ==="
cat << 'PYEOF' | $C exec -s "$SESSION" --timeout 600
import os, subprocess
root = '/content/JEPA'
drive = '/content/drive/MyDrive/openjev'
os.makedirs(root, exist_ok=True); os.makedirs(f'{drive}/data', exist_ok=True); os.makedirs(f'{drive}/hf', exist_ok=True)
subprocess.run(['tar', '-xzf', '/content/openjev_code.tgz', '-C', root], check=True)
if not os.path.islink(f'{root}/data'):
    os.symlink(f'{drive}/data', f'{root}/data')
r = subprocess.run(['pip', 'install', '-q', '-U', 'datasets', 'scikit-learn', 'huggingface_hub'], capture_output=True, text=True)
print('pip rc', r.returncode, r.stderr[-500:])
job = r'''
import subprocess, sys, os
from huggingface_hub import snapshot_download
drive = "/content/drive/MyDrive/openjev"
for repo in ["Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-0.6B"]:
    d = f"{drive}/hf/{repo.split('/')[1]}"
    if not os.path.exists(d + "/config.json"):
        snapshot_download(repo, local_dir=d)
    print("weights ready:", d, flush=True)
rc = subprocess.run([sys.executable, "/content/JEPA/experiments/exp9_scale/prepare_data.py",
                     "--root", "/content/JEPA"]).returncode
print("prepare_data rc", rc, flush=True)
if rc == 0:
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as e:
        print("could not self-release:", e)
'''
open('/content/prep_job.py', 'w').write(job)
log = open(f'{drive}/prep_job.log', 'a')
p = subprocess.Popen(['python3', '/content/prep_job.py'], stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True, cwd=root)
open('/content/prep_job.pid', 'w').write(str(p.pid))
print('prep job started, pid', p.pid)
PYEOF

echo ""
echo "=== Prep is running on the VM (safe to Ctrl+C this monitor; the job keeps going). ==="
echo "=== It releases the VM itself when done. Progress: MyDrive/openjev/data/_stages/prepare.log ==="
set +e
while true; do
  OUT=$(cat << 'PYEOF' | $C exec -s "$SESSION" --timeout 30 2>&1
import os, subprocess
d = '/content/drive/MyDrive/openjev'
for f in (f'{d}/prep_job.log', f'{d}/data/_stages/prepare.log'):
    if os.path.exists(f):
        print('\n'.join(open(f).read().splitlines()[-6:]))
print('ALL_DONE' if os.path.exists(f'{d}/data/_stages/ALL_DONE') else 'running...')
PYEOF
)
  echo "=== $(date +%H:%M:%S) ==="; echo "$OUT"
  if echo "$OUT" | grep -q "ALL_DONE"; then echo "=== Data + weights ready on Drive. ==="; break; fi
  if echo "$OUT" | grep -qi "session .* not found\|no such session"; then
    echo "Session gone (released itself or disconnected). Check MyDrive/openjev/data/_stages/prepare.log"; break
  fi
  sleep 120
done

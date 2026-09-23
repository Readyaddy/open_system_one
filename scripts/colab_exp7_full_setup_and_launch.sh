#!/bin/bash
# Complete, from-scratch setup + launch + LIVE MONITORING for experiment 7,
# meant to be run ONCE from a single terminal you keep open -- same
# convention as exp6's colab_full_setup_and_launch.sh. Run this only after
# terminating any old sessions yourself (website: Runtime > Manage
# sessions), so this creates exactly one clean A100 session.
#
# IMPORTANT -- batch_size below is a DELIBERATELY CONSERVATIVE STARTING
# POINT, not a tuned value. exp7's packed sequences are up to ~2048 tokens
# (vs. exp6's 32-384 token sequences), so memory-per-example is much
# higher and the right batch size has to be recalibrated from scratch --
# exp6 itself staged its batch size up in real stages (48 -> 160 -> 384)
# by watching actual GPU memory via colab_exp7_watch.sh, not by guessing
# up front. Target: ~35GB/40GB used on the A100 (5GB headroom, per the
# project's own instruction), raised in stages once you've watched a few
# real steps -- see colab_exp7_relaunch_only.sh to push a new batch_size
# without redoing setup.
#
# Usage:
#   wsl -d kali-linux
#   bash /mnt/d/projects/JEPA/scripts/colab_exp7_full_setup_and_launch.sh

set -e
source ~/colab-cli-env/bin/activate

SESSION=jepa-exp7-train
LOCAL_ROOT=/mnt/d/projects/JEPA
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WANDB_API_KEY="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"

# AUTO-RECONNECT (see colab_reconnect.py's docstring): local session
# tracking can get pruned on a single flaky assignments listing even
# though the VM and anything running on it are untouched. Used by the
# monitor loop at the end of this script.
check_and_reconnect() {
  if [ "$2" -eq 0 ] && ! echo "$1" | grep -qi "session .* not found"; then
    return 0
  fi
  echo "!!! '$SESSION' unreachable -- auto-reconnecting... !!!"
  if python3 "$SCRIPT_DIR/colab_reconnect.py" "$SESSION"; then
    echo "!!! Reconnected. Resuming. !!!"
    return 0
  else
    echo "!!! Auto-reconnect failed -- run manually: bash $SCRIPT_DIR/colab_reconnect.sh $SESSION --endpoint <endpoint>"
    return 1
  fi
}

echo "=== 1. Creating Colab session ($SESSION, A100) ==="
colab --auth=adc new -s "$SESSION" --gpu A100

echo "=== 2. Mounting Google Drive (approve the browser prompt when it appears) ==="
colab --auth=adc drivemount -s "$SESSION"

echo "=== 3. Creating remote directories ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 60
import os
for d in ['/content/experiments/exp7_hybrid_decision', '/content/src', '/content/scripts', '/content/logs',
          '/content/data/intent_corpus', '/content/data/qqp_paraphrase_pairs', '/content/data/mcq_corpus',
          '/content/data/exp7_bool_corpus', '/content/data/exp7_score_corpus',
          '/content/data/exp7_diversity_corpus', '/content/data/exp7_typed_decisions']:
    os.makedirs(d, exist_ok=True)
print('dirs ready')
PYEOF

echo "=== 4. Uploading code ==="
for f in model.py data.py train.py eval.py calibrate.py smoke_test.py; do
  colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/experiments/exp7_hybrid_decision/$f" "/content/experiments/exp7_hybrid_decision/$f"
done
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/local_data.py" /content/src/local_data.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/dataset_v7.py" /content/src/dataset_v7.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/scripts/build_exp7_data.py" /content/scripts/build_exp7_data.py

echo "=== 5. Uploading intent_corpus data ==="
for f in labels.json test.jsonl test_oos.jsonl test_zero_shot.jsonl train.jsonl val.jsonl; do
  colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/data/intent_corpus/$f" "/content/data/intent_corpus/$f"
done

echo "=== 6. Reusing the cached HF backbone from the CPU probe (if it ran) + installing packages ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 120
import shutil, subprocess, os
cache_backup = '/content/drive/MyDrive/jepa_checkpoints/hf_cache'
if os.path.exists(cache_backup):
    shutil.copytree(cache_backup, os.path.expanduser('~/.cache/huggingface'), dirs_exist_ok=True)
    print('reused cached HF backbone from Drive -- no re-download needed')
else:
    print('no cached HF backbone found -- will download on first train.py launch (run '
          'colab_exp7_cpu_probe.sh first next time to cache this and avoid paying GPU-time for it)')
r = subprocess.run(['pip', 'install', '-q', '-U', 'transformers', 'bitsandbytes', 'wandb'],
                    capture_output=True, text=True)
print('deps installed, returncode', r.returncode)
PYEOF

echo "=== 7. Copying mcq/qqp/exp7-corpora from Drive backup if present, else building on-VM ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 600
import shutil, subprocess, os
backup_root = '/content/drive/MyDrive/jepa_checkpoints/data_backup'

def restore_or_none(name):
    src = os.path.join(backup_root, name)
    dst = f'/content/data/{name}'
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f'{name}: restored from Drive backup')
        return True
    return False

for name in ['mcq_corpus', 'qqp_paraphrase_pairs']:
    if not restore_or_none(name):
        print(f'{name}: NOT FOUND in Drive backup -- this is unexpected (exp6 already built it); '
              f'training will run with this task INACTIVE. See data/README.md to rebuild.')

need_build = []
for name in ['exp7_bool_corpus', 'exp7_score_corpus', 'exp7_diversity_corpus', 'exp7_typed_decisions']:
    if not restore_or_none(name):
        need_build.append(name)

if need_build:
    print(f'building on-VM (not yet backed up to Drive): {need_build}')
    r = subprocess.run(['python3', '/content/scripts/build_exp7_data.py'], cwd='/content',
                        capture_output=True, text=True)
    print(r.stdout[-4000:])
    print(r.stderr[-2000:])
    for name in need_build:
        src = f'/content/data/{name}'
        if os.path.exists(src):
            dst = os.path.join(backup_root, name)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f'{name}: built on-VM and backed up to Drive for next time')
PYEOF

echo "=== 8. Logging into W&B on the VM ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 45
import subprocess
subprocess.run(['wandb', 'login', '$WANDB_API_KEY'], check=True)
print('wandb logged in')
PYEOF

echo "=== 9. Launching training (STAGE exp7a: k_max=1, MaxSim off -- see NOTES.md Sec 6) ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 60
import subprocess, os
ckpt_dir = '/content/drive/MyDrive/jepa_checkpoints/exp7'
os.makedirs(ckpt_dir, exist_ok=True)
log = open(os.path.join(ckpt_dir, 'train_full.log'), 'w')
env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
env['WANDB_API_KEY'] = '$WANDB_API_KEY'
resume_path = os.path.join(ckpt_dir, 'exp7_latest.pt')
args = ['python3', '/content/experiments/exp7_hybrid_decision/train.py',
     '--k_max', '1',
     '--epochs', '20', '--steps_per_epoch', '500',
     '--batch_size', '8', '--grad_accum', '4',
     '--option_chunk_size', '64', '--patience', '6',
     '--wandb_project', 'jepa-exp7', '--wandb_run_id', 'exp7a',
     '--ckpt_dir', ckpt_dir]
if os.path.exists(resume_path):
    args += ['--resume', resume_path]
    print('resuming from', resume_path)
else:
    print('no existing checkpoint -- starting fresh (exp7a: fixed depth, MaxSim off)')
proc = subprocess.Popen(args, cwd='/content/experiments/exp7_hybrid_decision',
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
with open('/content/logs/train_full.pid', 'w') as f:
    f.write(str(proc.pid))
print('launched, pid', proc.pid)
PYEOF

echo ""
echo "=== Training launched. Switching to live monitor (Ctrl+C to stop watching -- training keeps running). ==="
echo "=== Also viewable anytime at https://wandb.ai (project: jepa-exp7) ==="
echo "=== Watch the first few steps' gpu_mem line and raise --batch_size via colab_exp7_relaunch_only.sh ==="
echo "=== once you know how much headroom you actually have (target ~35GB/40GB). ==="
sleep 5

set +e
while true; do
  clear
  echo "=== $(date) ==="
  OUTPUT=$(cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 30 2>&1
import subprocess
print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                       '--format=csv'], capture_output=True, text=True).stdout)
pid = open('/content/logs/train_full.pid').read().strip()
ps = subprocess.run(['ps', '-p', pid, '-o', 'pid,stat,etimes'], capture_output=True, text=True).stdout
print(ps if len(ps.strip().splitlines()) > 1 else f'PROCESS {pid} NOT FOUND -- may have crashed/disconnected')
lines = open('/content/drive/MyDrive/jepa_checkpoints/exp7/train_full.log').read().splitlines()
step_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' in l][-10:]
summary_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' not in l][-3:]
other_recent = [l for l in lines if not l.strip().startswith('epoch')][-8:]
print('\n'.join(other_recent))
print('--- recent per-step progress ---')
print('\n'.join(step_lines))
print('--- recent epoch summaries ---')
print('\n'.join(summary_lines))
PYEOF
)
  STATUS=$?
  echo "$OUTPUT"
  check_and_reconnect "$OUTPUT" "$STATUS"
  sleep 30
done

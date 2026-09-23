#!/bin/bash
# Use this when the session/Drive/data are already set up and you just need
# to push updated code (or a new --batch_size / --k_max / --use_maxsim
# flag, e.g. moving from exp7a to exp7b/7d per NOTES.md Sec 6) and relaunch
# training -- skips session creation, drivemount, and data upload entirely.
# Mirrors exp6's colab_relaunch_only.sh.
#
# Edit TRAIN_ARGS below before running to change stage/flags, e.g.:
#   exp7a (fixed depth):        --k_max 1
#   exp7b (depth-conditioned):  --k_max 6
#   exp7d (MaxSim on):          --k_max 6 --use_maxsim
# Also this is where you raise --batch_size once colab_exp7_watch.sh has
# shown you real GPU memory headroom (target ~35GB/40GB, see NOTES.md).

set -e
source ~/colab-cli-env/bin/activate

SESSION=jepa-exp7-train
LOCAL_ROOT=/mnt/d/projects/JEPA
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WANDB_API_KEY="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"

# AUTO-RECONNECT (see colab_reconnect.py's docstring): local session
# tracking can get pruned on a single flaky assignments listing even
# though the VM and anything running on it are untouched. Used by the
# monitor loop below.
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

# EDIT THIS to change stage/batch-size between relaunches:
# Bumped again 2026-09-21: found train.py never actually enabled bf16
# autocast anywhere -- the whole 440M model was training in fp32 the entire
# time (fixed in train.py/eval.py/calibrate.py, see AUTOCAST_DTYPE/_ac).
# That's the real explanation for the earlier memory/speed profile, not
# just an under-sized batch. With bf16 active, memory should drop
# substantially, so batch_size 32->64 (grad_accum 2->1 holds the effective
# batch, batch_size*grad_accum, at 64 either way) and option_chunk_size
# 256->384. Watch gpu_mem after this relaunch and adjust again --
# these are still an estimate, not yet an observed-and-confirmed number.
TRAIN_ARGS="--k_max 1 --epochs 20 --steps_per_epoch 500 --batch_size 64 --grad_accum 1 --option_chunk_size 384 --patience 6"
WANDB_RUN_ID="exp7a"

echo "=== Verifying session is reachable ==="
colab --auth=adc status -s "$SESSION"

echo "=== Re-uploading updated code ==="
for f in model.py data.py train.py eval.py calibrate.py; do
  colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/experiments/exp7_hybrid_decision/$f" "/content/experiments/exp7_hybrid_decision/$f"
done

echo "=== Killing any existing training process ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 45
import os, signal
try:
    pid = int(open('/content/logs/train_full.pid').read().strip())
    os.kill(pid, signal.SIGKILL)
    print('killed', pid)
except (FileNotFoundError, ProcessLookupError, ValueError):
    print('nothing to kill')
PYEOF

echo "=== Launching training (auto-resumes from latest checkpoint) ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 60
import subprocess, os
ckpt_dir = '/content/drive/MyDrive/jepa_checkpoints/exp7'
log = open(os.path.join(ckpt_dir, 'train_full.log'), 'w')
env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
env['WANDB_API_KEY'] = '$WANDB_API_KEY'
resume_path = os.path.join(ckpt_dir, 'exp7_latest.pt')
args = ['python3', '/content/experiments/exp7_hybrid_decision/train.py'] + '$TRAIN_ARGS'.split() + [
     '--wandb_project', 'jepa-exp7', '--wandb_run_id', '$WANDB_RUN_ID', '--ckpt_dir', ckpt_dir]
if os.path.exists(resume_path):
    args += ['--resume', resume_path]
    print('resuming from', resume_path)
else:
    print('no checkpoint found -- starting fresh')
proc = subprocess.Popen(args, cwd='/content/experiments/exp7_hybrid_decision',
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
with open('/content/logs/train_full.pid', 'w') as f:
    f.write(str(proc.pid))
print('launched, pid', proc.pid, 'args', args)
PYEOF

echo ""
echo "=== Training relaunched. Switching to live monitor. ==="
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

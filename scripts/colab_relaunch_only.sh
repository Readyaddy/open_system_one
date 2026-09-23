#!/bin/bash
# Use this when the session/Drive/data are already set up (from a
# previous full run of colab_full_setup_and_launch.sh) and you just need
# to push updated code and relaunch training -- skips session creation,
# drivemount, and data upload entirely.

set -e
source ~/colab-cli-env/bin/activate

SESSION=jepa-train
LOCAL_ROOT=/mnt/d/projects/JEPA
export WANDB_API_KEY="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"

echo "=== Verifying session is reachable ==="
colab --auth=adc status -s "$SESSION"

echo "=== Re-uploading updated train.py ==="
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/experiments/exp6_diverse_data_qqp_aux/train.py" /content/experiments/exp6_diverse_data_qqp_aux/train.py

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
ckpt_dir = '/content/drive/MyDrive/jepa_checkpoints/exp6'
log = open(os.path.join(ckpt_dir, 'train_full.log'), 'w')
env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
env['WANDB_API_KEY'] = '$WANDB_API_KEY'
resume_path = os.path.join(ckpt_dir, 'exp6_latest.pt')
args = ['python3', '/content/experiments/exp6_diverse_data_qqp_aux/train.py',
     '--epochs', '30', '--batch_size', '384', '--qqp_batch_size', '384', '--mcq_batch_size', '224',
     '--grad_accum', '1', '--outcome_chunk_size', '255',
     '--qqp_every_n_steps', '1', '--mcq_every_n_steps', '1', '--patience', '8',
     '--intent_options', '50',
     '--wandb_project', 'jepa-exp6', '--wandb_run_id', 'exp6',
     '--ckpt_dir', ckpt_dir]
if os.path.exists(resume_path):
    args += ['--resume', resume_path]
    print('resuming from', resume_path)
else:
    print('no checkpoint found -- starting fresh')
proc = subprocess.Popen(args, cwd='/content/experiments/exp6_diverse_data_qqp_aux',
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
with open('/content/logs/train_full.pid', 'w') as f:
    f.write(str(proc.pid))
print('launched, pid', proc.pid)
PYEOF

echo ""
echo "=== Training relaunched with per-step progress logging. Switching to live monitor. ==="
sleep 5

set +e
while true; do
  clear
  echo "=== $(date) ==="
  cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 30 2>&1
import subprocess
print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                       '--format=csv'], capture_output=True, text=True).stdout)
pid = open('/content/logs/train_full.pid').read().strip()
ps = subprocess.run(['ps', '-p', pid, '-o', 'pid,stat,etimes'], capture_output=True, text=True).stdout
print(ps if len(ps.strip().splitlines()) > 1 else f'PROCESS {pid} NOT FOUND -- may have crashed/disconnected')
lines = open('/content/drive/MyDrive/jepa_checkpoints/exp6/train_full.log').read().splitlines()
step_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' in l][-10:]
summary_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' not in l][-3:]
other_recent = [l for l in lines if not l.strip().startswith('epoch')][-5:]
print('\n'.join(other_recent))
print('--- recent per-step progress ---')
print('\n'.join(step_lines))
print('--- recent epoch summaries ---')
print('\n'.join(summary_lines))
PYEOF
  sleep 30
done

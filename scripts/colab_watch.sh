#!/bin/bash
# Standalone live monitor -- run anytime, independently of setup/launch.
# Shows GPU stats, process status, and recent per-step training progress,
# refreshing every 30s. Survives the flaky "Connection was lost" errors
# by just retrying instead of exiting. Ctrl+C stops watching only --
# training itself keeps running on the VM regardless.

source ~/colab-cli-env/bin/activate
SESSION=jepa-train

while true; do
  clear
  echo "=== $(date) ==="
  cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 30 2>&1
import subprocess
print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                       '--format=csv'], capture_output=True, text=True).stdout)
try:
    pid = open('/content/logs/train_full.pid').read().strip()
    ps = subprocess.run(['ps', '-p', pid, '-o', 'pid,stat,etimes'], capture_output=True, text=True).stdout
    print(ps if len(ps.strip().splitlines()) > 1 else f'PROCESS {pid} NOT FOUND -- may have crashed/disconnected')
except FileNotFoundError:
    print('no pid file found -- training may not be launched yet')
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

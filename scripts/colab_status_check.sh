#!/bin/bash
# One-shot status check -- prints current progress once and exits.
# Use this for a quick glance; use colab_watch.sh for continuous monitoring.

source ~/colab-cli-env/bin/activate
SESSION=jepa-train

echo "=== Session status ==="
colab --auth=adc status -s "$SESSION"

echo ""
echo "=== Training progress ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 30
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
step_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' in l][-5:]
summary_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' not in l][-3:]
print('--- recent per-step progress ---')
print('\n'.join(step_lines))
print('--- recent epoch summaries ---')
print('\n'.join(summary_lines))
PYEOF

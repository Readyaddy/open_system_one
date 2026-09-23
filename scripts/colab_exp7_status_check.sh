#!/bin/bash
# One-shot status check for exp7 -- prints current progress once and exits.
# Use this for a quick glance; use colab_exp7_watch.sh for continuous
# monitoring. Mirrors exp6's colab_status_check.sh.
#
# AUTO-RECONNECT: if local session tracking was pruned (see
# colab_reconnect.py's docstring), retries once after reconnecting instead
# of just printing "Session not found" and exiting.

source ~/colab-cli-env/bin/activate
SESSION=jepa-exp7-train
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

status_output() {
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
lines = open('/content/drive/MyDrive/jepa_checkpoints/exp7/train_full.log').read().splitlines()
step_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' in l][-5:]
summary_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' not in l][-3:]
print('--- recent per-step progress ---')
print('\n'.join(step_lines))
print('--- recent epoch summaries ---')
print('\n'.join(summary_lines))
PYEOF
}

echo "=== Session status ==="
colab --auth=adc status -s "$SESSION"

echo ""
echo "=== Training progress ==="
OUTPUT=$(status_output)
STATUS=$?
echo "$OUTPUT"

if [ "$STATUS" -ne 0 ] || echo "$OUTPUT" | grep -qi "session .* not found"; then
  echo "!!! '$SESSION' unreachable -- auto-reconnecting... !!!"
  if python3 "$SCRIPT_DIR/colab_reconnect.py" "$SESSION"; then
    echo "!!! Reconnected. Retrying status check. !!!"
    colab --auth=adc status -s "$SESSION"
    echo ""
    echo "=== Training progress (retried) ==="
    status_output
  else
    echo "!!! Auto-reconnect failed -- run manually: bash $SCRIPT_DIR/colab_reconnect.sh $SESSION --endpoint <endpoint>"
  fi
fi

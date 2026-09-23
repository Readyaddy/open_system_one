#!/bin/bash
# Standalone live monitor for exp7 -- run anytime, independently of
# setup/launch. Shows GPU stats, process status, and recent per-step
# training progress, refreshing every 30s. Survives flaky "Connection was
# lost" errors by just retrying instead of exiting. Ctrl+C stops watching
# only -- training itself keeps running on the VM regardless. Mirrors
# exp6's colab_watch.sh.
#
# AUTO-RECONNECT: if the local session name gets pruned from colab-cli's
# tracking (colab_cli/common.py's sync_sessions() does this on a single
# flaky assignments listing -- confirmed to happen with training completely
# unaffected, see colab_reconnect.py's docstring), this loop detects the
# resulting "Session not found" failure, runs colab_reconnect.py to
# reattach to the still-alive VM, and resumes watching automatically --
# instead of silently showing a stale screen or erroring out.
#
# This is also where you should read the gpu_mem figure printed on every
# per-step progress line to decide whether --batch_size has headroom left
# (target ~35GB/40GB on an A100, 5GB spare) before raising it via
# colab_exp7_relaunch_only.sh.

source ~/colab-cli-env/bin/activate
SESSION=jepa-exp7-train
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_and_reconnect() {
  # $1 = captured output, $2 = exit status of the exec call that produced it
  if [ "$2" -eq 0 ] && ! echo "$1" | grep -qi "session .* not found"; then
    return 0
  fi
  echo "!!! '$SESSION' unreachable (local tracking likely pruned, VM is probably still alive) -- auto-reconnecting... !!!"
  if python3 "$SCRIPT_DIR/colab_reconnect.py" "$SESSION"; then
    echo "!!! Reconnected. Resuming watch. !!!"
    return 0
  else
    echo "!!! Auto-reconnect failed (see message above -- e.g. ambiguous: multiple orphaned VMs)."
    echo "!!! Run manually: bash $SCRIPT_DIR/colab_reconnect.sh $SESSION --endpoint <endpoint> !!!"
    return 1
  fi
}

while true; do
  clear
  echo "=== $(date) ==="
  OUTPUT=$(cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 30 2>&1
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
step_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' in l][-10:]
summary_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' not in l][-3:]
other_recent = [l for l in lines if not l.strip().startswith('epoch')][-5:]
print('\n'.join(other_recent))
print('--- recent per-step progress ---')
print('\n'.join(step_lines))
print('--- recent epoch summaries ---')
print('\n'.join(summary_lines))
PYEOF
)
  STATUS=$?
  echo "$OUTPUT"

  if ! check_and_reconnect "$OUTPUT" "$STATUS"; then
    echo "!!! Will retry auto-reconnect again in 30s. !!!"
  fi
  sleep 30
done

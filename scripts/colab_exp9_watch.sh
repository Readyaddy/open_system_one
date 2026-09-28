#!/bin/bash
# Live view of an exp9 Colab run: GPU, process, recent steps, last eval.
# Ctrl+C stops watching only. Reconnects if colab-cli loses the session name.
#   bash /mnt/d/projects/JEPA/scripts/colab_exp9_watch.sh            (RUN=exp9a by default)
source ~/colab-cli-env/bin/activate
RUN=${RUN:-exp9a}
SESSION=${SESSION:-openjev-exp9-train}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while true; do
  OUT=$(cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 40 2>&1
import subprocess, os, glob, json
print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
                      '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
try:
    pid = open('/content/train.pid').read().strip()
    alive = subprocess.run(['ps', '-p', pid], capture_output=True).returncode == 0
    print('train process', pid, 'RUNNING' if alive else 'EXITED')
except FileNotFoundError:
    print('no train.pid')
out = '/content/runs/$RUN' if os.path.isdir('/content/runs/$RUN') else '/content/drive/MyDrive/openjev/runs/$RUN'
lines = open(f'{out}/train.log').read().splitlines() if os.path.exists(f'{out}/train.log') else []
print('\n'.join([l for l in lines if 'step ' in l][-6:]))
print('\n'.join([l for l in lines if 'SELECT' in l or 'best' in l or 'depth curve' in l or 'stopping' in l or 'done.' in l][-4:]))
evals = sorted(glob.glob(f'{out}/eval_*.json'))
if evals:
    m = json.load(open(evals[-1]))['metrics']
    sw = {k: round(v * 100, 1) for k, v in m.items() if k.startswith('sweep/') and k.endswith('/acc')}
    print('last eval', os.path.basename(evals[-1]), 'sweep:', json.dumps(sw))
print('DONE' if os.path.exists(f'{out}/DONE') else '')
PYEOF
)
  clear; echo "=== $(date) | $RUN ==="; echo "$OUT"
  if echo "$OUT" | grep -qi "session .* not found"; then
    python3 "$SCRIPT_DIR/colab_reconnect.py" "$SESSION" || echo "(VM gone -- released at end of run, or disconnected)"
  fi
  echo "$OUT" | grep -q "^DONE" && { echo "=== run finished; results on Drive: MyDrive/openjev/runs/$RUN ==="; break; }
  sleep 60
done

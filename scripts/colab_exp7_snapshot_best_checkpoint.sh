#!/bin/bash
# Snapshots exp7a's current best-zero-shot checkpoint to a STABLE, separate
# filename on the same Drive -- so the frozen-backbone recursion/head-only
# continuation phase (train.py --freeze_backbone --resume <snapshot>) has a
# fixed, known-good weight source, independent of exp7a continuing to
# overwrite exp7_best_zeroshot.pt as it finds new bests while this runs.
#
# Runs entirely on the IDLE CPU session (jepa-exp7-cpu), not the A100 --
# this never touches the A100 training process, not even a read on it.
# Both sessions mount the SAME Google Drive, so nothing needs to be
# downloaded anywhere; this is a plain file copy on the Drive mount.
#
# "Best" here means exp7_best_zeroshot.pt specifically, not
# exp7_best_val.pt -- NOTES.md Sec 4.2 and this project's own established
# practice key everything (early stopping, "which checkpoint to trust") to
# zero-shot accuracy, not in-distribution val_acc, which has repeatedly
# been shown to be a misleading signal on its own (see PROJECT_HISTORY.md /
# Key Lessons).
#
# Usage:
#   wsl -d kali-linux
#   bash /mnt/d/projects/JEPA/scripts/colab_exp7_snapshot_best_checkpoint.sh                    # tries jepa-exp7-cpu
#   bash /mnt/d/projects/JEPA/scripts/colab_exp7_snapshot_best_checkpoint.sh jepa-exp7-train     # explicit session
#
# Output: a new file at
#   /content/drive/MyDrive/jepa_checkpoints/exp7_frozen_seed/exp7a_best_zeroshot_epoch<N>.pt
# plus a manifest.json alongside it recording exactly which run/epoch/
# metrics it was snapshotted from. Prints that path -- pass it straight to
# a later launch script's --resume.

set -e
source ~/colab-cli-env/bin/activate

# Prefer the idle CPU session (never touches A100 at all, not even a read).
# Falls back to the A100 session if the CPU one isn't around -- Colab's own
# idle timeout reclaims CPU sessions faster than GPU ones, and re-creating
# one needs an interactive `colab drivemount` OAuth step (COLAB_SKILL.md),
# so it can't be done unattended. The A100 fallback is still a single quick
# `colab exec` file copy -- the same class of read-only call every
# colab_exp7_status_check.sh/watch.sh run already makes -- NOT a code push
# and NOT anything that touches the detached training subprocess (PID
# tracked separately in train_full.pid, never referenced here).
SESSION="${1:-jepa-exp7-cpu}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_and_reconnect() {
  if [ "$2" -eq 0 ] && ! echo "$1" | grep -qi "session .* not found"; then
    return 0
  fi
  echo "!!! '$SESSION' unreachable -- auto-reconnecting... !!!"
  if python3 "$SCRIPT_DIR/colab_reconnect.py" "$SESSION"; then
    echo "!!! Reconnected. Retrying. !!!"
    return 0
  else
    echo "!!! Auto-reconnect failed -- run manually: bash $SCRIPT_DIR/colab_reconnect.sh $SESSION --endpoint <endpoint>"
    echo "!!! If '$SESSION' genuinely doesn't exist yet, run colab_exp7_cpu_probe.sh first --"
    echo "!!! this script needs an existing session with Drive already mounted, it doesn't create one."
    return 1
  fi
}

run_snapshot() {
  cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 60 2>&1
import json, os, shutil, time

src_dir = '/content/drive/MyDrive/jepa_checkpoints/exp7'
src = os.path.join(src_dir, 'exp7_best_zeroshot.pt')
if not os.path.exists(src):
    print(f'ERROR: {src} does not exist -- has exp7a saved a best-zeroshot checkpoint yet?')
    raise SystemExit(1)

dst_dir = '/content/drive/MyDrive/jepa_checkpoints/exp7_frozen_seed'
os.makedirs(dst_dir, exist_ok=True)

# Read metadata WITHOUT importing torch/the model classes -- checkpoints
# are plain torch.save'd dicts, and torch.load can read the small metadata
# fields even without model code available, as long as weights_only=False
# is not needed for that (the dict's top-level int/float entries load fine
# via a lightweight pass). If this environment doesn't have torch for some
# reason, fall back to copying blind rather than failing the whole snapshot.
epoch = 'unknown'
metrics = {}
try:
    import torch
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', 'unknown')
    metrics = {k: ckpt.get(k) for k in ('val_acc', 'zero_shot_acc', 'banking77_holdout_acc') if k in ckpt}
    del ckpt
except Exception as e:
    print(f'(could not read checkpoint metadata: {e} -- copying blind, filename will say epoch_unknown)')

dst_name = f'exp7a_best_zeroshot_epoch{epoch}.pt'
dst = os.path.join(dst_dir, dst_name)

t0 = time.time()
shutil.copy2(src, dst)
dt = time.time() - t0

manifest = {
    'snapshotted_from': src, 'source_epoch': epoch, 'source_metrics': metrics,
    'snapshot_path': dst, 'snapshotted_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'note': 'Stable copy, decoupled from exp7a continuing to overwrite exp7_best_zeroshot.pt. '
            'Use this path as --resume for the --freeze_backbone continuation phase.',
}
with open(os.path.join(dst_dir, dst_name.replace('.pt', '_manifest.json')), 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'Snapshotted: {src}')
print(f'         -> {dst}  ({dt:.1f}s, {os.path.getsize(dst) / 1e6:.1f} MB)')
print(f'Source epoch: {epoch}   metrics: {metrics}')
print(f'\nFor the recursion continuation launch, use:')
print(f'  --resume {dst} --freeze_backbone --reset_training_state --k_max 6 --ckpt_prefix exp7_frozen')
print(f'(--reset_training_state only on this FIRST launch off exp7a weights -- a later relaunch of')
print(f'this same frozen phase should resume its own optimizer/epoch state, just keep --freeze_backbone)')
PYEOF
}

echo "=== Checking session ==="
colab --auth=adc status -s "$SESSION"

echo ""
echo "=== Snapshotting exp7a's current best-zero-shot checkpoint ==="
OUTPUT=$(run_snapshot)
STATUS=$?
echo "$OUTPUT"

if ! check_and_reconnect "$OUTPUT" "$STATUS"; then
  exit 1
fi
if [ "$STATUS" -ne 0 ] && echo "$OUTPUT" | grep -qi "session .* not found"; then
  echo "=== Retrying snapshot after reconnect ==="
  run_snapshot
fi

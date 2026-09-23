#!/bin/bash
# Reconnects a locally-orphaned colab-cli session name to the Colab VM it
# actually belongs to, when `colab status -s <name>` / `colab exec -s <name>`
# fail with "Session not found" but `colab sessions` still shows the VM
# alive (unnamed, as `[?]`). See colab_reconnect.py's module docstring for
# the full why -- this is a thin wrapper so you don't need to remember the
# WSL/venv activation dance every time.
#
# Usage:
#   wsl -d kali-linux
#   bash /mnt/d/projects/JEPA/scripts/colab_reconnect.sh <name> [--endpoint <endpoint>]
#
# Examples:
#   bash colab_reconnect.sh jepa-exp7-train
#   bash colab_reconnect.sh jepa-exp7-train --endpoint gpu-a100-s-kkb-usc1b0-2k4m04zh1op5e
#
# After it reports success, verify with:
#   colab --auth=adc status -s <name>

source ~/colab-cli-env/bin/activate
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/colab_reconnect.py" "$@"

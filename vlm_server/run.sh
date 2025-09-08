#!/usr/bin/env bash
set -euo pipefail

echo "Starting VLM Server"

# Activate conda env if available
if command -v conda >/dev/null 2>&1; then
  # Prefer a named env VLM_SERVER if exists, else current
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | grep -q "vlm_server"; then
    conda activate vlm_server
  fi
fi

export PYTHONUNBUFFERED=1

# Avoid forcing CUDA on macOS
UNAME=$(uname -s || echo "")
if [[ "$UNAME" != "Darwin" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}

# Ensure Python can import the package from repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

exec python -m uvicorn vlm_server.server:app --host "$HOST" --port "$PORT" --no-server-header

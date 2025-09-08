#!/usr/bin/env bash
set -euo pipefail

echo "VLM: one-command server launcher"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OS_NAME="$(uname -s || echo unknown)"
HAS_CONDA=0
HAS_GPU=0

if command -v conda >/dev/null 2>&1; then
  HAS_CONDA=1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1
fi

PYTHON_BIN="python3"

create_or_activate_env() {
  if [[ $HAS_CONDA -eq 1 ]]; then
    echo "Using conda environment: vlm_server"
    # Avoid nounset errors from third-party deactivate scripts
    # shellcheck disable=SC1091
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | grep -q "^vlm_server\b"; then
      conda create -y -n vlm_server python=3.10
    fi
    conda activate vlm_server
    set -u
    PYTHON_BIN="python"
  else
    echo "Conda not found; using venv .venv_vlm_server"
    if [[ ! -d .venv_vlm_server ]]; then
      python3 -m venv .venv_vlm_server
    fi
    # shellcheck disable=SC1091
    source .venv_vlm_server/bin/activate
    PYTHON_BIN="python"
  fi
}

install_python_deps() {
  echo "Installing Python dependencies (best-effort)…"
  $PYTHON_BIN -m pip install -U pip setuptools wheel >/dev/null 2>&1 || true

  # Prefer CUDA-enabled PyTorch on Linux with NVIDIA GPU
  if [[ "$OS_NAME" != "Darwin" && $HAS_GPU -eq 1 ]]; then
    echo "Installing CUDA-enabled PyTorch (cu121) …"
    $PYTHON_BIN -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision || true
  fi

  # Base requirements for the server
  if [[ -f vlm_server/requirements.txt ]]; then
    $PYTHON_BIN -m pip install -r vlm_server/requirements.txt || true
  else
    $PYTHON_BIN -m pip install fastapi uvicorn[standard] transformers torch torchvision pillow opencv-python numpy psutil python-multipart || true
  fi

  # Ensure core server deps present even if the requirements step partially failed
  $PYTHON_BIN -m pip install -U fastapi uvicorn[standard] || true

  # Try optional acceleration libs conditionally
  if [[ "$OS_NAME" != "Darwin" && $HAS_GPU -eq 1 ]]; then
    # vLLM (CUDA only); ignore failures
    $PYTHON_BIN -m pip install vllm || true
  fi

  # Audio transcription stack; ignore failures
  $PYTHON_BIN -m pip install faster-whisper soundfile || true
}

create_or_activate_env
install_python_deps

echo "Starting VLM Server…"
cd vlm_server
exec ./run.sh

#!/usr/bin/env bash

echo "VLM Video QA - Web Interface"
echo "============================"

if command -v conda &> /dev/null; then
  echo "Activating environment: vlm-web"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate vlm-web || true
else
  echo "Conda not found, using system Python"
fi

export PYTHONUNBUFFERED=1
echo "Starting Video QA web server (port 5050)…"
python vlm_video_qa.py


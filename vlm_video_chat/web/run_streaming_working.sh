#!/bin/bash

echo "VLM Video Chat - Streaming Working Version"
echo "=========================================="

# Set CUDA environment
export CUDA_HOME=/usr
export CUDA_ROOT=/usr
export CUDA_PATH=/usr
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Clear any existing processes using GPU
python -c "
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
print('GPU memory cleared')
" 2>/dev/null || true

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vlm

echo "Starting streaming VLM interface..."
echo "This version includes:"
echo "- Real-time response streaming"
echo "- Collapsible images"  
echo "- Markdown formatting"
echo "- Conversation grouping"
echo "- Memory-optimized processing"
echo ""
echo "Available at: http://localhost:5000"
echo ""

python vlm_streaming_working.py
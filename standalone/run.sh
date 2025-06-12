#!/bin/bash

echo "VLM Video Chat - Standalone"
echo "==========================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Activate conda environment
echo "Activating environment: vlm-standalone"
eval "$(conda shell.bash hook)"
conda activate vlm-standalone

# Check if environment exists
if [ $? -ne 0 ]; then
    echo "Error: 'vlm-standalone' conda environment not found."
    echo "Please run './setup.sh' first to create the environment."
    exit 1
fi

# Set CUDA environment variables
export CUDA_HOME=/usr
export CUDA_ROOT=/usr
export CUDA_PATH=/usr
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Starting VLM Video Chat..."
python vlm_standalone.py
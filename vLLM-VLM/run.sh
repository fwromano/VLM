#!/bin/bash

echo "vLLM Simple VLM"
echo "==============="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Activating environment: vllm_vlm"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate vllm_vlm
else
    echo "Using system Python"
fi

# Set CUDA environment variables
export CUDA_HOME=/usr
export CUDA_ROOT=/usr
export CUDA_PATH=/usr
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Check if no arguments provided
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage Options:"
    echo ""
    echo "1. Interactive Camera Mode:"
    echo "   ./run.sh --interactive"
    echo ""
    echo "2. Single Image Analysis:"
    echo "   ./run.sh --image path/to/image.jpg --question 'What do you see?'"
    echo ""
    echo "3. Quick Test:"
    echo "   ./run.sh --test"
    echo ""
    echo "4. Different Model:"
    echo "   ./run.sh --model google/gemma-3-12b-it --interactive"
    echo ""
    echo "Examples:"
    echo "   ./run.sh --interactive"
    echo "   ./run.sh --image photo.jpg --question 'Describe this scene'"
    echo "   ./run.sh --test"
    echo ""
    exit 0
fi

# Run the VLM program with all arguments
python vlm_simple.py "$@"
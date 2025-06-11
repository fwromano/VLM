#!/bin/bash
# run_vlm.sh - VLM Video Chat Launcher

echo "VLM Video Chat Launcher"
echo "======================="

# Activate conda environment
source /home/fwromano/anaconda3/bin/activate vlm

# Check if we're in the right environment
if [ "$CONDA_DEFAULT_ENV" != "vlm" ]; then
    echo "Failed to activate vlm environment"
    echo "Run: conda activate vlm"
    exit 1
fi

echo "Environment activated: $CONDA_DEFAULT_ENV"

# Run the VLM application
echo "Starting VLM Video Chat..."
python /home/fwromano/Documents/Code/VLM/vlm.py
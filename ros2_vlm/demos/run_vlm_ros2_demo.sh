#!/bin/bash

# VLM ROS2-Style Demo (No ROS2 Required)
# This runs VLM with a ROS2-like interface without needing ROS2 installed

echo "====================================="
echo "VLM ROS2-Style Demo"
echo "====================================="
echo ""
echo "This demo provides ROS2-like functionality without ROS2 installation."
echo ""

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^vlm "; then
        echo "Activating VLM environment..."
        conda activate vlm
    else
        echo "Error: VLM conda environment not found!"
        echo "Please run ./setup.sh first"
        exit 1
    fi
else
    echo "Warning: Conda not found, using system Python"
fi

# Check if camera exists
if [ ! -e "/dev/video0" ]; then
    echo "Warning: No camera at /dev/video0"
    echo "Available devices:"
    ls /dev/video* 2>/dev/null || echo "No video devices found"
fi

# Run the standalone version
python run_vlm_ros2_standalone.py "$@"
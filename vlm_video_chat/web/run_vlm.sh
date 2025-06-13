#!/bin/bash

echo "VLM Video Chat - Web Interface"
echo "==============================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Activating environment: vlm-web"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate vlm-web
    
    # Check if environment exists
    if [ $? -ne 0 ]; then
        echo "Error: 'vlm-web' conda environment not found."
        echo "Please run './setup.sh' first to create the environment."
        exit 1
    fi
else
    echo "Conda not found, using system Python"
fi

# Set CUDA environment variables
export CUDA_HOME=/usr
export CUDA_ROOT=/usr
export CUDA_PATH=/usr
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Starting Web VLM Video Chat..."

# Check if Flask dependencies are installed
python -c "import flask, flask_socketio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Flask dependencies..."
    pip install flask>=2.3.0 flask-socketio>=5.3.0
fi

echo ""
echo "🚀 Starting VLM Video Chat Web Server..."
echo "📹 Camera resolution: 1280x720 → 896x896 (model input)"
echo "🤖 Models available: Gemma 3 4B, 12B (quantized)"
echo "🌐 Web interface: http://localhost:5000"
echo "📱 Mobile friendly: Access from any device on your network"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the web interface
python vlm.py
#!/bin/bash

echo "vLLM VLM Video Chat - Web Interface"
echo "==================================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Activating environment: vllm_vlm"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate vllm_vlm
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

echo "Starting vLLM Web VLM Video Chat..."

# Check if vLLM and Flask dependencies are installed
python -c "import vllm, flask, flask_socketio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing missing dependencies..."
    pip install vllm flask>=2.3.0 flask-socketio>=5.3.0
fi

echo ""
echo "🚀 Starting vLLM VLM Video Chat Web Server..."
echo "⚡ Backend: vLLM (High-Performance - 2-5x faster)"
echo "📹 Camera resolution: 1280x720 → 896x896 (model input)"
echo "🤖 Models available: Gemma 3 4B, 12B (auto-quantized by vLLM)"
echo "🌐 Web interface: http://localhost:5000"
echo "📱 Mobile friendly: Access from any device on your network"
echo "💾 Memory optimized: PagedAttention for 40-60% less VRAM"
echo ""
echo "Performance benefits:"
echo "  • 2-5x faster inference than transformers"
echo "  • Continuous batching for better throughput"
echo "  • Advanced CUDA kernel optimizations"
echo "  • Automatic quantization for large models"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the vLLM web interface
python vlm_web.py
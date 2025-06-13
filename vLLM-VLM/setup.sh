#!/bin/bash

echo "vLLM Simple VLM Setup"
echo "===================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Setting up conda environment: vllm_vlm"
    
    # Create environment if it doesn't exist
    if ! conda env list | grep -q "vllm_vlm"; then
        echo "Creating new conda environment..."
        conda create -n vllm_vlm python=3.10 -y
    fi
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate vllm_vlm
    
    echo "Environment 'vllm_vlm' activated"
else
    echo "Conda not found, using system Python"
    echo "Warning: Recommended to use conda for better dependency management"
fi

# Set CUDA environment variables
export CUDA_HOME=/usr
export CUDA_ROOT=/usr
export CUDA_PATH=/usr
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "Installing dependencies..."

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (this may take a while)
echo "Installing vLLM (this may take several minutes)..."
pip install vllm

# Install other dependencies
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."

python -c "
import torch
import vllm
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ vLLM {vllm.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"

echo ""
echo "Setup complete! 🚀"
echo ""
echo "Usage:"
echo "  # Command-line interface"
echo "  python vlm_simple.py --image path/to/image.jpg --question 'What do you see?'"
echo "  python vlm_simple.py --interactive"
echo "  python vlm_simple.py --test"
echo ""
echo "  # Web interface (Recommended)"
echo "  ./run_web.sh"
echo "  # Then open: http://localhost:5000"
echo ""
echo "  # Quick launcher"
echo "  ./run.sh --interactive     # Command-line mode"
echo "  ./run_web.sh              # Web interface mode"
echo ""
echo "Note: First run will download the Gemma 3 model (~8GB)"
echo "vLLM provides 2-5x faster inference than standard transformers!"
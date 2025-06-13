#!/bin/bash

echo "VLM Standalone Setup"
echo "===================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Anaconda/Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'vlm-standalone'..."
conda create -n vlm-standalone python=3.10 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate vlm-standalone

# Detect CUDA version and install matching PyTorch
echo "Detecting CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "Detected CUDA version: $CUDA_VERSION"
    
    # Use CUDA 11.8 for better compatibility with existing CUDA installations
    echo "Installing PyTorch with CUDA 11.8 support (for compatibility)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    TORCH_CUDA="cu118"
else
    echo "CUDA not detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    TORCH_CUDA="cpu"
fi

# Install other requirements
echo "Installing standalone requirements..."
pip install -r requirements.txt

# Install transformers with latest version for Gemma 3 support
echo "Installing transformers for Gemma 3..."
pip install -U transformers

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Authenticate with HuggingFace: huggingface-cli login"
echo "2. Accept Gemma license at: https://huggingface.co/google/gemma-3-4b-it"
echo "3. Run the application: ./run_vlm.sh"
echo ""
echo "Note: Make sure tkinter is installed on your system"
echo "Ubuntu/Debian: sudo apt-get install python3-tk"
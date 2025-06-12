#!/bin/bash

echo "VLM Video Chat - Standalone Setup"
echo "================================="

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
    
    if [[ "$CUDA_VERSION" == "12.0" ]] || [[ "$CUDA_VERSION" == "12.1" ]] || [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "Installing PyTorch with CUDA 12.1 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        TORCH_CUDA="cu121"
    elif [[ "$CUDA_VERSION" == "11.8" ]]; then
        echo "Installing PyTorch with CUDA 11.8 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        TORCH_CUDA="cu118"
    else
        echo "Installing PyTorch with CUDA 11.8 support (default)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        TORCH_CUDA="cu118"
    fi
else
    echo "CUDA not detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    TORCH_CUDA="cpu"
fi

# Install other requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install transformers with latest version for Gemma 3 support
echo "Installing transformers for Gemma 3..."
pip install -U transformers

# Install flash attention for performance (if CUDA available and versions match)
echo "Installing flash attention for performance optimization..."
if [[ "$TORCH_CUDA" != "cpu" ]] && command -v nvcc &> /dev/null; then
    echo "CUDA development tools found, installing flash attention..."
    
    # Try installing flash attention with matching CUDA version
    if [[ "$TORCH_CUDA" == "cu121" ]]; then
        echo "Installing flash attention for CUDA 12.1..."
        pip install flash-attn --no-build-isolation || {
            echo "Flash attention compilation failed. Trying pre-built wheel..."
            pip install flash-attn || echo "Flash attention installation failed, continuing without it..."
        }
    elif [[ "$TORCH_CUDA" == "cu118" ]]; then
        echo "Installing flash attention for CUDA 11.8..."
        pip install flash-attn --no-build-isolation || {
            echo "Flash attention compilation failed. Trying pre-built wheel..."  
            pip install flash-attn || echo "Flash attention installation failed, continuing without it..."
        }
    fi
else
    echo "CUDA not available or development tools not found, skipping flash attention."
    echo "Flash attention will be automatically disabled in the application."
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Authenticate with HuggingFace: huggingface-cli login"
echo "2. Accept Gemma license at: https://huggingface.co/google/gemma-3-4b-it"
echo "3. Run the application: ./run.sh"
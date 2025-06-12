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

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing requirements..."
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
echo "3. Run the application: ./run.sh"
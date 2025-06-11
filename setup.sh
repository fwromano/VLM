#!/bin/bash
# VLM Camera Setup - RTX 5000 Ada Optimized

ENV_NAME="vlm"
echo "🚀 Setting up VLM Camera environment..."

# Create conda environment
if ! conda env list | grep -q "^$ENV_NAME\s"; then
    echo "📦 Creating conda environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=3.10 -y
else
    echo "✓ Environment '$ENV_NAME' exists"
fi

# Install PyTorch with CUDA 12.1 for RTX 5000 Ada
echo "🔥 Installing PyTorch with CUDA support..."
conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "📋 Installing VLM dependencies..."
conda run -n $ENV_NAME pip install -r requirements.txt

# Set up environment variables for GPU
echo "🔧 Setting up GPU environment variables..."
mkdir -p $HOME/anaconda3/envs/$ENV_NAME/etc/conda/activate.d
mkdir -p $HOME/anaconda3/envs/$ENV_NAME/etc/conda/deactivate.d

cat > $HOME/anaconda3/envs/$ENV_NAME/etc/conda/activate.d/cuda.sh << 'EOF'
#!/bin/bash
export CUDA_HOME=/usr
export CUDA_ROOT=/usr
export CUDA_PATH=/usr
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EOF

cat > $HOME/anaconda3/envs/$ENV_NAME/etc/conda/deactivate.d/cuda.sh << 'EOF'
#!/bin/bash
unset CUDA_HOME
unset CUDA_ROOT
unset CUDA_PATH
unset CUDA_VISIBLE_DEVICES
unset CUDA_DEVICE_ORDER
EOF

chmod +x $HOME/anaconda3/envs/$ENV_NAME/etc/conda/activate.d/cuda.sh
chmod +x $HOME/anaconda3/envs/$ENV_NAME/etc/conda/deactivate.d/cuda.sh

echo ""
echo "✅ Setup complete!"
echo "To run: conda activate vlm && python camera_vlm.py"
echo ""
echo "🔧 GPU environment variables will be automatically set when you activate the environment."
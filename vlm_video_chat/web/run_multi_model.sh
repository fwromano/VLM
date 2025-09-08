#!/bin/bash
# Run multi-model VLM web interface with dependency management

echo "Starting Multi-Model VLM Video Chat..."
echo "======================================"

# Set environment variables
export CUDA_HOME=/usr
export CUDA_ROOT=/usr
export CUDA_PATH=/usr
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate vlm 2>/dev/null || {
        echo "Warning: Could not activate 'vlm' environment"
        echo "Continuing with current environment..."
    }
fi

# Check Python and CUDA
echo ""
echo "Environment:"
python3 -c "import sys; print(f'Python: {sys.version.split()[0]}')"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: Not installed"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null
python3 -c "import torch; torch.cuda.is_available() and print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check and install dependencies
echo "Checking dependencies..."
echo "========================"

# Function to check if a Python package meets version requirement
check_package() {
    local package=$1
    local required_version=$2
    
    python3 -c "
import sys
try:
    import importlib.metadata
    version = importlib.metadata.version('$package')
    print(f'$package: {version}')
    
    # Check version requirement if specified
    if '$required_version':
        from packaging import version as v
        if v.parse(version) < v.parse('$required_version'):
            sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null
    return $?
}

# Install missing or outdated packages
install_if_needed() {
    local package=$1
    local version_spec=$2
    local package_name=$(echo $package | cut -d'>' -f1 | cut -d'=' -f1)
    local required_version=$(echo $version_spec | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "")
    
    if ! check_package "$package_name" "$required_version"; then
        echo "Installing/updating $package..."
        pip install "$package$version_spec" --upgrade
    else
        echo "✓ $package_name is up to date"
    fi
}

# Check for packaging (needed for version comparison)
python3 -c "import packaging" 2>/dev/null || pip install packaging

# Read requirements and check each one
if [ -f "requirements.txt" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        if [[ -z "$line" || "$line" =~ ^# ]]; then
            continue
        fi
        
        # Extract package name and version
        package=$(echo "$line" | sed 's/\s*#.*//')  # Remove inline comments
        if [[ -n "$package" ]]; then
            install_if_needed "$package" ""
        fi
    done < requirements.txt
else
    echo "Warning: requirements.txt not found, checking core dependencies..."
    install_if_needed "transformers" ">=4.53.0"
    install_if_needed "torch" ">=2.0.0"
    install_if_needed "flask" ">=2.3.0"
    install_if_needed "flask-socketio" ">=5.3.0"
    install_if_needed "opencv-python" ">=4.8.0"
    install_if_needed "pillow" ">=9.0.0"
    install_if_needed "accelerate" ">=0.20.0"
fi

echo ""
echo "Dependency check complete!"
echo ""

# Verify transformers version specifically for Gemma-3n
echo "Verifying Gemma-3n support..."
python3 -c "
import transformers
from packaging import version
if version.parse(transformers.__version__) < version.parse('4.53.0'):
    print(f'Warning: transformers {transformers.__version__} is older than 4.53.0')
    print('Gemma-3n requires transformers>=4.53.0')
    print('Updating now...')
    import subprocess
    subprocess.run(['pip', 'install', 'transformers>=4.53.0', '--upgrade'])
else:
    print(f'✓ transformers {transformers.__version__} supports Gemma-3n')
"

echo ""

# Run the multi-model VLM application
echo "Launching VLM Multi-Model Interface..."
echo ""
python3 vlm_multi_model.py
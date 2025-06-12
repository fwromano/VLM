#!/bin/bash

echo "VLM ROS2 Integration Launcher"
echo "============================"

# Function to clean conda from environment
clean_conda_env() {
    echo "Cleaning conda from environment..."
    
    # Deactivate conda completely
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true
    
    # Remove conda from PATH
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v anaconda | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
    
    # Clear conda variables
    unset PYTHONPATH
    unset CONDA_DEFAULT_ENV
    unset CONDA_PREFIX
    
    echo "✓ Environment cleaned"
}

# Function to check ROS2 is working
test_ros2() {
    echo "Testing ROS2..."
    /usr/bin/python3 -c "
import rclpy
print('✓ ROS2 working')
" 2>/dev/null
    return $?
}

# Function to check VLM conda environment
test_vlm_conda() {
    echo "Testing VLM conda environment..."
    if [ -d "/home/fwromano/anaconda3/envs/vlm" ]; then
        echo "✓ VLM conda environment found"
        return 0
    else
        echo "❌ VLM conda environment not found"
        echo "Create it with: cd ../standalone && ./setup.sh"
        return 1
    fi
}

# Main execution
main() {
    echo ""
    echo "Step 1: Cleaning environment..."
    clean_conda_env
    
    echo ""
    echo "Step 2: Sourcing ROS2..."
    source /opt/ros/jazzy/setup.bash
    
    echo ""
    echo "Step 3: Testing ROS2..."
    if ! test_ros2; then
        echo "❌ ROS2 not working. Run: sudo ./install_ros2_packages.sh"
        exit 1
    fi
    
    echo ""
    echo "Step 4: Testing VLM conda environment..."
    if ! test_vlm_conda; then
        exit 1
    fi
    
    echo ""
    echo "Step 5: Starting hybrid VLM node..."
    echo "Python executable: $(which python3)"
    echo "ROS_DISTRO: $ROS_DISTRO"
    echo ""
    
    # Run the hybrid VLM node
    /usr/bin/python3 hybrid_vlm_node.py
}

# Handle Ctrl+C gracefully
trap 'echo ""; echo "Shutting down VLM ROS2 node..."; exit 0' INT

main "$@"
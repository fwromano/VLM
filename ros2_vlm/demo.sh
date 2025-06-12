#!/bin/bash

echo "VLM Demo Launcher"
echo "================="

# Function to clean conda from environment
clean_env() {
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v anaconda | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
    unset PYTHONPATH CONDA_DEFAULT_ENV CONDA_PREFIX
}

# Function to check requirements
check_requirements() {
    echo "Checking requirements..."
    
    # Check camera
    if [ ! -e "/dev/video0" ]; then
        echo "ERROR: No camera found at /dev/video0"
        return 1
    fi
    echo "OK: Camera found"
    
    # Check VLM conda environment
    if [ ! -d "/home/fwromano/anaconda3/envs/vlm" ]; then
        echo "ERROR: VLM conda environment not found"
        echo "Create it with: cd ../standalone && ./setup.sh"
        return 1
    fi
    echo "OK: VLM conda environment found"
    
    return 0
}

# Function to show menu
show_menu() {
    echo ""
    echo "Choose demo type:"
    echo ""
    echo "1. LIVE DEMO (Recommended)"
    echo "   - Integrated camera + VLM analysis"
    echo "   - Real-time overlay"
    echo "   - Keyboard controls for prompts"
    echo "   - Single window interface"
    echo ""
    echo "2. ROS2 DEMO"
    echo "   - ROS2 integration with live camera and analysis"
    echo "   - Two terminal windows"
    echo "   - Full model output displayed"
    echo "   - For robotics development"
    echo ""
    echo "3. TEST VLM ONLY"
    echo "   - Test VLM processor"
    echo "   - No camera needed"
    echo ""
    read -p "Enter choice (1-3): " choice
    echo
}

# Main execution
main() {
    clean_env
    
    if ! check_requirements; then
        exit 1
    fi
    
    show_menu
    
    case $choice in
        1)
            echo "Starting Live Demo..."
            echo ""
            echo "CONTROLS:"
            echo "   H: Show/hide help"
            echo "   1-9: Quick prompts"
            echo "   SPACE: Force analysis"
            echo "   Q/ESC: Quit"
            echo ""
            echo "Starting in 3 seconds..."
            sleep 3
            /usr/bin/python3 demos/live_vlm_demo.py
            ;;
        2)
            echo "Starting ROS2 Demo..."
            ./demos/ros2_camera_demo.sh
            ;;
        3)
            echo "Testing VLM Processor..."
            ./setup/test_vlm_processor.sh
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
}

# Handle Ctrl+C
trap 'echo ""; echo "Demo interrupted."; exit 0' INT

main "$@"
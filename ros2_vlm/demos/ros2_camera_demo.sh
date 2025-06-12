#!/bin/bash

echo "ROS2 Camera Demo Launcher"
echo "========================="

# Function to clean conda from environment
clean_env() {
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v anaconda | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
    unset PYTHONPATH CONDA_DEFAULT_ENV CONDA_PREFIX
}

# Function to check camera
check_camera() {
    if [ ! -e "/dev/video0" ]; then
        echo "ERROR: No camera found at /dev/video0"
        return 1
    fi
    echo "OK: Camera found"
    return 0
}

# Main demo
main() {
    echo ""
    echo "Setting up environment..."
    clean_env
    source /opt/ros/jazzy/setup.bash
    
    echo ""
    echo "Checking camera..."
    if ! check_camera; then
        exit 1
    fi
    
    echo ""
    echo "Starting ROS2 VLM Demo..."
    echo ""
    echo "This will open 2 terminal windows:"
    echo "1. Camera + Image Viewer - Shows live camera feed"
    echo "2. VLM Analysis - Shows full VLM analysis results"
    echo ""
    
    # Check if GUI is available
    if [ -n "$DISPLAY" ]; then
        read -p "Press Enter to start ROS2 demo..."
        
        # Start camera and image viewer in one terminal
        echo "Starting camera node with image viewer..."
        gnome-terminal --tab --title="ROS2 Camera Feed" --geometry=100x30+0+0 -- bash -c "
            export PATH=\$(echo \"\$PATH\" | tr ':' '\\n' | grep -v anaconda | tr '\\n' ':' | sed 's/:\$//')
            source /opt/ros/jazzy/setup.bash
            echo 'ROS2 Camera Feed'
            echo '================'
            echo 'Starting camera node and image viewer...'
            echo ''
            
            # Start camera node in background
            ros2 run usb_cam usb_cam_node_exe --ros-args \\
                -p video_device:=/dev/video0 \\
                -p image_width:=640 \\
                -p image_height:=480 \\
                -p framerate:=15.0 \\
                -r /image_raw:=/camera/image_raw &
            
            CAMERA_PID=\$!
            echo \"Camera node started (PID: \$CAMERA_PID)\"
            
            # Wait a moment for camera to initialize
            sleep 3
            
            # Start image viewer
            echo 'Starting image viewer...'
            ros2 run rqt_image_view rqt_image_view /camera/image_raw &
            VIEWER_PID=\$!
            
            echo ''
            echo 'Camera feed active. Close this terminal to stop camera.'
            echo 'Press Ctrl+C to stop camera node.'
            
            # Wait for camera node
            wait \$CAMERA_PID
            
            # Clean up viewer if still running
            kill \$VIEWER_PID 2>/dev/null
            
            echo 'Camera stopped.'
            read -p 'Press Enter to close...'
        " 2>/dev/null
        
        sleep 3
        
        # Start VLM analysis in second terminal
        echo "Starting VLM analysis node..."
        gnome-terminal --tab --title="VLM Analysis Results" --geometry=100x30+600+0 -- bash -c "
            cd '/home/fwromano/Documents/Code/VLM/ros2_vlm'
            export PATH=\$(echo \"\$PATH\" | tr ':' '\\n' | grep -v anaconda | tr '\\n' ':' | sed 's/:\$//')
            source /opt/ros/jazzy/setup.bash
            echo 'VLM Analysis Results'
            echo '==================='
            echo 'Starting VLM analysis node...'
            echo ''
            
            # Start VLM node in background  
            ./nodes/run_vlm_ros2.sh &
            VLM_PID=\$!
            
            # Wait for VLM node to initialize
            sleep 5
            
            echo 'VLM node started. Listening for analysis results...'
            echo 'Change prompts with:'
            echo '  ros2 topic pub --once /vlm/set_prompt std_msgs/msg/String \"data: YOUR_PROMPT\"'
            echo ''
            echo 'Full VLM Analysis Output:'
            echo '========================='
            echo 'Each analysis result will appear below:'
            echo ''
            
            # Listen to analysis topic with enhanced formatting
            ros2 topic echo /vlm/analysis --field data
            
            # Clean up VLM node
            kill \$VLM_PID 2>/dev/null
            
            echo 'VLM analysis stopped.'
            read -p 'Press Enter to close...'
        " 2>/dev/null
        
        echo ""
        echo "ROS2 Demo started!"
        echo ""
        echo "What you should see:"
        echo "- Camera feed in image viewer window"
        echo "- Full VLM analysis results in second terminal"
        echo ""
        echo "To change prompts, in the VLM terminal run:"
        echo "  ros2 topic pub --once /vlm/set_prompt std_msgs/msg/String \"data: What objects do you see?\""
        echo ""
        echo "To stop: Close the terminal windows or press Ctrl+C in each"
        
    else
        echo "ERROR: No GUI display detected"
        echo ""
        echo "Manual startup (run each in separate terminals):"
        echo ""
        echo "Terminal 1 - Camera:"
        echo "  source /opt/ros/jazzy/setup.bash"
        echo "  ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video0 -r /image_raw:=/camera/image_raw"
        echo ""
        echo "Terminal 2 - VLM Analysis:"
        echo "  ./run_vlm_ros2.sh"
        echo ""
        echo "Terminal 3 - View Results:"
        echo "  source /opt/ros/jazzy/setup.bash"
        echo "  ros2 topic echo /vlm/analysis"
        echo ""
        echo "Change Prompts:"
        echo "  ros2 topic pub --once /vlm/set_prompt std_msgs/msg/String 'data: What do you see?'"
    fi
}

# Handle Ctrl+C
trap 'echo ""; echo "ROS2 demo interrupted."; exit 0' INT

main "$@"
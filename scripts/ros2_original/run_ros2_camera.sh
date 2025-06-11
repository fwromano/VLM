#!/bin/bash

# VLM ROS2 Camera Launch Script
# Runs VLM with live camera feed

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}==================================="
echo "VLM ROS2 Camera Mode"
echo "===================================${NC}"

# Check if setup has been run
if [ ! -f "$HOME/.vlm_ros2_setup.bash" ]; then
    echo -e "${RED}Error: ROS2 setup not found!${NC}"
    echo "Please run ./setup_ros2.sh first"
    exit 1
fi

# Source ROS2 environment
source $HOME/.vlm_ros2_setup.bash

# Check if ROS2 is working
if ! command -v ros2 &> /dev/null; then
    echo -e "${RED}Error: ROS2 not found in environment!${NC}"
    exit 1
fi

# Check for camera device
CAMERA_DEVICE="/dev/video0"
if [ ! -e "$CAMERA_DEVICE" ]; then
    echo -e "${YELLOW}Warning: Camera not found at $CAMERA_DEVICE${NC}"
    echo "Available video devices:"
    ls /dev/video* 2>/dev/null || echo "No video devices found"
    echo ""
    read -p "Enter camera device path (or press Enter to use default): " custom_device
    if [ ! -z "$custom_device" ]; then
        CAMERA_DEVICE="$custom_device"
    fi
fi

# Parse command line arguments
CONTINUOUS="true"
RATE="0.5"
DEVICE="cuda"
PROMPT="What do you see in this image?"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-continuous)
            CONTINUOUS="false"
            shift
            ;;
        --rate)
            RATE="$2"
            shift 2
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-continuous    Disable continuous analysis"
            echo "  --rate <hz>        Analysis rate (default: 0.5 Hz)"
            echo "  --cpu              Use CPU instead of GPU"
            echo "  --prompt <text>    Default prompt for analysis"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Camera: $CAMERA_DEVICE"
echo "  Device: $DEVICE"
echo "  Continuous: $CONTINUOUS"
echo "  Rate: $RATE Hz"
echo "  Prompt: $PROMPT"
echo ""

# Check GPU availability
if [ "$DEVICE" == "cuda" ]; then
    if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        echo -e "${YELLOW}Warning: CUDA not available, falling back to CPU${NC}"
        DEVICE="cpu"
    fi
fi

# Launch ROS2 nodes
echo -e "${GREEN}Launching VLM ROS2 nodes...${NC}"

# Create temporary launch file with parameters
TEMP_LAUNCH="/tmp/vlm_camera_custom.launch.py"
cat > $TEMP_LAUNCH << EOF
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='camera',
            output='screen',
            parameters=[{
                'video_device': '$CAMERA_DEVICE',
                'image_width': 1280,
                'image_height': 720,
                'pixel_format': 'yuyv',
                'camera_frame_id': 'camera_optical_frame',
                'io_method': 'mmap',
                'framerate': 30.0,
            }],
            remappings=[
                ('image_raw', '/camera/image_raw'),
            ]
        ),
        Node(
            package='vlm_ros2',
            executable='vlm_node.py',
            name='vlm',
            output='screen',
            parameters=[{
                'model_name': 'InternVL3-2B-hf',
                'device': '$DEVICE',
                'image_topic': '/camera/image_raw',
                'continuous_analysis': $CONTINUOUS,
                'analysis_rate': $RATE,
                'default_prompt': '$PROMPT',
                'queue_size': 10,
            }]
        ),
    ])
EOF

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    rm -f $TEMP_LAUNCH
    # Kill any remaining nodes
    pkill -f "ros2.*vlm_node.py" 2>/dev/null
    pkill -f "ros2.*usb_cam" 2>/dev/null
}

trap cleanup EXIT

# Launch in background and monitor
ros2 launch $TEMP_LAUNCH &
LAUNCH_PID=$!

# Give nodes time to start
sleep 3

# Show how to interact
echo -e "\n${GREEN}==================================="
echo "VLM ROS2 is running!"
echo "===================================${NC}"
echo ""
echo "In another terminal, you can:"
echo "  - View topics: ros2 topic list"
echo "  - See analysis: ros2 topic echo /vlm/analysis"
echo "  - Call service: ros2 service call /vlm/analyze_image vlm_ros2/srv/AnalyzeImage \"{image: {}, prompt: 'What do you see?'}\""
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Monitor for analysis output
echo -e "${GREEN}Monitoring analysis output...${NC}"
ros2 topic echo /vlm/analysis vlm_ros2/msg/VLMAnalysis --field text &
ECHO_PID=$!

# Wait for launch process
wait $LAUNCH_PID
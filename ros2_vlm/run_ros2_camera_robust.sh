#!/bin/bash

# Robust ROS2 Camera Launch Script with Hard Failure Checks
# This script WILL NOT continue if prerequisites aren't met

set -e  # Exit immediately on any error
set -u  # Exit on undefined variables
set -o pipefail  # Exit on pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print and exit on error
fail_hard() {
    echo -e "${RED}FATAL ERROR: $1${NC}" >&2
    echo -e "${RED}Launch STOPPED. Fix the error above before continuing.${NC}" >&2
    exit 1
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

info() {
    echo -e "${GREEN}$1${NC}"
}

# Cleanup function
cleanup() {
    info "Shutting down ROS2 nodes..."
    # Kill any remaining VLM or camera nodes
    pkill -f "ros2.*vlm_node.py" 2>/dev/null || true
    pkill -f "ros2.*usb_cam" 2>/dev/null || true
    pkill -f "usb_cam_node" 2>/dev/null || true
    
    # Remove temporary files
    rm -f /tmp/vlm_camera_custom.launch.py
    
    echo "Cleanup completed."
}

# Set up cleanup on exit
trap cleanup EXIT

echo "=============================================="
echo "VLM ROS2 Camera (Robust Mode)"
echo "=============================================="
echo ""

# Step 1: Check if setup script exists and load it
if [ ! -f "$HOME/.vlm_ros2_setup.bash" ]; then
    fail_hard "VLM ROS2 setup script not found. Run ./install_ros2_robust.sh first."
fi

info "Step 1: Loading ROS2 environment..."
if ! source "$HOME/.vlm_ros2_setup.bash"; then
    fail_hard "Failed to load VLM ROS2 environment"
fi

# Step 2: Verify ROS2 is working
info "Step 2: Verifying ROS2 installation..."

if ! command -v ros2 &> /dev/null; then
    fail_hard "ros2 command not found. Install ROS2 first with ./install_ros2_robust.sh"
fi

if ! ros2 --help > /dev/null 2>&1; then
    fail_hard "ros2 command not working properly"
fi

if [ -z "${ROS_DISTRO:-}" ]; then
    fail_hard "ROS_DISTRO not set. ROS2 environment not properly loaded."
fi

success "ROS2 verified ($ROS_DISTRO)"

# Step 3: Check that workspace is built
ROS2_WS="$HOME/ros2_vlm_ws"
if [ ! -f "$ROS2_WS/install/setup.bash" ]; then
    fail_hard "ROS2 workspace not built. Run ./build_ros2_robust.sh first."
fi

if ! source "$ROS2_WS/install/setup.bash"; then
    fail_hard "Failed to source workspace setup"
fi

# Step 4: Verify VLM package is available
if ! ros2 pkg list | grep -q "vlm_ros2"; then
    fail_hard "vlm_ros2 package not found. Build failed or incomplete."
fi

if [ ! -f "$ROS2_WS/install/vlm_ros2/lib/vlm_ros2/vlm_node.py" ]; then
    fail_hard "vlm_node.py not found in install directory"
fi

success "VLM package verified"

# Step 5: Check camera
CAMERA_DEVICE="/dev/video0"
DEFAULT_DEVICE="$CAMERA_DEVICE"

if [ ! -e "$CAMERA_DEVICE" ]; then
    warning "Default camera not found at $CAMERA_DEVICE"
    
    # List available cameras
    AVAILABLE_CAMERAS=$(ls /dev/video* 2>/dev/null || echo "")
    if [ -z "$AVAILABLE_CAMERAS" ]; then
        fail_hard "No video devices found. Connect a camera and try again."
    fi
    
    echo "Available cameras:"
    for cam in $AVAILABLE_CAMERAS; do
        echo "  $cam"
    done
    
    # Use first available camera
    CAMERA_DEVICE=$(echo $AVAILABLE_CAMERAS | cut -d' ' -f1)
    warning "Using $CAMERA_DEVICE instead"
fi

# Test camera access
if ! python3 -c "import cv2; cap = cv2.VideoCapture('$CAMERA_DEVICE'); print('OK' if cap.isOpened() else 'FAIL'); cap.release()" | grep -q "OK"; then
    fail_hard "Cannot access camera at $CAMERA_DEVICE. Check permissions or device."
fi

success "Camera verified at $CAMERA_DEVICE"

# Step 6: Parse arguments with validation
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
            if [ -z "${2:-}" ] || ! [[ "$2" =~ ^[0-9]*\.?[0-9]+$ ]]; then
                fail_hard "Invalid rate value: ${2:-}. Must be a positive number."
            fi
            RATE="$2"
            shift 2
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --prompt)
            if [ -z "${2:-}" ]; then
                fail_hard "Empty prompt provided"
            fi
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
            fail_hard "Unknown option: $1"
            ;;
    esac
done

# Step 7: Validate rate
if ! python3 -c "float('$RATE')" 2>/dev/null; then
    fail_hard "Invalid rate: $RATE"
fi

if [ "$(python3 -c "print(float('$RATE') > 10)")" == "True" ]; then
    fail_hard "Rate too high: $RATE Hz. Maximum recommended: 10 Hz"
fi

# Step 8: Check GPU availability if requested
if [ "$DEVICE" == "cuda" ]; then
    if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        warning "CUDA not available, falling back to CPU"
        DEVICE="cpu"
    else
        success "GPU acceleration available"
    fi
fi

# Step 9: Check required ROS2 packages
REQUIRED_PACKAGES=("usb_cam" "cv_bridge")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! find /opt/ros/$ROS_DISTRO -name "*$pkg*" | grep -q "$pkg"; then
        fail_hard "Required ROS2 package not found: $pkg. Install with: sudo apt install ros-$ROS_DISTRO-$pkg"
    fi
done

success "Required packages verified"

# Step 10: Display configuration
echo ""
info "Configuration:"
echo "  Camera: $CAMERA_DEVICE"
echo "  Device: $DEVICE"
echo "  Continuous: $CONTINUOUS"
echo "  Rate: $RATE Hz"
echo "  Prompt: $PROMPT"
echo "  ROS2: $ROS_DISTRO"
echo ""

# Step 11: Create launch file
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

# Step 12: Validate launch file
if ! python3 -c "exec(open('$TEMP_LAUNCH').read())" 2>/dev/null; then
    fail_hard "Generated launch file is invalid"
fi

success "Launch configuration created"

# Step 13: Launch nodes
info "Launching VLM ROS2 nodes..."
echo "Press Ctrl+C to stop"
echo ""

# Launch and monitor
ros2 launch $TEMP_LAUNCH

# If we get here, launch was terminated normally
success "VLM ROS2 camera session completed"
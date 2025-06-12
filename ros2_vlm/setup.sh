#!/bin/bash

echo "VLM ROS2 Integration - Complete Setup"
echo "====================================="

# Check Ubuntu version
if ! lsb_release -a 2>/dev/null | grep -q "Ubuntu"; then
    echo "Error: This script is designed for Ubuntu."
    exit 1
fi

# Function to check if ROS2 is installed
check_ros2() {
    if [ -d "/opt/ros/jazzy" ]; then
        return 0
    else
        return 1
    fi
}

# Step 1: Install ROS2 if needed
if ! check_ros2; then
    echo "ROS2 Jazzy not found. Installing..."
    ./install_ros2_robust.sh
else
    echo "ROS2 Jazzy already installed."
fi

# Step 2: Source ROS2
source /opt/ros/jazzy/setup.bash

# Step 3: Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Step 4: Install ROS2 dependencies
echo ""
echo "Installing ROS2 dependencies..."
sudo apt update
sudo apt install -y \
    ros-jazzy-usb-cam \
    ros-jazzy-image-view \
    ros-jazzy-cv-bridge \
    python3-colcon-common-extensions

# Step 5: Build the package
echo ""
echo "Building VLM ROS2 package..."
./build_ros2_robust.sh

# Step 6: Validate setup
echo ""
echo "Validating installation..."
./validate_ros2_setup.sh

echo ""
echo "Setup complete!"
echo ""
echo "To run with camera:"
echo "  ./run_ros2_camera_robust.sh"
echo ""
echo "To run demos:"
echo "  cd demos && ./run_vlm_ros2_demo.sh"
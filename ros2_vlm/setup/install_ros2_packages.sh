#!/bin/bash

echo "Installing Additional ROS2 Packages"
echo "==================================="

if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo:"
    echo "sudo $0"
    exit 1
fi

echo "Installing ROS2 packages for VLM integration..."

# Core packages
apt install -y \
    ros-jazzy-demo-nodes-py \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-usb-cam \
    ros-jazzy-image-view

# Development tools
apt install -y \
    ros-jazzy-rqt-image-view \
    ros-jazzy-rqt-graph \
    python3-colcon-common-extensions

echo ""
echo "✓ ROS2 packages installed!"
echo ""
echo "Now test with:"
echo "1. Remove conda from PATH (as shown in previous script)"
echo "2. source /opt/ros/jazzy/setup.bash"
echo "3. ros2 run demo_nodes_py talker"
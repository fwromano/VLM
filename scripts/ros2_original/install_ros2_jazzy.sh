#!/bin/bash

# Quick ROS2 Jazzy Installation for Ubuntu 24.04

echo "====================================="
echo "ROS2 Jazzy Quick Install"
echo "====================================="
echo ""
echo "This will install ROS2 Jazzy for Ubuntu 24.04"
echo "You'll need to enter your sudo password."
echo ""

# Add ROS2 repository if not already added
if [ ! -f "/etc/apt/sources.list.d/ros2.list" ]; then
    echo "Adding ROS2 repository..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository universe
    
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
fi

# Update and install
echo "Installing ROS2 Jazzy..."
sudo apt update
sudo apt install -y \
    ros-jazzy-desktop \
    ros-jazzy-ros-base \
    python3-colcon-common-extensions \
    ros-jazzy-usb-cam \
    ros-jazzy-cv-bridge \
    python3-rosdep2

# Initialize rosdep if needed
if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
    sudo rosdep init
fi
rosdep update

echo ""
echo "ROS2 Jazzy installed!"
echo "Now run:"
echo "  source ~/.bashrc"
echo "  vlm_ros2"
echo "  ./build_ros2.sh"
echo "  ./run_ros2_camera.sh"
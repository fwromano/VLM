#!/bin/bash

# Simple ROS2 build script for VLM
# Rebuilds the VLM ROS2 package

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}==================================="
echo "VLM ROS2 Build Script"
echo "===================================${NC}"

# Check if setup has been run
if [ ! -f "$HOME/.vlm_ros2_setup.bash" ]; then
    echo -e "${RED}Error: ROS2 setup not found!${NC}"
    echo "Please run ./setup_ros2.sh first"
    exit 1
fi

# Source ROS2 environment
source $HOME/.vlm_ros2_setup.bash

# Check workspace exists
ROS2_WS="$HOME/ros2_vlm_ws"
if [ ! -d "$ROS2_WS" ]; then
    echo -e "${RED}Error: ROS2 workspace not found at $ROS2_WS${NC}"
    echo "Please run ./setup_ros2.sh first"
    exit 1
fi

# Check if source has been updated
if [ -d "ros2_vlm/src/vlm_ros2" ]; then
    echo -e "${GREEN}Updating VLM ROS2 package in workspace...${NC}"
    cp -r ros2_vlm/src/vlm_ros2/* "$ROS2_WS/src/vlm_ros2/"
fi

# Navigate to workspace
cd "$ROS2_WS"

# Clean option
if [[ "$1" == "--clean" ]]; then
    echo -e "${YELLOW}Performing clean build...${NC}"
    rm -rf build install log
fi

# Build
echo -e "${GREEN}Building VLM ROS2 package...${NC}"
colcon build --packages-select vlm_ros2 --symlink-install

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
    
    # Source the workspace
    source install/setup.bash
    
    echo ""
    echo "Package built successfully!"
    echo "You can now run:"
    echo "  ./run_ros2_camera.sh    - For live camera"
    echo "  ./run_ros2_bag.sh <bag> - For bag files"
    echo "  ./run_ros2_interactive.sh - For interactive testing"
else
    echo -e "${RED}Build failed!${NC}"
    echo "Check the errors above and try again"
    exit 1
fi
#!/bin/bash

# Manual ROS2 Setup Guide for VLM

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}==================================="
echo "VLM ROS2 Manual Setup Guide"
echo "===================================${NC}"

echo -e "\n${YELLOW}This script will guide you through the setup process.${NC}"
echo "You'll need to run some commands with sudo."

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
echo -e "\n${GREEN}Detected Ubuntu $UBUNTU_VERSION${NC}"

# Set ROS2 distribution
if [[ "$UBUNTU_VERSION" == "24.04" ]]; then
    ROS_DISTRO="jazzy"
    echo -e "${GREEN}Using ROS2 Jazzy for Ubuntu 24.04${NC}"
elif [[ "$UBUNTU_VERSION" == "22.04" ]]; then
    ROS_DISTRO="humble"
elif [[ "$UBUNTU_VERSION" == "20.04" ]]; then
    ROS_DISTRO="foxy"
else
    ROS_DISTRO="humble"
    echo -e "${YELLOW}Unsupported Ubuntu version. Defaulting to Humble.${NC}"
fi

echo -e "\n${BLUE}Step 1: Install system dependencies${NC}"
echo "Run this command:"
echo -e "${GREEN}sudo apt update && sudo apt install -y cmake build-essential python3-pip python3-venv git wget curl${NC}"
read -p "Press Enter after running the command..."

echo -e "\n${BLUE}Step 2: Check if ROS2 is installed${NC}"
if ! command -v ros2 &> /dev/null; then
    echo -e "${YELLOW}ROS2 not found. Let's install it.${NC}"
    echo ""
    echo "Run these commands:"
    echo -e "${GREEN}# Add ROS2 repository${NC}"
    echo "sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg"
    echo "echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \$(lsb_release -cs) main\" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null"
    echo ""
    echo -e "${GREEN}# Install ROS2${NC}"
    echo "sudo apt update"
    echo "sudo apt install -y ros-$ROS_DISTRO-desktop python3-colcon-common-extensions"
    echo ""
    read -p "Press Enter after running these commands..."
else
    echo -e "${GREEN}ROS2 is already installed!${NC}"
fi

echo -e "\n${BLUE}Step 3: Install ROS2 packages for VLM${NC}"
echo "Run this command:"
echo -e "${GREEN}sudo apt install -y ros-$ROS_DISTRO-usb-cam ros-$ROS_DISTRO-image-view ros-$ROS_DISTRO-cv-bridge${NC}"
read -p "Press Enter after running the command..."

echo -e "\n${BLUE}Step 4: Create ROS2 workspace${NC}"
ROS2_WS="$HOME/ros2_vlm_ws"
echo "Creating workspace at: $ROS2_WS"
mkdir -p "$ROS2_WS/src"

# Copy VLM package
if [ -d "ros2_vlm/src/vlm_ros2" ]; then
    cp -r ros2_vlm/src/vlm_ros2 "$ROS2_WS/src/"
    echo -e "${GREEN}Copied VLM ROS2 package to workspace${NC}"
else
    echo -e "${YELLOW}Warning: VLM ROS2 package not found${NC}"
fi

echo -e "\n${BLUE}Step 5: Build the workspace${NC}"
echo "Run these commands:"
echo -e "${GREEN}cd $ROS2_WS${NC}"
echo -e "${GREEN}source /opt/ros/$ROS_DISTRO/setup.bash${NC}"
echo -e "${GREEN}colcon build --packages-select vlm_ros2${NC}"
read -p "Press Enter after running these commands..."

echo -e "\n${BLUE}Step 6: Create environment setup${NC}"
cat > "$HOME/.vlm_ros2_setup.bash" << EOF
#!/bin/bash
# VLM ROS2 Environment Setup

# Source ROS2
source /opt/ros/$ROS_DISTRO/setup.bash

# Source workspace
source $ROS2_WS/install/setup.bash

# Set ROS2 domain ID (optional, change if needed)
export ROS_DOMAIN_ID=0

# Activate conda environment if available
if command -v conda &> /dev/null && conda env list | grep -q "^vlm "; then
    eval "\$(conda shell.bash hook)"
    conda activate vlm
fi

echo "VLM ROS2 environment loaded!"
echo "Workspace: $ROS2_WS"
echo "ROS2 Distribution: $ROS_DISTRO"
EOF

chmod +x "$HOME/.vlm_ros2_setup.bash"

echo -e "\n${BLUE}Step 7: Add alias to bashrc${NC}"
if ! grep -q "vlm_ros2_setup" "$HOME/.bashrc"; then
    echo "" >> "$HOME/.bashrc"
    echo "# VLM ROS2 Setup (added by setup_ros2.sh)" >> "$HOME/.bashrc"
    echo "alias vlm_ros2='source $HOME/.vlm_ros2_setup.bash'" >> "$HOME/.bashrc"
    echo -e "${GREEN}Added 'vlm_ros2' alias to bashrc${NC}"
else
    echo -e "${GREEN}Alias already exists${NC}"
fi

echo -e "\n${GREEN}==================================="
echo "Setup Complete!"
echo "===================================${NC}"
echo ""
echo "Now run:"
echo "1. source ~/.bashrc"
echo "2. vlm_ros2"
echo "3. ./run_ros2_camera.sh"
echo ""
echo "If you had any errors, please run the failed commands manually."
#!/bin/bash

# VLM ROS2 Setup Script
# This script installs all dependencies needed for ROS2 integration

set -e  # Exit on error

echo "==================================="
echo "VLM ROS2 Integration Setup Script"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running on Ubuntu
if ! grep -q "Ubuntu" /etc/os-release; then
    print_error "This script is designed for Ubuntu. Your system might need manual setup."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
print_status "Detected Ubuntu $UBUNTU_VERSION"

# Set ROS2 distribution based on Ubuntu version
if [[ "$UBUNTU_VERSION" == "22.04" ]]; then
    ROS_DISTRO="humble"
elif [[ "$UBUNTU_VERSION" == "20.04" ]]; then
    ROS_DISTRO="foxy"
else
    print_warning "Unsupported Ubuntu version. Defaulting to Humble."
    ROS_DISTRO="humble"
fi

print_status "Using ROS2 $ROS_DISTRO"

# Step 1: Install system dependencies
echo -e "\n${GREEN}Step 1: Installing system dependencies...${NC}"
sudo apt update
sudo apt install -y \
    cmake \
    build-essential \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    software-properties-common \
    lsb-release \
    gnupg2

# Step 2: Check if ROS2 is installed
echo -e "\n${GREEN}Step 2: Checking ROS2 installation...${NC}"
if ! command -v ros2 &> /dev/null; then
    print_warning "ROS2 not found. Installing ROS2 $ROS_DISTRO..."
    
    # Add ROS2 repository
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    
    # Install ROS2
    sudo apt update
    sudo apt install -y ros-$ROS_DISTRO-desktop
    sudo apt install -y python3-rosdep2
    
    # Initialize rosdep
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
        sudo rosdep init
    fi
    rosdep update
    
    print_status "ROS2 $ROS_DISTRO installed successfully"
else
    print_status "ROS2 is already installed"
fi

# Step 3: Install ROS2 development tools
echo -e "\n${GREEN}Step 3: Installing ROS2 development tools...${NC}"
sudo apt install -y \
    ros-$ROS_DISTRO-ros-base \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-argcomplete

# Step 4: Install ROS2 packages for VLM
echo -e "\n${GREEN}Step 4: Installing ROS2 packages for camera and visualization...${NC}"
sudo apt install -y \
    ros-$ROS_DISTRO-usb-cam \
    ros-$ROS_DISTRO-image-view \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-image-transport \
    ros-$ROS_DISTRO-camera-info-manager \
    ros-$ROS_DISTRO-diagnostic-msgs

# Step 5: Setup Python environment
echo -e "\n${GREEN}Step 5: Setting up Python environment...${NC}"

# Check if conda is available
if command -v conda &> /dev/null; then
    print_status "Conda detected. Checking for vlm environment..."
    
    # Check if vlm environment exists
    if conda env list | grep -q "^vlm "; then
        print_status "VLM conda environment exists"
        
        # Activate and install ROS2 Python packages
        eval "$(conda shell.bash hook)"
        conda activate vlm
        
        # Install ROS2 Python dependencies in conda env
        pip install --upgrade pip
        pip install \
            rclpy \
            cv_bridge \
            sensor_msgs \
            std_msgs \
            geometry_msgs \
            diagnostic_msgs
            
        print_status "ROS2 Python packages installed in vlm environment"
    else
        print_warning "VLM conda environment not found. Run setup.sh first!"
    fi
else
    print_warning "Conda not found. Installing Python packages globally..."
    pip3 install --user \
        rclpy \
        cv_bridge \
        sensor_msgs \
        std_msgs \
        geometry_msgs \
        diagnostic_msgs
fi

# Step 6: Create ROS2 workspace
echo -e "\n${GREEN}Step 6: Setting up ROS2 workspace...${NC}"
ROS2_WS="$HOME/ros2_vlm_ws"

if [ ! -d "$ROS2_WS" ]; then
    mkdir -p "$ROS2_WS/src"
    print_status "Created ROS2 workspace at $ROS2_WS"
else
    print_status "ROS2 workspace already exists at $ROS2_WS"
fi

# Step 7: Copy VLM ROS2 package to workspace
echo -e "\n${GREEN}Step 7: Setting up VLM ROS2 package...${NC}"
if [ -d "ros2_vlm/src/vlm_ros2" ]; then
    cp -r ros2_vlm/src/vlm_ros2 "$ROS2_WS/src/"
    print_status "Copied VLM ROS2 package to workspace"
else
    print_error "VLM ROS2 package not found in current directory"
    exit 1
fi

# Step 8: Build the workspace
echo -e "\n${GREEN}Step 8: Building ROS2 workspace...${NC}"
cd "$ROS2_WS"

# Source ROS2
source /opt/ros/$ROS_DISTRO/setup.bash

# Build
colcon build --packages-select vlm_ros2
if [ $? -eq 0 ]; then
    print_status "Build completed successfully!"
else
    print_error "Build failed. Please check the errors above."
    exit 1
fi

# Step 9: Create convenience scripts
echo -e "\n${GREEN}Step 9: Creating convenience scripts...${NC}"

# Create source script
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
print_status "Created environment setup script"

# Step 10: Setup bashrc
echo -e "\n${GREEN}Step 10: Updating bashrc...${NC}"
if ! grep -q "vlm_ros2_setup" "$HOME/.bashrc"; then
    echo "" >> "$HOME/.bashrc"
    echo "# VLM ROS2 Setup (added by setup_ros2.sh)" >> "$HOME/.bashrc"
    echo "alias vlm_ros2='source $HOME/.vlm_ros2_setup.bash'" >> "$HOME/.bashrc"
    print_status "Added 'vlm_ros2' alias to bashrc"
else
    print_status "Bashrc already configured"
fi

# Final instructions
echo -e "\n${GREEN}==================================="
echo "Setup Complete!"
echo "===================================${NC}"
echo ""
echo "To use VLM with ROS2:"
echo "1. Open a new terminal or run: source ~/.bashrc"
echo "2. Type: vlm_ros2"
echo "3. Run one of the launch scripts in this directory"
echo ""
echo "Quick test:"
echo "  vlm_ros2"
echo "  ./run_ros2_camera.sh"
echo ""
print_warning "Note: Make sure your camera is connected at /dev/video0"

# Create a simple test script
cat > test_ros2_setup.sh << 'EOF'
#!/bin/bash
source $HOME/.vlm_ros2_setup.bash
echo "Testing ROS2 setup..."
ros2 topic list
echo "If you see topic list above, ROS2 is working!"
EOF
chmod +x test_ros2_setup.sh

print_status "Created test_ros2_setup.sh - run this to verify installation"
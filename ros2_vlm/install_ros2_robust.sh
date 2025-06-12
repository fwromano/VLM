#!/bin/bash

# Robust ROS2 Installation with Hard Failure Checks
# This script WILL NOT continue if any step fails

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
    echo -e "${RED}Installation STOPPED. Do not continue until this is fixed.${NC}" >&2
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

# Function to check if command succeeded
check_command() {
    if ! command -v "$1" &> /dev/null; then
        fail_hard "Command '$1' not found after installation"
    fi
}

# Function to check if package is installed
check_package() {
    if ! dpkg -l | grep -q "^ii.*$1"; then
        fail_hard "Package '$1' not properly installed"
    fi
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        fail_hard "Required file not found: $1"
    fi
}

echo "=============================================="
echo "ROS2 Robust Installation with Hard Failures"
echo "=============================================="
echo ""

# Check if running as root (should not be)
if [ "$EUID" -eq 0 ]; then
    fail_hard "Do not run this script as root (don't use sudo)"
fi

# Check Ubuntu version
if ! command -v lsb_release &> /dev/null; then
    fail_hard "lsb_release not found. Cannot determine Ubuntu version."
fi

UBUNTU_VERSION=$(lsb_release -rs)
info "Detected Ubuntu $UBUNTU_VERSION"

# Determine correct ROS2 distribution
case $UBUNTU_VERSION in
    "24.04")
        ROS_DISTRO="jazzy"
        warning "Ubuntu 24.04 detected. ROS2 Jazzy support is still experimental."
        ;;
    "22.04")
        ROS_DISTRO="humble"
        ;;
    "20.04")
        ROS_DISTRO="foxy"
        warning "Ubuntu 20.04 is older. Consider upgrading."
        ;;
    *)
        fail_hard "Unsupported Ubuntu version: $UBUNTU_VERSION. Supported: 20.04, 22.04, 24.04"
        ;;
esac

info "Will install ROS2 $ROS_DISTRO"

# Step 1: Update package lists
info "Step 1: Updating package lists..."
if ! sudo apt update; then
    fail_hard "Failed to update package lists"
fi
success "Package lists updated"

# Step 2: Install prerequisites
info "Step 2: Installing prerequisites..."
PREREQ_PACKAGES="software-properties-common curl gnupg2 lsb-release"
if ! sudo apt install -y $PREREQ_PACKAGES; then
    fail_hard "Failed to install prerequisites: $PREREQ_PACKAGES"
fi

# Verify prerequisites
for pkg in curl gnupg; do
    check_command $pkg
done
success "Prerequisites installed"

# Step 3: Add ROS2 repository
info "Step 3: Adding ROS2 repository..."

# Check if repository already exists
if [ ! -f "/etc/apt/sources.list.d/ros2.list" ]; then
    # Add ROS2 GPG key
    if ! sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg; then
        fail_hard "Failed to download ROS2 GPG key"
    fi
    
    # Verify key was downloaded
    check_file "/usr/share/keyrings/ros-archive-keyring.gpg"
    
    # Add repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    
    # Verify repository file was created
    check_file "/etc/apt/sources.list.d/ros2.list"
    
    success "ROS2 repository added"
else
    success "ROS2 repository already exists"
fi

# Step 4: Update after adding repository
info "Step 4: Updating package lists with ROS2 repository..."
if ! sudo apt update; then
    fail_hard "Failed to update package lists after adding ROS2 repository"
fi

# Check if ROS2 packages are available
if ! apt-cache search ros-$ROS_DISTRO-desktop | grep -q "ros-$ROS_DISTRO-desktop"; then
    fail_hard "ROS2 $ROS_DISTRO packages not found in repositories. Check your Ubuntu version and network connection."
fi
success "ROS2 packages found in repositories"

# Step 5: Install ROS2
info "Step 5: Installing ROS2 $ROS_DISTRO..."

# Core packages that must be installed
CORE_PACKAGES="ros-$ROS_DISTRO-desktop python3-colcon-common-extensions"

if ! sudo apt install -y $CORE_PACKAGES; then
    fail_hard "Failed to install core ROS2 packages: $CORE_PACKAGES"
fi

# Verify ROS2 installation
check_file "/opt/ros/$ROS_DISTRO/setup.bash"
success "ROS2 core installed"

# Step 6: Install rosdep
info "Step 6: Installing rosdep..."

# For Ubuntu 24.04, the package name might be different
if [ "$UBUNTU_VERSION" = "24.04" ]; then
    # Try different package names
    if sudo apt install -y python3-rosdep; then
        success "rosdep installed (python3-rosdep)"
    elif sudo apt install -y python3-rosdep2; then
        success "rosdep installed (python3-rosdep2)"
    else
        warning "Could not install rosdep package. Trying pip..."
        if ! pip3 install --user rosdep; then
            fail_hard "Failed to install rosdep via apt or pip"
        fi
        success "rosdep installed via pip"
    fi
else
    if ! sudo apt install -y python3-rosdep2; then
        fail_hard "Failed to install python3-rosdep2"
    fi
    success "rosdep2 installed"
fi

# Step 7: Initialize rosdep
info "Step 7: Initializing rosdep..."

# Source ROS2 first
source /opt/ros/$ROS_DISTRO/setup.bash

# Check if rosdep command exists now
if ! command -v rosdep &> /dev/null; then
    fail_hard "rosdep command not found after installation"
fi

# Initialize rosdep if not already done
if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
    if ! sudo rosdep init; then
        fail_hard "Failed to initialize rosdep"
    fi
    success "rosdep initialized"
else
    success "rosdep already initialized"
fi

# Update rosdep
if ! rosdep update; then
    fail_hard "Failed to update rosdep"
fi
success "rosdep updated"

# Step 8: Install VLM-specific packages
info "Step 8: Installing VLM-specific ROS2 packages..."

VLM_PACKAGES="ros-$ROS_DISTRO-usb-cam ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-image-view"

if ! sudo apt install -y $VLM_PACKAGES; then
    fail_hard "Failed to install VLM-specific packages: $VLM_PACKAGES"
fi

# Verify key packages
for pkg in usb-cam cv-bridge; do
    if ! find /opt/ros/$ROS_DISTRO -name "*$pkg*" | grep -q "$pkg"; then
        fail_hard "Package $pkg not found in ROS2 installation"
    fi
done
success "VLM-specific packages installed"

# Step 9: Verify ROS2 works
info "Step 9: Verifying ROS2 installation..."

# Test that ros2 command works
source /opt/ros/$ROS_DISTRO/setup.bash
if ! ros2 --help > /dev/null 2>&1; then
    fail_hard "ros2 command not working after installation"
fi

check_command ros2
success "ROS2 command verified"

# Test that we can list packages
if ! ros2 pkg list | grep -q "std_msgs"; then
    fail_hard "ROS2 packages not properly installed (std_msgs not found)"
fi
success "ROS2 packages verified"

# Step 10: Update VLM setup script
info "Step 10: Updating VLM ROS2 setup script..."

cat > "$HOME/.vlm_ros2_setup.bash" << EOF
#!/bin/bash
# VLM ROS2 Environment Setup (Generated by install_ros2_robust.sh)

# Source ROS2
if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    source /opt/ros/$ROS_DISTRO/setup.bash
    export ROS_DISTRO="$ROS_DISTRO"
else
    echo "ERROR: ROS2 $ROS_DISTRO not found at /opt/ros/$ROS_DISTRO/"
    exit 1
fi

# Source workspace if it exists
if [ -f "\$HOME/ros2_vlm_ws/install/setup.bash" ]; then
    source \$HOME/ros2_vlm_ws/install/setup.bash
fi

# Set ROS2 domain ID
export ROS_DOMAIN_ID=0

# Activate conda environment if available
if command -v conda &> /dev/null && conda env list | grep -q "^vlm "; then
    eval "\$(conda shell.bash hook)"
    conda activate vlm
fi

echo "✓ VLM ROS2 environment loaded successfully!"
echo "  ROS2 Distribution: $ROS_DISTRO"
echo "  Workspace: \$HOME/ros2_vlm_ws"
EOF

chmod +x "$HOME/.vlm_ros2_setup.bash"
success "VLM setup script updated"

# Final verification
info "Final verification..."
source "$HOME/.vlm_ros2_setup.bash"
if [ "$?" -ne 0 ]; then
    fail_hard "VLM setup script failed to run"
fi

echo ""
echo -e "${GREEN}=============================================="
echo "ROS2 Installation Completed Successfully!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. source ~/.bashrc"
echo "2. vlm_ros2"
echo "3. ./build_ros2.sh"
echo "4. ./run_ros2_camera.sh"
echo ""
echo -e "ROS2 Distribution: $ROS_DISTRO"
echo -e "Installation verified: ✓"
echo -e "${NC}"
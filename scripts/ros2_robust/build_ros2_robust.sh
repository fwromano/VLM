#!/bin/bash

# Robust ROS2 build script with hard failure checks
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
    echo -e "${RED}Build STOPPED. Fix the error above before continuing.${NC}" >&2
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

echo "=============================================="
echo "VLM ROS2 Robust Build Script"
echo "=============================================="
echo ""

# Step 1: Check if VLM setup script exists
if [ ! -f "$HOME/.vlm_ros2_setup.bash" ]; then
    fail_hard "VLM ROS2 setup script not found. Run ./install_ros2_robust.sh first."
fi

# Step 2: Source the setup script and validate
info "Step 1: Loading ROS2 environment..."
if ! source "$HOME/.vlm_ros2_setup.bash"; then
    fail_hard "VLM ROS2 setup script failed to load"
fi

# Step 3: Verify ROS2 is actually working
info "Step 2: Verifying ROS2 installation..."

if ! command -v ros2 &> /dev/null; then
    fail_hard "ros2 command not found. ROS2 installation is broken or not installed."
fi

if ! ros2 --help > /dev/null 2>&1; then
    fail_hard "ros2 command not working properly"
fi

if [ -z "${ROS_DISTRO:-}" ]; then
    fail_hard "ROS_DISTRO environment variable not set"
fi

# Check that core ROS2 files exist
if [ ! -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    fail_hard "ROS2 installation incomplete: /opt/ros/$ROS_DISTRO/setup.bash not found"
fi

success "ROS2 environment verified ($ROS_DISTRO)"

# Step 4: Check workspace exists
ROS2_WS="$HOME/ros2_vlm_ws"
if [ ! -d "$ROS2_WS" ]; then
    fail_hard "ROS2 workspace not found at $ROS2_WS. Run setup script first."
fi

# Step 5: Check VLM package exists
if [ ! -d "$ROS2_WS/src/vlm_ros2" ]; then
    fail_hard "VLM ROS2 package not found at $ROS2_WS/src/vlm_ros2"
fi

# Step 6: Verify essential files exist
REQUIRED_FILES=(
    "$ROS2_WS/src/vlm_ros2/package.xml"
    "$ROS2_WS/src/vlm_ros2/CMakeLists.txt"
    "$ROS2_WS/src/vlm_ros2/scripts/vlm_node.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        fail_hard "Required file missing: $file"
    fi
done

success "VLM package structure verified"

# Step 7: Check for colcon
if ! command -v colcon &> /dev/null; then
    fail_hard "colcon build tool not found. Install with: sudo apt install python3-colcon-common-extensions"
fi

# Step 8: Update package if source exists
if [ -d "ros2_vlm/src/vlm_ros2" ]; then
    info "Step 3: Updating VLM ROS2 package..."
    if ! cp -r ros2_vlm/src/vlm_ros2/* "$ROS2_WS/src/vlm_ros2/"; then
        fail_hard "Failed to update VLM package"
    fi
    success "Package updated"
fi

# Step 9: Navigate to workspace
cd "$ROS2_WS" || fail_hard "Cannot change to workspace directory: $ROS2_WS"

# Step 10: Clean build if requested
if [[ "${1:-}" == "--clean" ]]; then
    info "Step 4: Performing clean build..."
    rm -rf build install log
    success "Build directories cleaned"
fi

# Step 11: Check dependencies
info "Step 5: Checking dependencies..."
if ! rosdep check --from-paths src --ignore-src -r; then
    warning "Some dependencies missing. Attempting to install..."
    if ! rosdep install --from-paths src --ignore-src -r -y; then
        fail_hard "Failed to install dependencies"
    fi
fi
success "Dependencies verified"

# Step 12: Build
info "Step 6: Building VLM ROS2 package..."
if ! colcon build --packages-select vlm_ros2 --symlink-install; then
    fail_hard "colcon build failed. Check the error messages above."
fi

success "Build completed successfully"

# Step 13: Verify build outputs
if [ ! -f "$ROS2_WS/install/vlm_ros2/lib/vlm_ros2/vlm_node.py" ]; then
    fail_hard "Build completed but vlm_node.py not found in install directory"
fi

if [ ! -f "$ROS2_WS/install/setup.bash" ]; then
    fail_hard "Build completed but setup.bash not generated"
fi

success "Build outputs verified"

# Step 14: Test the installation
info "Step 7: Testing installation..."
source install/setup.bash

if ! ros2 pkg list | grep -q "vlm_ros2"; then
    fail_hard "vlm_ros2 package not found in ROS2 package list after build"
fi

if ! ros2 interface list | grep -q "vlm_ros2"; then
    fail_hard "vlm_ros2 interfaces not found after build"
fi

success "Installation test passed"

echo ""
echo -e "${GREEN}=============================================="
echo "Build Completed Successfully!"
echo "=============================================="
echo ""
echo "You can now run:"
echo "  ./run_ros2_camera.sh"
echo "  ./run_ros2_bag.sh <bagfile>"
echo "  ./run_ros2_interactive.sh"
echo ""
echo "Package location: $ROS2_WS/src/vlm_ros2"
echo "Install location: $ROS2_WS/install/vlm_ros2"
echo -e "${NC}"
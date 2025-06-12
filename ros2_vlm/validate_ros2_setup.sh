#!/bin/bash

# ROS2 VLM Setup Validation Script
# Quickly check if everything is properly installed and configured

set -e  # Exit immediately on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

fail() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++))
}

pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++))
}

warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((WARNINGS++))
}

info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

PASSED=0
FAILED=0
WARNINGS=0

echo "=============================================="
echo "VLM ROS2 Setup Validation"
echo "=============================================="
echo ""

# Test 1: Check setup script exists
info "Checking setup script..."
if [ -f "$HOME/.vlm_ros2_setup.bash" ]; then
    pass "VLM setup script found"
else
    fail "VLM setup script missing - run ./install_ros2_robust.sh"
fi

# Test 2: Load environment
info "Loading ROS2 environment..."
if source "$HOME/.vlm_ros2_setup.bash" 2>/dev/null; then
    pass "ROS2 environment loaded"
else
    fail "Failed to load ROS2 environment"
fi

# Test 3: Check ROS2 command
info "Checking ROS2 command..."
if command -v ros2 &> /dev/null; then
    pass "ros2 command found"
    
    # Test ros2 works
    if ros2 --help > /dev/null 2>&1; then
        pass "ros2 command working"
    else
        fail "ros2 command not working"
    fi
else
    fail "ros2 command not found"
fi

# Test 4: Check ROS_DISTRO
info "Checking ROS distribution..."
if [ -n "${ROS_DISTRO:-}" ]; then
    pass "ROS_DISTRO set to: $ROS_DISTRO"
else
    fail "ROS_DISTRO not set"
fi

# Test 5: Check workspace
info "Checking workspace..."
ROS2_WS="$HOME/ros2_vlm_ws"
if [ -d "$ROS2_WS" ]; then
    pass "ROS2 workspace exists"
    
    # Check if built
    if [ -f "$ROS2_WS/install/setup.bash" ]; then
        pass "Workspace is built"
    else
        fail "Workspace not built - run ./build_ros2_robust.sh"
    fi
else
    fail "ROS2 workspace missing"
fi

# Test 6: Check VLM package
info "Checking VLM package..."
if [ -d "$ROS2_WS/src/vlm_ros2" ]; then
    pass "VLM source package found"
else
    fail "VLM source package missing"
fi

if [ -f "$ROS2_WS/install/vlm_ros2/lib/vlm_ros2/vlm_node.py" ]; then
    pass "VLM node installed"
else
    fail "VLM node not installed"
fi

# Test 7: Test package discovery
info "Testing package discovery..."
if ros2 pkg list | grep -q "vlm_ros2" 2>/dev/null; then
    pass "vlm_ros2 package discoverable"
else
    fail "vlm_ros2 package not found by ROS2"
fi

# Test 8: Check required ROS packages
info "Checking required ROS2 packages..."
REQUIRED_PACKAGES=("usb_cam" "cv_bridge" "std_msgs" "sensor_msgs")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if find /opt/ros/${ROS_DISTRO:-humble} -name "*$pkg*" 2>/dev/null | grep -q "$pkg"; then
        pass "$pkg package found"
    else
        fail "$pkg package missing"
    fi
done

# Test 9: Check camera
info "Checking camera..."
if ls /dev/video* &> /dev/null; then
    CAMERAS=$(ls /dev/video* 2>/dev/null)
    pass "Video devices found: $CAMERAS"
    
    # Test first camera
    FIRST_CAM=$(echo $CAMERAS | cut -d' ' -f1)
    if python3 -c "import cv2; cap = cv2.VideoCapture('$FIRST_CAM'); print('OK' if cap.isOpened() else 'FAIL'); cap.release()" 2>/dev/null | grep -q "OK"; then
        pass "Camera $FIRST_CAM accessible"
    else
        warn "Camera $FIRST_CAM not accessible"
    fi
else
    warn "No video devices found"
fi

# Test 10: Check conda environment
info "Checking conda environment..."
if command -v conda &> /dev/null; then
    pass "conda command found"
    
    if conda env list | grep -q "^vlm "; then
        pass "VLM conda environment exists"
    else
        warn "VLM conda environment missing - run ./setup.sh"
    fi
else
    warn "conda not found"
fi

# Test 11: Check Python dependencies
info "Checking Python dependencies..."
PYTHON_DEPS=("torch" "transformers" "cv2" "PIL")
for dep in "${PYTHON_DEPS[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        pass "Python package: $dep"
    else
        fail "Python package missing: $dep"
    fi
done

# Test 12: Check GPU
info "Checking GPU support..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    pass "CUDA GPU available"
    
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    pass "GPU count: $GPU_COUNT"
else
    warn "CUDA GPU not available - will use CPU"
fi

# Test 13: Check colcon
info "Checking build tools..."
if command -v colcon &> /dev/null; then
    pass "colcon build tool found"
else
    fail "colcon missing"
fi

if command -v rosdep &> /dev/null; then
    pass "rosdep found"
else
    fail "rosdep missing"
fi

# Summary
echo ""
echo "=============================================="
echo "Validation Summary"
echo "=============================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ System is ready for VLM ROS2!${NC}"
    echo ""
    echo "You can now run:"
    echo "  ./run_ros2_camera_robust.sh"
    echo ""
    exit 0
elif [ $FAILED -le 2 ] && [ $PASSED -ge 10 ]; then
    echo -e "${YELLOW}⚠ System mostly ready, but has some issues.${NC}"
    echo "Try running anyway, or fix the failed checks above."
    echo ""
    exit 1
else
    echo -e "${RED}✗ System not ready. Fix the failed checks above.${NC}"
    echo ""
    echo "Recommended actions:"
    if [ $FAILED -gt 5 ]; then
        echo "1. Run: ./install_ros2_robust.sh"
        echo "2. Run: ./build_ros2_robust.sh"
    elif ! command -v ros2 &> /dev/null; then
        echo "1. Run: ./install_ros2_robust.sh"
    elif [ ! -f "$ROS2_WS/install/setup.bash" ]; then
        echo "1. Run: ./build_ros2_robust.sh"
    fi
    echo ""
    exit 2
fi
#!/bin/bash

# VLM ROS2 Bag File Launch Script
# Analyzes recorded ROS2 bag files

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}==================================="
echo "VLM ROS2 Bag Analysis Mode"
echo "===================================${NC}"

# Check if setup has been run
if [ ! -f "$HOME/.vlm_ros2_setup.bash" ]; then
    echo -e "${RED}Error: ROS2 setup not found!${NC}"
    echo "Please run ./setup_ros2.sh first"
    exit 1
fi

# Source ROS2 environment
source $HOME/.vlm_ros2_setup.bash

# Check if ROS2 is working
if ! command -v ros2 &> /dev/null; then
    echo -e "${RED}Error: ROS2 not found in environment!${NC}"
    exit 1
fi

# Check for bag file argument
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: No bag file specified!${NC}"
    echo "Usage: $0 <bag_file> [options]"
    echo ""
    echo "Options:"
    echo "  --topic <name>     Image topic name (default: /camera/image_raw)"
    echo "  --rate <hz>        Analysis rate (default: 1.0 Hz)"
    echo "  --cpu              Use CPU instead of GPU"
    echo "  --prompt <text>    Analysis prompt"
    echo "  --loop             Loop the bag file"
    echo ""
    echo "Example:"
    echo "  $0 my_recording.db3 --rate 0.5 --prompt \"Describe the scene\""
    exit 1
fi

BAG_FILE="$1"
shift

# Check if bag file exists
if [ ! -f "$BAG_FILE" ] && [ ! -d "$BAG_FILE" ]; then
    echo -e "${RED}Error: Bag file not found: $BAG_FILE${NC}"
    exit 1
fi

# Parse remaining arguments
IMAGE_TOPIC="/camera/image_raw"
RATE="1.0"
DEVICE="cuda"
PROMPT="Describe what you see in this image"
LOOP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --topic)
            IMAGE_TOPIC="$2"
            shift 2
            ;;
        --rate)
            RATE="$2"
            shift 2
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --loop)
            LOOP="--loop"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Bag file: $BAG_FILE"
echo "  Image topic: $IMAGE_TOPIC"
echo "  Device: $DEVICE"
echo "  Rate: $RATE Hz"
echo "  Prompt: $PROMPT"
if [ ! -z "$LOOP" ]; then
    echo "  Loop: enabled"
fi
echo ""

# Check bag file info
echo -e "${GREEN}Checking bag file contents...${NC}"
ros2 bag info "$BAG_FILE"
echo ""

# Check if image topic exists in bag
if ! ros2 bag info "$BAG_FILE" | grep -q "$IMAGE_TOPIC"; then
    echo -e "${YELLOW}Warning: Topic $IMAGE_TOPIC not found in bag file${NC}"
    echo "Available image topics:"
    ros2 bag info "$BAG_FILE" | grep -E "(image|Image)" || echo "No image topics found"
    echo ""
    read -p "Enter image topic name (or press Enter to continue): " custom_topic
    if [ ! -z "$custom_topic" ]; then
        IMAGE_TOPIC="$custom_topic"
    fi
fi

# Check GPU availability
if [ "$DEVICE" == "cuda" ]; then
    if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        echo -e "${YELLOW}Warning: CUDA not available, falling back to CPU${NC}"
        DEVICE="cpu"
    fi
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    # Kill any remaining nodes
    pkill -f "ros2.*vlm_node.py" 2>/dev/null
    pkill -f "ros2 bag play" 2>/dev/null
}

trap cleanup EXIT

# Launch VLM node
echo -e "${GREEN}Launching VLM node...${NC}"
ros2 run vlm_ros2 vlm_node.py --ros-args \
    -p model_name:="InternVL3-2B-hf" \
    -p device:="$DEVICE" \
    -p image_topic:="$IMAGE_TOPIC" \
    -p continuous_analysis:=true \
    -p analysis_rate:=$RATE \
    -p default_prompt:="$PROMPT" \
    -p queue_size:=10 &
VLM_PID=$!

# Give node time to start
sleep 5

# Play bag file
echo -e "${GREEN}Playing bag file...${NC}"
ros2 bag play "$BAG_FILE" $LOOP --rate 1.0 &
BAG_PID=$!

# Monitor analysis output
echo -e "\n${GREEN}==================================="
echo "VLM Analysis Running"
echo "===================================${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Show analysis results
echo -e "${GREEN}Analysis results:${NC}"
ros2 topic echo /vlm/analysis vlm_ros2/msg/VLMAnalysis --field text &
ECHO_PID=$!

# Wait for processes
wait $BAG_PID
echo -e "${YELLOW}Bag playback completed${NC}"

# If not looping, wait a bit for final analyses then exit
if [ -z "$LOOP" ]; then
    sleep 3
    cleanup
fi

wait $VLM_PID
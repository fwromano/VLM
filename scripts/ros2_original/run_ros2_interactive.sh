#!/bin/bash

# VLM ROS2 Interactive Client Script
# Interactive interface for testing VLM services

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}==================================="
echo "VLM ROS2 Interactive Client"
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

# Check if VLM node is running
if ! ros2 node list | grep -q "/vlm"; then
    echo -e "${YELLOW}Warning: VLM node not detected${NC}"
    echo "Make sure to run ./run_ros2_camera.sh in another terminal first!"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Function to show menu
show_menu() {
    echo -e "\n${BLUE}=== VLM ROS2 Interactive Menu ===${NC}"
    echo "1. Set continuous analysis mode"
    echo "2. Analyze single image"
    echo "3. Start video analysis (action)"
    echo "4. View current analysis stream"
    echo "5. Check VLM status"
    echo "6. List available topics"
    echo "7. Custom prompt analysis"
    echo "8. Stop continuous mode"
    echo "9. Exit"
    echo ""
}

# Function to set continuous mode
set_continuous_mode() {
    echo -e "${GREEN}Setting continuous analysis mode...${NC}"
    read -p "Enter prompt (default: 'What do you see?'): " prompt
    prompt=${prompt:-"What do you see?"}
    
    read -p "Enter rate in Hz (default: 0.5): " rate
    rate=${rate:-0.5}
    
    ros2 service call /vlm/set_prompt vlm_ros2/srv/SetPrompt "{prompt: '$prompt', enable_continuous: true, analysis_rate: $rate}"
}

# Function to analyze single image
analyze_single() {
    echo -e "${GREEN}Analyzing single image...${NC}"
    read -p "Enter prompt: " prompt
    
    # This is a simplified version - in practice you'd need to grab current image
    echo "Calling analyze service..."
    ros2 run vlm_ros2 vlm_client_example.py single "$prompt"
}

# Function to start video analysis
start_video_analysis() {
    echo -e "${GREEN}Starting video analysis action...${NC}"
    read -p "Enter prompt: " prompt
    read -p "Enter duration in seconds (0 for continuous): " duration
    duration=${duration:-10.0}
    
    read -p "Enter rate in Hz (default: 0.5): " rate
    rate=${rate:-0.5}
    
    echo "Starting video analysis for $duration seconds..."
    ros2 run vlm_ros2 vlm_client_example.py video "$prompt" $duration $rate
}

# Function to view analysis stream
view_analysis() {
    echo -e "${GREEN}Viewing analysis stream (press Ctrl+C to stop)...${NC}"
    ros2 topic echo /vlm/analysis vlm_ros2/msg/VLMAnalysis
}

# Function to check status
check_status() {
    echo -e "${GREEN}VLM Node Status:${NC}"
    ros2 topic echo /vlm/status diagnostic_msgs/msg/DiagnosticStatus --once
}

# Function to list topics
list_topics() {
    echo -e "${GREEN}Available ROS2 topics:${NC}"
    ros2 topic list
    echo ""
    echo -e "${GREEN}VLM-related topics:${NC}"
    ros2 topic list | grep vlm
}

# Function for custom prompt
custom_prompt() {
    echo -e "${GREEN}Custom prompt analysis${NC}"
    echo "This will enable continuous mode with your custom prompt"
    read -p "Enter your custom prompt: " prompt
    
    if [ -z "$prompt" ]; then
        echo -e "${RED}Error: Empty prompt${NC}"
        return
    fi
    
    ros2 service call /vlm/set_prompt vlm_ros2/srv/SetPrompt "{prompt: '$prompt', enable_continuous: true, analysis_rate: 1.0}"
    
    echo -e "${GREEN}Monitoring results for 10 seconds...${NC}"
    timeout 10 ros2 topic echo /vlm/analysis vlm_ros2/msg/VLMAnalysis --field text
}

# Function to stop continuous mode
stop_continuous() {
    echo -e "${YELLOW}Stopping continuous analysis...${NC}"
    ros2 service call /vlm/set_prompt vlm_ros2/srv/SetPrompt "{prompt: '', enable_continuous: false, analysis_rate: 0.0}"
}

# Main loop
while true; do
    show_menu
    read -p "Select option (1-9): " choice
    
    case $choice in
        1) set_continuous_mode ;;
        2) analyze_single ;;
        3) start_video_analysis ;;
        4) view_analysis ;;
        5) check_status ;;
        6) list_topics ;;
        7) custom_prompt ;;
        8) stop_continuous ;;
        9) 
            echo -e "${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
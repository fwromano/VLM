#!/bin/bash

# Script to download and convert KITTI dataset sample for ROS2 testing

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}==================================="
echo "KITTI Dataset Download for ROS2"
echo "===================================${NC}"

# Create data directory
DATA_DIR="ros2_sample_data"
mkdir -p $DATA_DIR
cd $DATA_DIR

# Download KITTI sample
echo -e "${GREEN}Downloading KITTI raw data sample...${NC}"
echo "This will download ~350MB of data"

# Download sync data
if [ ! -f "2011_09_26_drive_0002_sync.zip" ]; then
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip
else
    echo "Sync data already downloaded"
fi

# Download calibration data
if [ ! -f "2011_09_26_calib.zip" ]; then
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip
else
    echo "Calibration data already downloaded"
fi

# Extract
echo -e "${GREEN}Extracting files...${NC}"
unzip -o 2011_09_26_drive_0002_sync.zip
unzip -o 2011_09_26_calib.zip

# Check if kitti2bag2 is installed
if ! command -v kitti2bag2 &> /dev/null; then
    echo -e "${YELLOW}kitti2bag2 not found. Installing...${NC}"
    pip install kitti2bag2
fi

# Convert to ROS2 bag
echo -e "${GREEN}Converting to ROS2 bag format...${NC}"
kitti2bag2 -t 2011_09_26 -r 0002 raw_synced .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Conversion successful!${NC}"
    echo ""
    echo "ROS2 bag created in: $(pwd)/kitti_2011_09_26_drive_0002_sync"
    echo ""
    echo "To analyze this data with VLM:"
    echo "  cd .."
    echo "  ./run_ros2_bag.sh $DATA_DIR/kitti_2011_09_26_drive_0002_sync --topic /camera_color_left/image_raw --rate 0.5"
else
    echo -e "${RED}Conversion failed!${NC}"
    echo "Make sure you have ROS2 environment activated"
fi

cd ..

# Create a simple test bag recorder
echo -e "\n${GREEN}Creating webcam recording script...${NC}"
cat > record_webcam_bag.sh << 'EOF'
#!/bin/bash
# Record webcam to ROS2 bag for testing

source $HOME/.vlm_ros2_setup.bash

OUTPUT_DIR="ros2_sample_data/webcam_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Recording webcam to $OUTPUT_DIR"
echo "Press Ctrl+C to stop recording"

# Start camera node in background
ros2 run usb_cam usb_cam_node_exe &
CAM_PID=$!

# Give camera time to start
sleep 2

# Record
ros2 bag record -o $OUTPUT_DIR/webcam_recording /camera/image_raw /camera/camera_info

# Cleanup
kill $CAM_PID 2>/dev/null

echo "Recording saved to: $OUTPUT_DIR"
EOF

chmod +x record_webcam_bag.sh

echo -e "${GREEN}Done!${NC}"
echo ""
echo "Created:"
echo "  - KITTI sample bag in: $DATA_DIR/"
echo "  - Webcam recorder: ./record_webcam_bag.sh"
# VLM ROS2 Integration

ROS2 package for integrating InternVL3 Vision Language Model with robotic systems.

## Features

- **Real-time Analysis**: Process camera streams with configurable rates
- **Service Interface**: On-demand image analysis
- **Action Server**: Long-running video analysis with feedback
- **Custom Messages**: Rich analysis results with metadata
- **Multi-threading**: Efficient processing with queue management
- **GPU Support**: Automatic CUDA detection and fallback

## Installation

```bash
# Clone into ROS2 workspace
cd ~/ros2_ws/src
git clone <this-repo> -b ROS

# Install dependencies
sudo apt install ros-humble-usb-cam ros-humble-image-view
pip install torch torchvision transformers

# Build
cd ~/ros2_ws
colcon build --packages-select vlm_ros2
source install/setup.bash
```

## Usage

### 1. Live Camera Analysis
```bash
ros2 launch vlm_ros2 vlm_camera.launch.py
```

### 2. Bag File Analysis
```bash
ros2 launch vlm_ros2 vlm_bag.launch.py bag_file:=/path/to/bag
```

### 3. Client Examples
```bash
# Continuous mode
ros2 run vlm_ros2 vlm_client_example.py continuous "What objects do you see?"

# Single image
ros2 run vlm_ros2 vlm_client_example.py single "Describe the scene"

# Video analysis
ros2 run vlm_ros2 vlm_client_example.py video "Track any movement" 30.0 2.0
```

## Topics

- **Subscribe**: `/camera/image_raw` (sensor_msgs/Image)
- **Publish**: `/vlm/analysis` (vlm_ros2/VLMAnalysis)
- **Publish**: `/vlm/status` (diagnostic_msgs/DiagnosticStatus)

## Services

- `/vlm/analyze_image`: Single image analysis
- `/vlm/set_prompt`: Configure continuous analysis

## Actions

- `/vlm/analyze_video`: Analyze video stream with progress feedback

## Parameters

- `model_name`: VLM model to use (default: InternVL3-2B-hf)
- `device`: cuda or cpu
- `continuous_analysis`: Enable automatic analysis
- `analysis_rate`: Hz for continuous mode
- `default_prompt`: Default question for analysis

## KITTI Dataset Example

```bash
# Download and convert KITTI to ROS2
pip install kitti2bag2
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip
unzip 2011_09_26_drive_0002_sync.zip
kitti2bag2 -t 2011_09_26 -r 0002 raw_synced .

# Analyze KITTI data
ros2 launch vlm_ros2 vlm_bag.launch.py bag_file:=./kitti_2011_09_26_drive_0002_sync
```
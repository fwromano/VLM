# VLM ROS2 Integration

A complete ROS2 package for integrating Vision-Language Models (VLMs) with robotic systems.

## Features

- **Real-time Analysis**: Process camera streams with configurable rates
- **Service Interface**: On-demand image analysis
- **Action Server**: Long-running video analysis with feedback
- **Custom Messages**: Rich analysis results with metadata
- **Multi-threading**: Efficient processing with queue management
- **GPU Support**: Automatic CUDA detection and fallback
- **Multiple Input Sources**: Live camera, ROS bag files, image topics

## Project Structure

```
ros2_vlm/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup.sh                   # One-click setup
├── install_ros2_robust.sh     # ROS2 Jazzy installer
├── build_ros2_robust.sh       # Build script
├── run_ros2_camera_robust.sh  # Camera launcher
├── validate_ros2_setup.sh     # Setup validator
├── demos/                     # Demo scripts
└── src/vlm_ros2/             # ROS2 package source
```

## Quick Start

### 1. Setup Environment
```bash
# Install everything (ROS2 + dependencies)
./setup.sh
```

### 2. Build Package
```bash
./build_ros2_robust.sh
```

### 3. Run with Camera
```bash
./run_ros2_camera_robust.sh
```

## Detailed Installation

### Prerequisites
- Ubuntu 22.04 or 24.04
- NVIDIA GPU (optional, will use CPU if not available)
- Python 3.10+

### Step 1: Install ROS2 Jazzy
```bash
./install_ros2_robust.sh
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Build the Package
```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vlm_ros2
source install/setup.bash
```

### Step 4: Validate Installation
```bash
./validate_ros2_setup.sh
```

## Usage

### Live Camera Analysis
```bash
ros2 launch vlm_ros2 vlm_camera.launch.py
```

### Bag File Analysis
```bash
ros2 launch vlm_ros2 vlm_bag.launch.py bag_file:=/path/to/bag
```

### Client Examples
```bash
# Continuous mode
ros2 run vlm_ros2 vlm_client_example.py continuous "What objects do you see?"

# Single image
ros2 run vlm_ros2 vlm_client_example.py single "Describe the scene"

# Video analysis
ros2 run vlm_ros2 vlm_client_example.py video "Track any movement" 30.0 2.0
```

## ROS2 Interface

### Topics
- **Subscribe**: `/camera/image_raw` (sensor_msgs/Image)
- **Publish**: `/vlm/analysis` (vlm_ros2/VLMAnalysis)
- **Publish**: `/vlm/status` (diagnostic_msgs/DiagnosticStatus)

### Services
- `/vlm/analyze_image`: Single image analysis
- `/vlm/set_prompt`: Configure continuous analysis

### Actions
- `/vlm/analyze_video`: Analyze video stream with progress feedback

### Parameters
- `model_name`: VLM model to use (default: InternVL3-2B-hf)
- `device`: cuda or cpu
- `continuous_analysis`: Enable automatic analysis
- `analysis_rate`: Hz for continuous mode
- `default_prompt`: Default question for analysis

## Configuration

Edit `src/vlm_ros2/config/vlm_params.yaml` to customize:
- Model selection
- Analysis parameters
- Performance settings

## Demo: KITTI Dataset

```bash
cd demos
./download_kitti_sample.sh
./run_vlm_ros2_demo.sh
```

## Troubleshooting

1. **GPU not detected**: Ensure CUDA is properly installed
2. **Import errors**: Run `./validate_ros2_setup.sh`
3. **Build failures**: Check ROS2 environment is sourced

## License

MIT License - See LICENSE file for details
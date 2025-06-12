# VLM Projects

Vision-Language Model (VLM) implementations for different use cases.

## Projects

### 1. [Standalone VLM Chat](./standalone/)
A simple, clean video chat interface with VLM integration.
- Real-time camera feed
- Interactive chat interface
- No ROS dependencies
- Quick setup and usage

**Quick Start:**
```bash
cd standalone
./setup.sh
./run.sh
```

### 2. [ROS2 VLM Integration](./ros2_vlm/)
Complete ROS2 package for robotic VLM applications.
- ROS2 services and actions
- Bag file support
- Multi-robot integration
- Professional robotics use

**Quick Start:**
```bash
cd ros2_vlm
./setup.sh
./run_ros2_camera_robust.sh
```

## Requirements

- Ubuntu 22.04 or 24.04
- Python 3.10+
- NVIDIA GPU (optional)
- Webcam

## Choose Your Project

- **For simple VLM chat**: Use the [standalone](./standalone/) version
- **For robotics/ROS2**: Use the [ros2_vlm](./ros2_vlm/) version

Each project is self-contained with its own README, setup scripts, and dependencies.
# VLM ROS2 Integration

Real-time vision-language model integration with live camera demo and ROS2 support.

## 🚀 Quick Start

```bash
# 1. One-time setup
sudo ./install_ros2_packages.sh

# 2. Run demo  
./demo.sh
```

## 🎥 Live Demo (Recommended)

**Integrated camera + VLM analysis in one window:**
- ✅ Real-time camera feed with analysis overlay
- ✅ 1.5 second analysis intervals  
- ✅ Keyboard shortcuts for instant prompt changes
- ✅ No ROS2 complexity - just works!

**Controls:**
- `H` - Show/hide help
- `1-9` - Quick prompt presets
- `SPACE` - Force immediate analysis
- `Q/ESC` - Quit

**Quick Prompts:**
1. What objects do you see?
2. Describe colors and scene
3. Is this environment safe?
4. Count people in image
5. What actions are happening?
6. Describe lighting and mood
7. What text or signs do you see?
8. Indoors or outdoors?
9. What time of day?

## 🤖 ROS2 Demo (For Robotics)

**Full ROS2 integration with topics/services:**
- Camera publisher on `/camera/image_raw`
- VLM analysis on `/vlm/analysis`  
- Prompt changes via `/vlm/set_prompt`

**Manual ROS2 usage:**
```bash
# Terminal 1: VLM Node
./run_vlm_ros2.sh

# Terminal 2: Camera
source /opt/ros/jazzy/setup.bash
ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video0

# Terminal 3: Change prompts
ros2 topic pub --once /vlm/set_prompt std_msgs/msg/String "data: 'What do you see?'"
```

## 🏗️ Architecture

**Hybrid Design:**
- **Live Demo** (`live_vlm_demo.py`) - Direct camera + VLM integration
- **ROS2 Node** (`hybrid_vlm_node.py`) - Runs in system Python 3.12
- **VLM Processor** (`vlm_processor.py`) - Runs in conda Python 3.10

**Benefits:**
- No conda/ROS2 environment conflicts
- Fast analysis (1.5s intervals)
- Easy prompt switching
- Production-ready architecture

## 📁 Files

**Main Demo:**
- `demo.sh` - Unified launcher with menu
- `live_vlm_demo.py` - Integrated live demo (recommended)

**ROS2 Integration:**
- `hybrid_vlm_node.py` - ROS2 VLM node
- `vlm_processor.py` - Gemma 3 processing
- `run_vlm_ros2.sh` - ROS2 launcher

**Setup/Test:**
- `install_ros2_packages.sh` - Install ROS2 dependencies
- `test_vlm_processor.sh` - Test VLM processor

## 📋 Requirements

- Ubuntu 22.04/24.04
- USB camera at `/dev/video0`
- Python 3.12 (system) + conda VLM environment
- NVIDIA GPU (optional, falls back to CPU)
- HuggingFace account with Gemma 3 access

## 💡 Usage Tips

**For quick testing:** Use the Live Demo - no ROS2 knowledge needed!

**For robotics projects:** Use the ROS2 Demo for full integration.

**Performance:** 
- GPU: ~1-2 second analysis
- CPU: ~3-5 second analysis

**Customization:**
- Edit prompts in `live_vlm_demo.py`
- Adjust analysis interval in demo files
- Change VLM model in `vlm_processor.py`
# VLM ROS2 Integration

Real-time vision-language model integration using Google Gemma 3 4B with live camera demos and ROS2 support.

## Quick Start

```bash
# 1. One-time setup (install ROS2 packages)
sudo ./setup/install_ros2_packages.sh

# 2. Run demos
./demo.sh
```

### Using the VLM Server (Recommended)

For best latency and to avoid per‑request cold loads, start the persistent VLM server first from the repo root:

```bash
./serve.sh
# Server runs at http://localhost:8080
```

The ROS2 live demo and node will automatically use the server for inference when it is reachable and fall back to the existing conda subprocess if not.

Optional environment overrides (set before running the demo):

```bash
export VLM_SERVER_URL=http://127.0.0.1:8080      # or remote server URL
export VLM_SERVER_MODEL=gemma-3-4b-it            # gemma-3n-e4b-it | internvl-3
export VLM_SERVER_BACKEND=transformers           # vllm if CUDA on the server
export VLM_SERVER_FAST=1                         # smaller image + fewer tokens
```

## What This Does

This project lets you ask questions about what your camera sees in real-time using Google's Gemma 3 vision-language model. You can either use a simple integrated demo or full ROS2 integration for robotics projects.

## Two Demo Options

### Option 1: Live Demo (Recommended for Testing)

**What it is:** Single window showing camera feed with VLM analysis overlay
**How it works:** Direct camera access + background VLM processing
**Use case:** Quick testing, demonstrations, simple applications

**Features:**
- Real-time camera feed with analysis overlay
- 2 second analysis intervals  
- Keyboard shortcuts for instant prompt changes
- No ROS2 complexity - just works!
- Uses VLM Server automatically when available for lower latency

**Controls:**
- `H` - Show/hide help
- `1-9` - Quick prompt presets (see below)
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

### Option 2: ROS2 Demo (For Robotics Development)

**What it is:** Full ROS2 integration with topics and services
**How it works:** ROS2 nodes communicating via topics
**Use case:** Robotics projects, multi-node systems, production deployments

**Features:**
- Camera publisher on `/camera/image_raw`
- VLM analysis published on `/vlm/analysis`  
- Dynamic prompt changes via `/vlm/set_prompt`
- 1.5 second analysis intervals
- Full ROS2 ecosystem integration
- Uses VLM Server automatically when available; falls back to local subprocess for isolation

**What you see:**
- Terminal 1: Camera feed viewer (rqt_image_view)
- Terminal 2: Full VLM analysis results (live text output)

**Change prompts in real-time:**
```bash
ros2 topic pub --once /vlm/set_prompt std_msgs/msg/String "data: What do you see?"
```

## Architecture Explained

### The Challenge: Environment Conflicts

**Problem:** ROS2 Jazzy requires Python 3.12 (system), but VLM models work best in conda (Python 3.10)

**Solution:** Hybrid architecture that keeps environments separate

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVE DEMO PATH                            │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Camera    │───▶│ OpenCV + VLM │───▶│   Display   │     │
│  │  /dev/video0│    │Integration   │    │  with       │     │
│  └─────────────┘    │(live_vlm_    │    │  Overlay    │     │
│                     │ demo.py)     │    └─────────────┘     │
│                     └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    ROS2 DEMO PATH                           │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │USB Cam Node │───▶│Hybrid VLM   │───▶│Topic        │     │
│  │(System      │    │Node         │    │Publishers   │     │
│  │ Python 3.12)│    │(System      │    │/vlm/analysis│     │
│  └─────────────┘    │ Python 3.12)│    └─────────────┘     │
│                     └──────┬──────┘                        │
│                            │                               │
│                            │ subprocess                    │
│                            ▼                               │
│                     ┌─────────────┐                        │
│                     │VLM Processor│                        │
│                     │(Conda       │                        │
│                     │ Python 3.10)│                        │
│                     └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

**1. Live Demo (`live_vlm_demo.py`)**
- Single Python process
- Direct camera access via OpenCV
- Background threading for VLM processing
- Real-time overlay rendering
- Uses system Python but calls conda VLM via subprocess

**2. Hybrid VLM Node (`hybrid_vlm_node.py`)**
- ROS2 node in system Python 3.12
- Subscribes to camera images
- Calls VLM processor via subprocess
- Publishes results to ROS2 topics

**3. VLM Processor (`vlm_processor.py`)**
- Runs in conda environment (Python 3.10)
- Loads Google Gemma 3 4B model
- Handles GPU/CPU device mapping
- Called as subprocess by both demos

**4. Environment Isolation**
- System Python for ROS2 (no conda interference)
- Conda Python for VLM models (proper dependencies)
- Subprocess communication bridges the gap

## File Structure

```
ros2_vlm/
├── README.md                 # This file  
├── demo.sh                   # Main launcher with menu
├── setup/                    # Setup and testing
│   ├── install_ros2_packages.sh  # Install ROS2 dependencies
│   └── test_vlm_processor.sh     # Test VLM functionality
├── demos/                    # Demo applications
│   ├── live_vlm_demo.py          # Option 1: Integrated demo
│   └── ros2_camera_demo.sh       # Option 2: ROS2 demo launcher
└── nodes/                    # ROS2 nodes and core processing
    ├── hybrid_vlm_node.py        # ROS2 node (system Python)
    ├── vlm_processor.py          # VLM processing (conda Python)
    └── run_vlm_ros2.sh          # ROS2 node launcher
```

## Dependencies & Requirements

**System Requirements:**
- Ubuntu 22.04/24.04
- USB camera at `/dev/video0`
- NVIDIA GPU recommended (16GB+ for optimal performance)
- HuggingFace account with Google Gemma 3 access
- VLM Server can run on the same host or a separate GPU host

**Python Environments:**
- System Python 3.12 (for ROS2)
- Conda environment 'vlm' with Python 3.10 (for VLM models)

**ROS2 Packages:**
- `usb_cam` - Camera driver
- `rqt_image_view` - Image viewer
- `cv_bridge` - OpenCV/ROS2 bridge

## Performance

**GPU Performance (RTX 5000 Ada 16GB):**
- Model loading: ~3-5 seconds (first run)
- Analysis time: ~1-2 seconds per image
- Memory usage: ~8-10GB VRAM

**CPU Fallback:**
- Model loading: ~10-15 seconds
- Analysis time: ~3-5 seconds per image
- Memory usage: ~6-8GB RAM

## Troubleshooting

**Model loading fails:**
- Check HuggingFace authentication: `huggingface-cli login`
- Verify Gemma 3 access permissions on HuggingFace
- Ensure conda environment exists: `conda activate vlm`

**Camera not found:**
- Check camera connection: `ls /dev/video*`
- Test camera: `cheese` or `vlc v4l2:///dev/video0`

**ROS2 errors:**
- Source ROS2 setup: `source /opt/ros/jazzy/setup.bash`
- Install packages: `sudo ./install_ros2_packages.sh`
- If using the VLM Server: check `/v1/health` and server logs

**Memory errors:**
- Reduce model precision in `vlm_processor.py`
- Use CPU fallback if GPU memory insufficient

## Customization

**Change VLM model:**
Edit `nodes/vlm_processor.py` line 35-36 to use different model

**Adjust analysis intervals:**
- Live demo: Edit `analysis_interval` in `demos/live_vlm_demo.py` line 259
- ROS2 demo: Edit `analysis_interval` in `nodes/hybrid_vlm_node.py` line 40

**Add custom prompts:**
Edit `prompts` dictionary in `demos/live_vlm_demo.py` lines 42-52

**Modify camera settings:**
Edit camera parameters in respective demo files (resolution, FPS, etc.)

# VLM Projects

Vision-Language Model (VLM) implementations using Google Gemma 3 4B for real-time camera analysis.

## What This Repository Contains

This repository provides two different approaches to using vision-language models with live camera feeds:

1. **VLM Video Chat** - Optimized video chat interface
2. **ROS2 VLM Integration** - Full robotics integration

## Project Structure

```
VLM/
├── README.md              # This file
├── CLAUDE.md             # Development notes
├── vlm_video_chat/      # Optimized video chat interface
│   ├── README.md
│   ├── setup.sh
│   ├── run.sh
│   └── vlm_standalone.py
└── ros2_vlm/            # Full ROS2 integration
    ├── README.md        # Detailed architecture guide
    ├── demo.sh          # Main launcher
    ├── setup/           # Setup and testing
    ├── demos/           # Demo applications
    └── nodes/           # ROS2 nodes and processing
```

## Quick Start Guide

### Launch VLM Server (one command)

Run a persistent VLM server with smart Linux/macOS detection, optional vLLM, and audio transcription:

```bash
./serve.sh
```

The server starts on `http://localhost:8080`. Demos auto-use it when available.

### For Simple Testing (Recommended)

If you just want to test VLM with your camera:

```bash
cd ros2_vlm
sudo ./setup/install_ros2_packages.sh
./demo.sh
# Choose option 1 (Live Demo)
```

This gives you a single window with live camera + VLM analysis overlay.

### For Robotics Development

If you're building ROS2 robotics applications:

```bash
cd ros2_vlm  
sudo ./setup/install_ros2_packages.sh
./demo.sh
# Choose option 2 (ROS2 Demo)
```

This provides full ROS2 topics/services integration.

### For Minimal Dependencies

If you want the simplest possible setup:

```bash
cd vlm_video_chat
./setup.sh
./run.sh
```

This is an optimized VLM video chat with performance enhancements.

## What Each Project Does

### VLM Video Chat (`vlm_video_chat/`)

**Purpose:** Optimized VLM video chat interface
**Use case:** Quick testing, demonstrations, interactive VLM usage
**Architecture:** Single Python process with performance optimizations

**Features:**
- Clean video chat interface with real-time camera
- 10 quick prompt buttons with tooltips
- Optimized for RTX 5000 Ada (16GB GPU)
- Performance tuned: 512px inputs, KV cache, explicit device mapping
- Chat history with timestamps and processing times

### ROS2 VLM Integration (`ros2_vlm/`)

**Purpose:** Production-ready VLM for robotics
**Use case:** Robot navigation, scene understanding, multi-node systems
**Architecture:** Hybrid system (ROS2 + conda environments)

**Features:**
- Two demo modes (integrated + ROS2)
- Real-time camera analysis
- ROS2 topic/service integration
- Keyboard shortcuts for quick prompts
- Full model output display
- GPU optimization for RTX 5000 Ada

## Architecture Comparison

### Standalone Approach
```
Camera → VLM Processing → Chat Interface
(Single environment, simple setup)
```

### ROS2 Hybrid Approach
```
Camera → ROS2 Node → VLM Subprocess → ROS2 Topics
(Environment isolation, production-ready)
```

## Key Technical Innovations

### Environment Isolation Solution

**Problem:** ROS2 Jazzy requires Python 3.12, but VLM models work best in conda Python 3.10

**Solution:** Hybrid architecture where:
- ROS2 nodes run in system Python 3.12
- VLM processing runs in conda Python 3.10
- Subprocess communication bridges environments

### GPU Memory Optimization

**Problem:** Gemma 3 4B model has complex device mapping requirements

**Solution:** Smart device mapping that:
- Detects GPU memory (15.7GB RTX 5000 Ada)
- Uses explicit device placement for large GPUs
- Falls back to auto-mapping for smaller GPUs
- Handles CPU fallback gracefully

## Requirements

**System:**
- Ubuntu 22.04/24.04
- USB camera at `/dev/video0`
- 8GB+ RAM (16GB+ recommended)

**GPU (Optional but Recommended):**
- NVIDIA GPU with 8GB+ VRAM
- RTX 5000 Ada (16GB) for optimal performance
- CPU fallback available

**Authentication:**
- HuggingFace account
- Access to google/gemma-3-4b-it model
- HuggingFace CLI login

## Performance Expectations

**With RTX 5000 Ada (16GB):**
- Model loading: 3-5 seconds
- Analysis: 1-2 seconds per frame
- Memory usage: 8-10GB VRAM

**CPU Fallback:**
- Model loading: 10-15 seconds  
- Analysis: 3-5 seconds per frame
- Memory usage: 6-8GB RAM

## Choose Your Path

| Use Case | Project | Setup Complexity | Features |
|----------|---------|------------------|----------|
| **Quick Testing** | `ros2_vlm/` (Option 1) | Medium | Live camera + overlay |
| **Robotics Development** | `ros2_vlm/` (Option 2) | Medium | Full ROS2 integration |
| **Optimized Video Chat** | `vlm_video_chat/` | Low | Performance-tuned interface |

## Getting Help

1. **Start with:** `ros2_vlm/` project for most use cases
2. **Read:** The detailed `ros2_vlm/README.md` for architecture understanding
3. **Test first:** Use the test script to verify VLM functionality
4. **Troubleshoot:** Check camera, GPU, and HuggingFace authentication

Each project directory contains its own detailed README with specific setup instructions and troubleshooting guides.

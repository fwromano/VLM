# VLM Video Chat - Organized Structure

Real-time camera analysis using InternVL3 vision-language models with interactive interfaces.

**🤖 ROS2 Integration Available!** Check out the [ROS branch](../../tree/ROS) for robotic applications.

## 📁 Directory Structure

```
VLM/
├── 🎯 MAIN APPLICATIONS
│   ├── vlm.py                    # Original VLM video chat GUI
│   ├── vlm_model_selector.py     # NEW: Multi-model comparison tool
│   ├── setup.sh                  # Setup for original VLM
│   └── run_vlm.sh               # Run original VLM
│
├── 🚀 QUICK COMPARISON TOOLS
│   ├── compare_models.sh         # Easy model comparison
│   └── run_vlm.sh               # Original app launcher
│
├── 🎮 demos/
│   ├── run_vlm_ros2_demo.sh     # ROS2-style demo (no ROS2 needed)
│   ├── run_vlm_ros2_standalone.py # Standalone VLM with ROS2 interface
│   └── download_kitti_sample.sh  # Get sample datasets
│
├── 🔧 scripts/ros2_robust/       # BULLETPROOF ROS2 SCRIPTS
│   ├── install_ros2_robust.sh   # Install ROS2 with hard failure checks
│   ├── build_ros2_robust.sh     # Build with validation
│   ├── run_ros2_camera_robust.sh # Run with prerequisites check
│   └── validate_ros2_setup.sh   # Complete system validation
│
├── 📜 scripts/ros2_original/     # Original ROS2 scripts (basic)
│   ├── setup_ros2.sh            # Basic ROS2 setup
│   ├── run_ros2_camera.sh       # Basic camera launch
│   └── ...                      # Other original scripts
│
└── 🤖 ros2_vlm/                 # Full ROS2 package
    └── src/vlm_ros2/            # ROS2 package source
        ├── scripts/vlm_node.py  # Main ROS2 node
        ├── msg/                 # Custom messages
        ├── srv/                 # Services
        └── launch/              # Launch files
```

## 🎯 What Should You Use?

### For Model Comparison & Performance Testing:
```bash
# NEW: Compare different InternVL3 models
./vlm_model_selector.py --list-models    # See all available models
./compare_models.sh 2B                   # Fast 2B model
./compare_models.sh 8B                   # Balanced 8B model  
./compare_models.sh 26B                  # High quality 26B model
```

### For Original VLM Experience:
```bash
./setup.sh && ./run_vlm.sh
```

### For ROS2-Style Demo (No ROS2 Required):
```bash
./demos/run_vlm_ros2_demo.sh
```

### For Full ROS2 Integration:
```bash
# Robust installation (recommended)
./scripts/ros2_robust/install_ros2_robust.sh
./scripts/ros2_robust/validate_ros2_setup.sh
./scripts/ros2_robust/build_ros2_robust.sh
./scripts/ros2_robust/run_ros2_camera_robust.sh
```

## 🔥 NEW: InternVL3 Model Selector

Compare performance across different InternVL3 models:

| Model | Parameters | VRAM | Speed | Quality | Best For |
|-------|------------|------|-------|---------|----------|
| **2B** | 2B | 4GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Real-time, edge devices |
| **8B** | 8B | 12GB | ⭐⭐⭐ | ⭐⭐⭐⭐ | General purpose |
| **26B** | 26B | 24GB | ⭐⭐ | ⭐⭐⭐⭐⭐ | High quality analysis |
| **72B** | 72B | 48GB | ⭐ | ⭐⭐⭐⭐⭐ | Research, max quality |

### Features:
- 📊 **Real-time performance tracking** (avg/min/max processing times)
- 🎮 **Interactive controls** (pause, single-shot, reset stats)
- 💾 **Performance logging** to JSON files
- 🖥️ **On-screen overlay** with model info and stats
- 🔄 **Hot-swappable** prompts and analysis rates

### Example Usage:
```bash
# Compare 2B vs 8B models
./compare_models.sh 2B "Count the people" 1.0
# Then run:
./compare_models.sh 8B "Count the people" 1.0
# Compare the performance logs!
```

## 🛡️ Robust vs Original Scripts

**Use Robust Scripts When:**
- You want bulletproof installation
- You need clear error messages
- You're setting up for production use
- You want validation before proceeding

**Use Original Scripts When:**  
- Quick testing
- You know your system is already set up
- You want simpler output

## 📊 Performance Analysis

The model selector saves detailed performance logs:
- Processing times per frame
- Model specifications
- Device information  
- Analysis results

Perfect for:
- Choosing the right model for your use case
- Benchmarking different hardware
- Optimizing analysis rates
- Research and development

---

## 🚀 Quick Start Recommendations:

1. **New Users**: Start with `./compare_models.sh 2B`
2. **Performance Testing**: Use `./vlm_model_selector.py --model 8B`  
3. **ROS2 Development**: Use robust scripts in `scripts/ros2_robust/`
4. **Just Want It Working**: Use `./demos/run_vlm_ros2_demo.sh`
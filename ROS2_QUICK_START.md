# VLM ROS2 Quick Start Guide

Get VLM running with ROS2 in minutes!

## 🚀 Super Quick Setup (3 Steps)

### 1. Install Everything
```bash
./setup_ros2.sh
```
This installs ROS2, cmake, and all dependencies automatically.

### 2. Open New Terminal
```bash
source ~/.bashrc
vlm_ros2
```

### 3. Run VLM
```bash
./run_ros2_camera.sh
```

That's it! VLM is now analyzing your camera feed with ROS2.

## 📹 What Can I Do?

### Live Camera Analysis
```bash
./run_ros2_camera.sh
```
- Analyzes webcam in real-time
- Publishes results to `/vlm/analysis` topic
- Default: "What do you see?" every 0.5 seconds

### Analyze Recorded Videos
```bash
# Download sample data
./download_kitti_sample.sh

# Analyze it
./run_ros2_bag.sh ros2_sample_data/kitti_2011_09_26_drive_0002_sync
```

### Interactive Testing
```bash
# Terminal 1: Start VLM
./run_ros2_camera.sh

# Terminal 2: Interactive client
./run_ros2_interactive.sh
```

## 🎮 Simple Commands

### Change What VLM Looks For
```bash
# In another terminal
vlm_ros2
ros2 service call /vlm/set_prompt vlm_ros2/srv/SetPrompt "{prompt: 'Count the people', enable_continuous: true, analysis_rate: 1.0}"
```

### See Results
```bash
ros2 topic echo /vlm/analysis --field text
```

### Record Your Own Video
```bash
./record_webcam_bag.sh
# Press Ctrl+C to stop
```

## 🛠️ Troubleshooting

### "cmake not found"
Run: `./setup_ros2.sh`

### "ROS2 not found"
Run: `source ~/.bashrc && vlm_ros2`

### "GPU not available"
Add `--cpu` flag: `./run_ros2_camera.sh --cpu`

### "Camera not found"
Check devices: `ls /dev/video*`
Use different camera: `./run_ros2_camera.sh` (will prompt)

## 📊 Quick Options

### Camera Mode Options
```bash
./run_ros2_camera.sh --rate 2.0              # 2 analyses per second
./run_ros2_camera.sh --cpu                   # Use CPU instead of GPU
./run_ros2_camera.sh --no-continuous         # Manual trigger only
./run_ros2_camera.sh --prompt "Find faces"   # Custom analysis
```

### Bag Mode Options
```bash
./run_ros2_bag.sh my_recording.db3 --loop    # Loop playback
./run_ros2_bag.sh data.db3 --rate 0.5        # Slower analysis
./run_ros2_bag.sh data.db3 --topic /my/image # Different topic
```

## 🎯 Common Use Cases

### Security Camera
```bash
./run_ros2_camera.sh --prompt "Alert if you see any person" --rate 2.0
```

### Object Counter
```bash
./run_ros2_camera.sh --prompt "Count the objects on the table" --rate 0.5
```

### Scene Monitor
```bash
./run_ros2_camera.sh --prompt "Describe any changes in the scene" --rate 1.0
```

## 💡 Tips

1. **First Run**: Takes 10-15s to load model
2. **Multiple Terminals**: Use `vlm_ros2` in each new terminal
3. **Stop Everything**: Press Ctrl+C in the terminal
4. **Rebuild After Changes**: `./build_ros2.sh`

---

Need help? Just run the setup and follow the prompts!
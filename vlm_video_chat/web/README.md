# VLM Web Interface

Web-based VLM video chat application with Flask and SocketIO.

## Features
- Browser-based interface accessible from any device
- Real-time camera feed via WebRTC
- Text input for custom questions  
- Chat history with timestamps
- Mobile-friendly responsive design
- Network access for multiple users

## Setup

```bash
# Setup environment (run once)
chmod +x setup.sh && ./setup.sh

# Run camera chat
./run_vlm.sh

# Run video upload + QA (HTML5 player, pause/play)
./run_video_qa.sh
```

## Requirements
- Python 3.10
- Conda/Miniconda  
- CUDA-capable GPU (recommended)
- Modern web browser with camera support

## Usage
1. Start the camera chat with `./run_vlm.sh` then open `http://localhost:5000`
   - Allow camera access when prompted
   - Type questions and press Enter to analyze live frames
2. Or start the video QA app with `./run_video_qa.sh` then open `http://localhost:5050/video_qa`
   - Upload a video (mp4/webm/mov)
   - Play/pause and seek
   - Ask about the current frame using Fast VLM (Gemma 3 4B), InternVL, or Gemma 3n E4B

## Network Access
- Local: `http://localhost:5000`
- Network: `http://[your-ip]:5000` (accessible from other devices)

## Dependencies
See `requirements.txt` for the complete list of Python packages needed.

### Server-First Mode
- For best latency and consistent results, start the VLM Server from repo root:
  - `./serve.sh` (server at `http://localhost:8080`)
- Both camera chat and video QA will automatically use the server for inference and fall back locally if unavailable.

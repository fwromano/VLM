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

# Run web server
./run_vlm.sh
```

## Requirements
- Python 3.10
- Conda/Miniconda  
- CUDA-capable GPU (recommended)
- Modern web browser with camera support

## Usage
1. Start the server with `./run_vlm.sh`
2. Open browser to `http://localhost:5000`
3. Allow camera access when prompted
4. Type questions and press Enter to analyze
5. View responses in the chat interface

## Network Access
- Local: `http://localhost:5000`
- Network: `http://[your-ip]:5000` (accessible from other devices)

## Dependencies
See `requirements.txt` for the complete list of Python packages needed.
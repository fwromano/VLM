# VLM Video Chat

Real-time camera analysis using vision-language models with multiple standalone interface options.

## What This Does

Interactive video chat with Google Gemma 3 models - ask questions about what your camera sees and get intelligent responses in real-time.

## Available Interfaces

### 🖥️ Standalone Desktop (`standalone/`)
- Native desktop application using Tkinter
- Lightweight and fast
- No network dependencies once set up
- Completely independent installation

```bash
cd standalone
./setup.sh    # Setup environment
./run_vlm.sh  # Run application
```

### 🌐 Web Interface (`web/`)
- Browser-based interface accessible from any device
- Mobile-friendly responsive design
- Network access for multiple users
- Completely independent installation

```bash
cd web
./setup.sh    # Setup environment
./run_vlm.sh  # Run web server (http://localhost:5000)
```

## Features

- **Real-time Camera**: Live camera feed with VLM processing
- **Interactive Chat**: Video call style interface with chat history
- **Custom Questions**: Type any question about what you see
- **Model Support**: Gemma 3 4B/12B (quantized) models
- **GPU Optimized**: CUDA support with CPU fallback
- **Multi-threaded**: Zero lag video with background AI processing
- **Independent Modules**: Each interface stands completely alone

## Quick Start

Choose your preferred interface and follow its setup:

```bash
# For Standalone Desktop:
cd standalone && ./setup.sh && ./run_vlm.sh

# For Web Interface:
cd web && ./setup.sh && ./run_vlm.sh
```

**Note**: Each interface requires separate setup and has its own conda environment.

## Interface Controls

- **Type questions** in text box
- **Enter** - Send message  
- **Ctrl+Enter** - New line
- **Number buttons (1-9)** - Quick prompts:
  1. What do you see in this image?
  2. Describe this scene in detail
  3. What objects are visible?
  4. What colors are prominent?
  5. Is this environment safe?
  6. What actions are happening?
  7. Count the people in this image
  8. What text or signs do you see?
  9. Is this indoors or outdoors?
  10. What time of day does this look like?

## Performance Optimizations

**GPU (RTX 5000 Ada):**
- Model loading: ~3-5 seconds
- Analysis: ~0.5-1.5 seconds per image
- Memory usage: ~8-10GB VRAM

**CPU Fallback:**
- Model loading: ~10-15 seconds
- Analysis: ~3-5 seconds per image
- Memory usage: ~6-8GB RAM

## Technical Details

- **Model**: Google Gemma 3 4B Instruction-Tuned
- **Framework**: PyTorch with CUDA 11.8
- **Input Processing**: Images resized to 512px max for speed
- **Generation**: 100 token responses with KV cache
- **Camera**: Optimized settings with reduced buffer lag
- **Environment**: Shared conda 'vlm' environment

## Files

### Interface Applications
- `vlm_web.py` - Web interface (Flask + WebSocket)
- `vlm_standalone.py` - Desktop interface (Tkinter)  
- `vlm_modern.py` - Modern desktop interface (Dear PyGui)

### Launcher Scripts
- `run_web.sh` - Web interface launcher
- `run.sh` - Desktop interface launcher
- `run_modern.sh` - Modern desktop launcher
- `setup.sh` - Environment setup

### Templates & Config
- `templates/vlm_chat.html` - Web interface template
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Troubleshooting

**Model loading fails:**
- Check HuggingFace authentication
- Ensure Gemma 3 license is accepted
- Verify conda environment exists

**Camera not found:**
- Check `/dev/video0` exists
- Test with `cheese` or similar app

**Performance issues:**
- App auto-detects GPU vs CPU
- GPU gives ~3x faster performance
- Optimizations work on both GPU and CPU
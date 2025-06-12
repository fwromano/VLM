# VLM Video Chat

A clean, optimized vision-language model chat interface with real-time camera support.

## What This Does

Interactive video chat with Google Gemma 3 4B model - ask questions about what your camera sees and get intelligent responses in real-time.

## Features

- **Real-time Camera**: 640x480 @ 30 FPS optimized for performance
- **Interactive Chat**: Video call style interface with chat history
- **Quick Prompts**: 10 pre-configured buttons for common questions
- **GPU Optimized**: Explicit device mapping for RTX 5000 Ada (16GB)
- **Performance Tuned**: 512px input images, 100 token responses, KV cache
- **Smart UI**: Tooltips, timestamps, processing time display

## Quick Start

```bash
# 1. Setup (one-time)
./setup.sh

# 2. Authenticate with HuggingFace
huggingface-cli login

# 3. Accept Gemma license at:
#    https://huggingface.co/google/gemma-3-4b-it

# 4. Run
./run.sh
```

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

- `vlm_standalone.py` - Main application
- `run.sh` - Launcher script  
- `setup.sh` - Environment setup
- `requirements.txt` - Python dependencies
- `README.md` - This file

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
# VLM Video Chat

Real-time camera analysis using InternVL3-2B vision-language model with interactive chat interface.

## Features

- **Real-time Video**: 30 FPS camera stream, zero processing lag
- **Interactive Chat**: Custom text input for image questions
- **AI Vision Analysis**: InternVL3-2B model (2B parameters)
- **Multi-threaded**: Background AI processing, responsive GUI
- **GPU Accelerated**: RTX 5000 Ada optimized (16GB VRAM)
- **Modern Interface**: Video chat style with timestamps
- **Quick Prompts**: Pre-defined question buttons
- **Performance Metrics**: Processing time display

## Interface Layout

```
┌─────────────────────────────────────────────────────┐
│                VLM Video Chat                       │
├──────────────────────┬──────────────────────────────┤
│   Camera Feed        │      Chat Interface          │
│                      │                              │
│  [Live Video Stream] │   Chat History:              │
│                      │   [12:34:56] You: What do    │
│  GPU: InternVL3-2B   │   you see?                   │
│  Queue: 0/1          │   [12:34:57] VLM: I can see  │
│                      │   a desk with a computer...  │
│                      │                              │
│                      │   Type your question:        │
│                      │   [Text Input Box]           │
│                      │   [Send] [Quick Prompts...]  │
└──────────────────────┴──────────────────────────────┘
```

## Quick Start

### 1. Setup

```bash
cd VLM
chmod +x setup.sh
./setup.sh
```

### 2. Launch

```bash
./run_vlm.sh
```

Interface opens with live camera feed (left) and chat panel (right).

## Usage

1. Type questions in text input
2. Press Enter to send for AI analysis
3. View responses in chat history with timestamps
4. Use quick prompt buttons for common questions

### Controls
- **Enter**: Send message
- **Ctrl+Enter**: New line in input
- **Close Window**: Shutdown camera and model

### Sample Questions
- "What's on my desk?"
- "Count people in room"
- "Describe lighting conditions"
- "Identify safety hazards"
- "What colors are visible?"

## Technical Requirements

### Hardware
- **Recommended**: RTX 5000 Ada (16GB VRAM)
- **Minimum**: CUDA GPU (4GB+ VRAM)
- **Fallback**: CPU mode
- **Camera**: USB/built-in webcam

### Software
- Linux (Ubuntu tested)
- Python 3.8+
- CUDA 12.1+
- Conda

### Dependencies (auto-installed)
- PyTorch + CUDA
- Transformers
- OpenCV
- Tkinter
- InternVL3-2B model

## Project Structure

```
VLM/
├── README.md
├── vlm.py
├── run_vlm.sh
├── setup.sh
├── requirements.txt
└── CLAUDE.md
```

## Configuration

Auto-detects hardware:
- **GPU**: InternVL3-2B, bfloat16 precision
- **CPU**: InternVL3-2B, float32 precision

### Model
- OpenGVLab/InternVL3-2B-hf (2B parameters)
- Capabilities: Image understanding, object detection, scene description
- Language: English
- Max response: 150 tokens

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi                    # Verify drivers
nvcc --version               # Check CUDA
conda activate vlm
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Camera Issues
```bash
ls /dev/video*               # List cameras
python -c "import cv2; print('OK' if cv2.VideoCapture(0).isOpened() else 'Failed')"
```

### Model Issues
- First run: 4GB model download
- Out of memory: Use CPU mode

### GUI Issues
```bash
sudo apt-get install python3-tk    # Ubuntu/Debian
```

## Manual Installation

```bash
conda create -n vlm python=3.10 -y
conda activate vlm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.37.0 opencv-python pillow accelerate
sudo apt-get install python3-tk
python vlm.py
```

## Performance

### GPU Mode (RTX 5000 Ada)
- Model loading: 10-15s (first run)
- Analysis speed: 2-4s per question
- Video FPS: 30 (unaffected)
- Memory: 8GB VRAM

### CPU Mode
- Model loading: 30-60s (first run) 
- Analysis speed: 15-30s per question
- Video FPS: 30 (unaffected)
- Memory: 4GB RAM

## Dependencies

- InternVL3 (OpenGVLab)
- HuggingFace Transformers
- OpenCV
- Tkinter

## License

Educational and research use.

---

Run `./run_vlm.sh` to start.
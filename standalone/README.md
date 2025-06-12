# VLM Video Chat - Standalone

A clean, simple vision-language model (VLM) chat interface with real-time camera support.

## Features

- Real-time camera feed (30 FPS)
- Interactive chat interface
- Quick prompts for common questions
- GPU acceleration support (RTX 5000 Ada)
- Gemma 3 4B model integration

## Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- Webcam
- HuggingFace account with access to Gemma models

## Setup

1. **Install dependencies:**
   ```bash
   ./setup.sh
   ```

2. **Authenticate with HuggingFace:**
   ```bash
   huggingface-cli login
   ```
   Make sure your token has "Read access to gated repos" permission.

3. **Accept Gemma license:**
   Visit https://huggingface.co/google/gemma-3-4b-it and accept the license.

## Usage

```bash
./run.sh
```

## Interface

- **Text Input**: Type custom questions about what you see
- **Enter**: Send message
- **Ctrl+Enter**: New line in text input
- **Quick Prompts**: Click buttons for common questions
- **Chat History**: Scrollable conversation with timestamps

## Model

Currently configured to use Google's Gemma 3 4B instruction-tuned model.
Falls back to CPU if GPU is not available.
# VLM Standalone Interface

Standalone VLM video chat application with Tkinter GUI.

## Features
- Native desktop application using Tkinter
- Real-time camera feed with VLM analysis
- Text input for custom questions
- Chat history with timestamps
- Works offline once models are downloaded

## Setup

```bash
# Setup environment (run once)
chmod +x setup.sh && ./setup.sh

# Run application
./run_vlm.sh
```

## Requirements
- Python 3.10
- Conda/Miniconda
- CUDA-capable GPU (recommended)
- Tkinter (usually included with Python)
  - Ubuntu/Debian: `sudo apt-get install python3-tk`

## Usage
1. Start the application with `./run_vlm.sh`
2. Allow camera access when prompted
3. Type questions in the text box
4. Press Enter to analyze the current frame
5. View responses in the chat history

## Dependencies
See `requirements.txt` for the complete list of Python packages needed.
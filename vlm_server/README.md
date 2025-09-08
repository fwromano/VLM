VLM Server
==========

Persistent VLM microservice that keeps models in memory and serves:
- Vision Q&A over images (Gemma 3; InternVL 3 via config)
- Optional vLLM backend for fast text/vision where supported
- Audio transcription from MP3 and QA over transcripts

Why
- Drops per‑request cold start/model load (seconds → milliseconds)
- Eliminates temp image files and subprocess overhead
- Enables shared use across ROS2, demos, and web clients

Endpoints
- POST /v1/vision/analyze
  - Body: multipart (image file) or JSON {image_base64}
  - Params: question, model (default gemma-3-4b-it), backend (transformers|vllm), fast=true|false
  - Returns: {text, timings, model, backend}

- POST /v1/audio/transcribe
  - Body: multipart (mp3 file)
  - Returns: {transcript_id, text, duration, timings}

- POST /v1/audio/qa
  - Body: {transcript_id or text, question, model, backend}
  - Returns: {answer, timings}

Simple browser demo (no code needed)
- GET /demo/audio → HTML form to upload an MP3 and ask a question.
  - Does server-side transcription + QA and shows results.

Backends
- transformers (Linux/macOS, CUDA/MPS/CPU)
- vLLM (CUDA only; auto‑detected; falls back gracefully)

Models
- Default: Google Gemma 3 4B IT (vision)
- InternVL 3: configurable via config.json (requires proper HF ID and dependencies)

Platform Support
- Linux: CUDA → GPU; else CPU
- macOS: MPS (Apple Silicon) → GPU; else CPU. CUDA env vars are ignored.

Run
```bash
cd vlm_server
./run.sh  # uses conda env if available
# Server on http://0.0.0.0:8080
```

One-command launcher from repo root:
```bash
./serve.sh
```

Configure
- Edit vlm_server/config.json to change model IDs and defaults.

Client (Python)
```python
import requests, base64
img_b64 = base64.b64encode(open('image.jpg','rb').read()).decode()
r = requests.post('http://localhost:8080/v1/vision/analyze', json={
  'image_base64': img_b64,
  'question': 'What do you see?',
  'model': 'gemma-3-4b-it',
  'backend': 'transformers',
  'fast': False,
})
print(r.json()['text'])
```

Notes
- If vLLM import fails or no CUDA, the server auto‑disables vLLM.
- Audio features use faster-whisper if installed; otherwise disable with clear errors.
- Web and desktop UIs plus ROS2 node attempt to use this server automatically and fall back locally if unavailable.
 - On macOS, audio transcription runs on CPU (int8) by default and works out-of-the-box.

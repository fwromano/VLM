#!/usr/bin/env python3
import base64
import io
import json
import os
import time
import uuid
import platform
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

# Optional imports guarded at runtime
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

# Local
from . import vlm_backends

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config() -> Dict[str, Any]:
    cfg_path = os.path.join(BASE_DIR, 'config.json')
    with open(cfg_path, 'r') as f:
        return json.load(f)


def detect_device() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        'platform': platform.system(),
        'device': 'cpu',
        'cuda_available': False,
        'mps_available': False,
        'vllm_available': False,
    }

    # vLLM availability
    try:
        import vllm  # noqa: F401
        info['vllm_available'] = True
    except Exception:
        info['vllm_available'] = False

    if torch is None:
        return info

    # CUDA first (Linux/Windows)
    if torch.cuda.is_available():
        info['device'] = 'cuda'
        info['cuda_available'] = True
        return info

    # MPS for macOS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['device'] = 'mps'
        info['mps_available'] = True
        return info

    return info


app = FastAPI(title="VLM Server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CFG = load_config()
ENV = detect_device()
MANAGER = vlm_backends.ModelManager(CFG, ENV)


@app.get("/v1/health")
def health():
    return {
        'status': 'ok',
        'platform': ENV['platform'],
        'device': ENV['device'],
        'cuda': ENV['cuda_available'],
        'mps': ENV['mps_available'],
        'vllm': ENV['vllm_available'],
        'loaded_models': list(MANAGER.loaded_models.keys()),
    }


def _decode_image_from_request(
    image_file: Optional[UploadFile],
    image_base64: Optional[str],
):
    if Image is None:
        raise RuntimeError("Pillow not available")
    if image_file is not None:
        content = image_file.file.read()
        return Image.open(io.BytesIO(content)).convert('RGB')
    if image_base64:
        data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(data)).convert('RGB')
    raise ValueError("No image provided")


@app.post("/v1/vision/analyze")
async def analyze_image(
    image: Optional[UploadFile] = File(default=None),
    question: str = Form(default="What do you see?"),
    model: str = Form(default=None),
    backend: str = Form(default=None),
    fast: bool = Form(default=False),
    image_base64: Optional[str] = Form(default=None),
):
    """Analyze an image with a VLM.

    Supports multipart (image file) or JSON-like form with base64.
    """
    t0 = time.time()
    try:
        pil = _decode_image_from_request(image, image_base64)
    except Exception as e:
        return JSONResponse(status_code=400, content={'error': f'image decode failed: {e}'})

    selected_model = model or CFG.get('default_model', 'gemma-3-4b-it')
    selected_backend = backend or CFG.get('default_backend', 'transformers')

    # Respect capabilities
    if selected_backend == 'vllm' and not MANAGER.capabilities['vllm']:
        selected_backend = 'transformers'

    try:
        out = MANAGER.analyze_image(pil, question, selected_model, selected_backend, fast=fast)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})

    out['timings'] = {
        **out.get('timings', {}),
        'total': time.time() - t0,
    }
    return out


# In-memory transcripts (simple store)
TRANSCRIPTS: Dict[str, Dict[str, Any]] = {}


@app.post("/v1/audio/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe an MP3 with faster-whisper if available."""
    t0 = time.time()
    try:
        content = await audio.read()
        text, info = MANAGER.transcribe_mp3_bytes(content)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

    tid = str(uuid.uuid4())
    TRANSCRIPTS[tid] = {
        'text': text,
        'info': info,
        'created': time.time(),
        'filename': audio.filename,
    }
    return {
        'transcript_id': tid,
        'text': text,
        'duration': info.get('duration', None),
        'timings': {'total': time.time() - t0},
    }


@app.post("/v1/audio/qa")
async def qa_over_transcript(
    transcript_id: Optional[str] = Form(default=None),
    transcript_text: Optional[str] = Form(default=None),
    question: str = Form(...),
    model: Optional[str] = Form(default=None),
    backend: Optional[str] = Form(default=None),
):
    t0 = time.time()
    if not transcript_text and transcript_id:
        item = TRANSCRIPTS.get(transcript_id)
        if not item:
            return JSONResponse(status_code=404, content={'error': 'transcript not found'})
        transcript_text = item['text']
    if not transcript_text:
        return JSONResponse(status_code=400, content={'error': 'no transcript provided'})

    selected_model = model or CFG.get('default_model', 'gemma-3-4b-it')
    selected_backend = backend or CFG.get('default_backend', 'transformers')
    if selected_backend == 'vllm' and not MANAGER.capabilities['vllm']:
        selected_backend = 'transformers'

    try:
        answer = MANAGER.answer_over_text(transcript_text, question, selected_model, selected_backend)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
    return {
        'answer': answer,
        'timings': {'total': time.time() - t0},
        'model': selected_model,
        'backend': selected_backend,
    }


@app.get("/demo/audio", response_class=HTMLResponse)
def audio_demo_form():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html><head><meta charset='utf-8'><title>Audio QA Demo</title></head>
        <body style="font-family: sans-serif; padding: 20px;">
          <h2>Audio (MP3) Transcribe + QA</h2>
          <form action="/demo/audio" method="post" enctype="multipart/form-data">
            <div><label>MP3 File: <input type="file" name="audio" accept="audio/mpeg"></label></div>
            <div style="margin-top:10px;"><label>Question: <input type="text" name="question" size="60" placeholder="What is being discussed?"></label></div>
            <div style="margin-top:10px;"><button type="submit">Transcribe and Ask</button></div>
          </form>
        </body></html>
        """
    )


@app.post("/demo/audio", response_class=HTMLResponse)
async def audio_demo_submit(audio: UploadFile = File(...), question: str = Form(...)):
    try:
        content = await audio.read()
        transcript, info = MANAGER.transcribe_mp3_bytes(content)
        answer = MANAGER.answer_over_text(transcript, question, CFG.get('default_model','gemma-3-4b-it'), CFG.get('default_backend','transformers'))
        page = f"""
        <!DOCTYPE html>
        <html><head><meta charset='utf-8'><title>Audio QA Result</title></head>
        <body style='font-family: sans-serif; padding: 20px;'>
          <h2>Audio (MP3) Transcribe + QA</h2>
          <p><b>File:</b> {audio.filename}</p>
          <p><b>Duration:</b> {info.get('duration','?')} s</p>
          <h3>Transcript</h3>
          <pre style='background:#f6f6f6; padding:10px; white-space:pre-wrap;'>{transcript}</pre>
          <h3>Question</h3>
          <p>{question}</p>
          <h3>Answer</h3>
          <p style='background:#eef; padding:10px;'>{answer}</p>
          <p><a href='/demo/audio'>Back</a></p>
        </body></html>
        """
        return HTMLResponse(page)
    except Exception as e:
        return HTMLResponse(f"<pre>Error: {e}</pre>", status_code=500)

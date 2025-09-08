#!/usr/bin/env python
import os
import uuid
import json
import time
import shutil
import platform
import base64 as _b64
from io import BytesIO
from typing import Optional

from flask import Flask, render_template, request, jsonify, send_from_directory

import cv2
import numpy as np
from PIL import Image
import urllib.request as _urlreq
import urllib.parse as _urlparse


app = Flask(__name__)
app.config['SECRET_KEY'] = 'vlm_video_qa_secret'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


def server_url() -> str:
    return os.environ.get('VLM_SERVER_URL', 'http://127.0.0.1:8080').rstrip('/')


def call_vlm_server(image_bgr: np.ndarray, question: str, model: Optional[str], backend: Optional[str], fast: bool) -> dict:
    # Encode frame to JPEG and send to server
    ok, buf = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError('Failed to encode frame to JPEG')
    payload = {
        'image_base64': _b64.b64encode(buf.tobytes()).decode('utf-8'),
        'question': question,
    }
    if model:
        payload['model'] = model
    if backend:
        payload['backend'] = backend
    if fast:
        payload['fast'] = True
    data = _urlparse.urlencode(payload).encode('utf-8')
    req = _urlreq.Request(server_url() + '/v1/vision/analyze', data=data)
    with _urlreq.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read().decode('utf-8'))
    return out


def extract_frame(video_path: str, timestamp_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Could not open video')
    # Try seek by time
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp_sec * 1000.0))
    ret, frame = cap.read()
    if not ret or frame is None:
        # Fallback: compute frame index
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = int(max(0, timestamp_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError('Failed to read frame at requested time')
    cap.release()
    return frame


@app.route('/video_qa')
def page_video_qa():
    return render_template('vlm_video_qa.html')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['video']
    if f.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ('.mp4', '.mov', '.webm', '.mkv', '.avi'):
        return jsonify({'error': 'Unsupported format'}), 400
    vid_id = str(uuid.uuid4())
    save_name = f"{vid_id}{ext}"
    path = os.path.join(UPLOAD_DIR, save_name)
    f.save(path)
    return jsonify({'video_id': vid_id, 'video_url': f"/videos/{save_name}"})


@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json(force=True)
        video_url = data.get('video_url')
        question = data.get('question', '').strip()
        ts = float(data.get('timestamp', 0))
        model_choice = data.get('model', 'gemma-3-4b-it')
        fast = bool(data.get('fast', False))
        backend = data.get('backend')  # typically transformers
        if not video_url or not question:
            return jsonify({'error': 'Missing video or question'}), 400
        # Map URL to file path under uploads
        # Expect format /videos/<filename>
        filename = os.path.basename(video_url)
        video_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video not found'}), 404

        frame = extract_frame(video_path, ts)
        out = call_vlm_server(frame, question, model_choice, backend, fast)
        if 'error' in out:
            return jsonify({'error': out['error']}), 500
        # Return response with a small preview frame
        disp_h = 240
        h, w = frame.shape[:2]
        disp_w = int(w * (disp_h / h)) if h else 320
        preview = cv2.resize(frame, (disp_w, disp_h))
        ok, pbuf = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        preview_b64 = _b64.b64encode(pbuf.tobytes()).decode('utf-8') if ok else None
        return jsonify({
            'answer': out.get('text', ''),
            'processing': out.get('timings', {}),
            'frame': preview_b64,
            'frame_w': disp_w,
            'frame_h': disp_h,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run development server
    app.run(host='0.0.0.0', port=5050, debug=False)


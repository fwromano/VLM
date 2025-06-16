#!/usr/bin/env python
# vlm_streaming_working.py - Working streaming version with all features

import os
import sys
import warnings
import threading
import queue
import time
import base64
import numpy as np
from dataclasses import dataclass
from typing import Optional
from io import BytesIO
warnings.filterwarnings('ignore')

# Set CUDA environment before imports
os.environ.update({
    'CUDA_HOME': '/usr',
    'CUDA_ROOT': '/usr', 
    'CUDA_PATH': '/usr',
    'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu',
    'CUDA_VISIBLE_DEVICES': '0',
    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'
})

print("VLM Video Chat - Streaming Working Version")
print("==========================================")

try:
    import torch
    import cv2
    from flask import Flask, render_template, request, jsonify
    from flask_socketio import SocketIO, emit
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor, TextIteratorStreamer
    from PIL import Image
except ImportError as e:
    print(f"Missing library: {e}")
    sys.exit(1)

@dataclass
class AnalysisRequest:
    image: np.ndarray
    question: str
    timestamp: float

class VLMProcessor:
    def __init__(self, gpu_available=False):
        self.gpu_available = gpu_available
        self.device = torch.device("cuda" if gpu_available else "cpu")
        self.model = None
        self.processor = None
        self.is_loaded = False
        
    def load_model(self, model_id="google/gemma-3-4b-it"):
        try:
            print("Loading Gemma 3 4B model...")
            
            # Load fresh without cache to avoid corruption
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.gpu_available else torch.float32,
                device_map={"": 0} if self.gpu_available else None,
                low_cpu_mem_usage=True
            )
            
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            
            # Disable problematic features for stability
            self.model.generation_config.do_sample = False
            self.model.generation_config.use_cache = False  # Disable to save memory
            
            if not self.gpu_available:
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False
    
    def process_image_streaming(self, image: np.ndarray, question: str, socketio, conversation_id: str):
        if not self.is_loaded:
            socketio.emit('stream_error', {
                'conversation_id': conversation_id,
                'error': "Model not loaded"
            })
            return
        
        try:
            # Convert to PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Resize to 448x448 for memory efficiency
            target_size = 448
            pil_image = pil_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Create message format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question}
                ]
            }]
            
            # Signal streaming start
            socketio.emit('stream_start', {'conversation_id': conversation_id})
            
            # Process
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[pil_image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Setup streaming
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                timeout=60,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 150,
                "do_sample": False,
                "use_cache": False,
                "streamer": streamer,
                "pad_token_id": self.processor.tokenizer.eos_token_id
            }
            
            # Generate in thread
            def generate():
                with torch.no_grad():
                    self.model.generate(**generation_kwargs)
            
            generation_thread = threading.Thread(target=generate)
            generation_thread.start()
            
            # Stream tokens with simple buffering
            buffer = []
            for token in streamer:
                buffer.append(token)
                
                # Send when buffer has 2-3 tokens or hits sentence ending
                if len(buffer) >= 3 or token.strip().endswith(('.', '!', '?', '\n')):
                    token_batch = ''.join(buffer)
                    if token_batch.strip():
                        socketio.emit('stream_token', {
                            'conversation_id': conversation_id,
                            'token': token_batch
                        })
                    buffer = []
                    time.sleep(0.03)  # Small delay for natural feel
            
            # Send remaining tokens
            if buffer:
                final_batch = ''.join(buffer)
                if final_batch.strip():
                    socketio.emit('stream_token', {
                        'conversation_id': conversation_id,
                        'token': final_batch
                    })
            
            generation_thread.join()
            socketio.emit('stream_complete', {'conversation_id': conversation_id})
            
        except Exception as e:
            print(f"Streaming error: {e}")
            socketio.emit('stream_error', {
                'conversation_id': conversation_id,
                'error': str(e)
            })

class VLMWebApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'vlm_streaming'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup GPU
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU mode")
        
        self.processor = VLMProcessor(self.gpu_available)
        
        # Camera and state
        self.cap = None
        self.current_frame = None
        self.conversation_counter = 0
        self.is_processing = False
        self.is_running = False
        
        self.setup_routes()
        
    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("Camera ready")
            return True
        return False
        
    def video_worker(self):
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    
                    # Stream video with aspect ratio preservation
                    display_height = 480
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    display_width = int(display_height * aspect_ratio)
                    
                    frame_resized = cv2.resize(frame, (display_width, display_height))
                    _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    self.socketio.emit('video_frame', {'frame': frame_base64})
            time.sleep(1/30)
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('vlm_chat_optimized.html', 
                                 gpu_available=self.gpu_available,
                                 config={'model_name': 'Gemma-3-4B-Streaming'})
        
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            emit('status_update', {'status': 'ready'})
        
        @self.socketio.on('send_message')
        def handle_message(data):
            if self.is_processing:
                emit('error', {'message': 'Processing...'})
                return
                
            question = data.get('message', '').strip()
            if not question or self.current_frame is None:
                return
            
            self.conversation_counter += 1
            conversation_id = f"conv_{self.conversation_counter}"
            
            # Capture frame for chat
            captured_frame = self.current_frame.copy()
            frame_resized = cv2.resize(captured_frame, (356, 200))
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send user message with conversation structure
            emit('chat_message', {
                'sender': 'You',
                'message': question,
                'timestamp': time.strftime("%H:%M:%S"),
                'frame': frame_base64,
                'frame_format': 'jpeg',
                'frame_width': 356,
                'frame_height': 200,
                'conversation_id': conversation_id
            }, broadcast=True)
            
            self.is_processing = True
            emit('status_update', {'status': 'processing'})
            
            # Process in background
            def process():
                try:
                    self.processor.process_image_streaming(
                        captured_frame, question, self.socketio, conversation_id
                    )
                finally:
                    self.is_processing = False
                    self.socketio.emit('status_update', {'status': 'ready'})
            
            threading.Thread(target=process, daemon=True).start()
    
    def run(self):
        if not self.processor.load_model():
            print("Failed to load model")
            return
            
        if not self.setup_camera():
            print("Failed to setup camera")
            return
        
        # Start video thread
        self.is_running = True
        threading.Thread(target=self.video_worker, daemon=True).start()
        
        print("Streaming VLM interface ready at http://localhost:5000")
        print("Features: Real-time streaming, collapsible images, markdown support")
        
        self.socketio.run(self.app, host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    app = VLMWebApp()
    app.run()
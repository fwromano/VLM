#!/usr/bin/env python
# vlm_multi_model.py - Multi-model VLM streaming with Gemma-3N-E4B-it support

import os
import sys
import warnings
import threading
import queue
import time
import base64
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
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

print("VLM Video Chat - Multi-Model Version")
print("=====================================")

try:
    import torch
    import cv2
    from flask import Flask, render_template, request, jsonify
    from flask_socketio import SocketIO, emit
    from transformers import AutoModel, AutoProcessor, TextIteratorStreamer, AutoModelForCausalLM
    from PIL import Image
    import json
    import io
    import base64 as _b64
    import urllib.request as _urlreq
    import urllib.error as _urlerr
    import urllib.parse as _urlparse
except ImportError as e:
    print(f"Missing library: {e}")
    sys.exit(1)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

@dataclass
class AnalysisRequest:
    image: np.ndarray
    question: str
    timestamp: float
    model_key: str = None

class VLMProcessor:
    def __init__(self, gpu_available=False):
        self.gpu_available = gpu_available
        self.device = torch.device("cuda" if gpu_available else "cpu")
        self.models = {}
        self.processors = {}
        self.current_model_key = None
        
    def load_model(self, model_key: str) -> bool:
        """Load a specific model by key"""
        if model_key not in CONFIG['MODELS']:
            print(f"Unknown model key: {model_key}")
            return False
            
        if model_key == self.current_model_key and model_key in self.models:
            print(f"Model {model_key} already loaded")
            return True
            
        try:
            # Clear previous model if different
            if self.current_model_key and self.current_model_key != model_key:
                self.unload_current_model()
            
            model_info = CONFIG['MODELS'][model_key]
            model_id = model_info['id']
            print(f"Loading {model_info['name']} ({model_id})...")
            
            # Try different loading strategies based on model
            try:
                # First try AutoModel with trust_remote_code for custom model classes
                self.models[model_key] = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16 if self.gpu_available else torch.float32,
                    device_map="auto" if self.gpu_available else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True  # This allows loading custom model classes
                )
            except Exception as e1:
                print(f"AutoModel failed: {e1}, trying AutoModelForCausalLM...")
                try:
                    # Fallback to causal LM for some models
                    self.models[model_key] = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16 if self.gpu_available else torch.float32,
                        device_map="auto" if self.gpu_available else None,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                except Exception as e2:
                    # Last resort: try pipeline
                    from transformers import pipeline
                    print(f"AutoModelForCausalLM failed: {e2}, trying pipeline...")
                    pipe = pipeline(
                        "image-text-to-text",
                        model=model_id,
                        device=0 if self.gpu_available else -1,
                        torch_dtype=torch.bfloat16 if self.gpu_available else torch.float32,
                    )
                    self.models[model_key] = pipe.model
                    self.processors[model_key] = pipe.tokenizer
                    return True  # Skip normal processor loading
            
            self.processors[model_key] = AutoProcessor.from_pretrained(
                model_id, 
                trust_remote_code=True
            )
            
            # Configure generation settings if available
            if hasattr(self.models[model_key], 'generation_config') and self.models[model_key].generation_config is not None:
                self.models[model_key].generation_config.do_sample = False
                if hasattr(self.models[model_key].generation_config, 'use_cache'):
                    self.models[model_key].generation_config.use_cache = True
            
            if not self.gpu_available:
                self.models[model_key] = self.models[model_key].to(self.device)
            
            self.current_model_key = model_key
            print(f"Model {model_info['name']} loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_current_model(self):
        """Unload current model to free memory"""
        if self.current_model_key:
            if self.current_model_key in self.models:
                del self.models[self.current_model_key]
            if self.current_model_key in self.processors:
                del self.processors[self.current_model_key]
            torch.cuda.empty_cache() if self.gpu_available else None
            print(f"Unloaded model: {self.current_model_key}")
            self.current_model_key = None
    
    def process_image_streaming(self, image: np.ndarray, question: str, 
                              socketio, conversation_id: str, model_key: str = None):
        # Try server first; map model_key to server model names when possible
        server_url = os.environ.get('VLM_SERVER_URL', 'http://127.0.0.1:8080')
        if server_url:
            try:
                server_model = None
                if model_key:
                    mk = model_key.lower()
                    if mk.startswith('gemma-3n'):
                        server_model = 'gemma-3n-e4b-it'
                    elif 'internvl' in mk:
                        server_model = 'internvl-3'
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                pil_image.save(buf, format='JPEG', quality=90)
                payload = {
                    'image_base64': _b64.b64encode(buf.getvalue()).decode('utf-8'),
                    'question': question,
                }
                if server_model:
                    payload['model'] = server_model
                if os.environ.get('VLM_SERVER_BACKEND'):
                    payload['backend'] = os.environ['VLM_SERVER_BACKEND']
                if os.environ.get('VLM_SERVER_FAST', '0') in ('1','true','TRUE'):
                    payload['fast'] = True
                data = _urlparse.urlencode(payload).encode('utf-8')
                req = _urlreq.Request(server_url.rstrip('/') + '/v1/vision/analyze', data=data)
                socketio.emit('stream_start', {'conversation_id': conversation_id, 'model': CONFIG['MODELS'].get(model_key, {}).get('name', '')})
                with _urlreq.urlopen(req, timeout=10) as resp:
                    out = json.loads(resp.read().decode('utf-8'))
                    text = out.get('text', '')
                    if text:
                        socketio.emit('stream_token', {'conversation_id': conversation_id, 'token': text})
                        socketio.emit('stream_complete', {'conversation_id': conversation_id})
                        return
            except Exception:
                pass

        # Use current model or default if not specified
        if not model_key:
            model_key = self.current_model_key or CONFIG['DEFAULT_MODEL']
        
        # Load model if needed
        if model_key != self.current_model_key:
            if not self.load_model(model_key):
                socketio.emit('stream_error', {
                    'conversation_id': conversation_id,
                    'error': f"Failed to load model {model_key}"
                })
                return
        
        if not self.current_model_key or self.current_model_key not in self.models:
            socketio.emit('stream_error', {
                'conversation_id': conversation_id,
                'error': "No model loaded"
            })
            return
        
        try:
            model = self.models[self.current_model_key]
            processor = self.processors[self.current_model_key]
            
            # Convert to PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Resize for efficiency
            target_size = CONFIG.get('MODEL_INPUT_SIZE', 448)
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
            socketio.emit('stream_start', {
                'conversation_id': conversation_id,
                'model': CONFIG['MODELS'][self.current_model_key]['name']
            })
            
            # Process based on model type
            try:
                # Try standard chat template first
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=text, images=[pil_image], return_tensors="pt")
            except Exception:
                # Fallback for models without chat template
                inputs = processor(text=question, images=[pil_image], return_tensors="pt")
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Setup streaming
            streamer = TextIteratorStreamer(
                processor.tokenizer,
                timeout=60,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs = {
                **inputs,
                "max_new_tokens": CONFIG.get('MAX_NEW_TOKENS', 150),
                "do_sample": False,
                "use_cache": False,
                "streamer": streamer,
                "pad_token_id": processor.tokenizer.eos_token_id
            }
            
            # Generate in thread
            def generate():
                with torch.no_grad():
                    model.generate(**generation_kwargs)
            
            generation_thread = threading.Thread(target=generate)
            generation_thread.start()
            
            # Stream tokens with buffering
            buffer = []
            for token in streamer:
                buffer.append(token)
                
                if len(buffer) >= 3 or token.strip().endswith(('.', '!', '?', '\n')):
                    token_batch = ''.join(buffer)
                    if token_batch.strip():
                        socketio.emit('stream_token', {
                            'conversation_id': conversation_id,
                            'token': token_batch
                        })
                    buffer = []
                    time.sleep(0.03)
            
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
        self.app.config['SECRET_KEY'] = CONFIG.get('SECRET_KEY', 'vlm_multi_model')
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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['CAMERA_WIDTH'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['CAMERA_HEIGHT'])
            self.cap.set(cv2.CAP_PROP_FPS, CONFIG['CAMERA_FPS'])
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
                    _, buffer = cv2.imencode('.jpg', frame_resized, 
                                           [cv2.IMWRITE_JPEG_QUALITY, CONFIG['JPEG_QUALITY']])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    self.socketio.emit('video_frame', {'frame': frame_base64})
            time.sleep(1/CONFIG['CAMERA_FPS'])
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('vlm_chat_multi_model.html', 
                                 gpu_available=self.gpu_available,
                                 config=CONFIG,
                                 models=CONFIG['MODELS'],
                                 default_model=CONFIG['DEFAULT_MODEL'])
        
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            # Send available models
            emit('models_available', {
                'models': CONFIG['MODELS'],
                'current': self.processor.current_model_key or CONFIG['DEFAULT_MODEL']
            })
            emit('status_update', {'status': 'ready'})
        
        @self.socketio.on('switch_model')
        def handle_switch_model(data):
            model_key = data.get('model_key')
            if not model_key or model_key not in CONFIG['MODELS']:
                emit('error', {'message': 'Invalid model key'})
                return
            
            emit('status_update', {'status': 'loading_model'})
            
            def load():
                success = self.processor.load_model(model_key)
                if success:
                    self.socketio.emit('model_switched', {
                        'model_key': model_key,
                        'model_info': CONFIG['MODELS'][model_key]
                    })
                    self.socketio.emit('status_update', {'status': 'ready'})
                else:
                    self.socketio.emit('error', {'message': f'Failed to load {model_key}'})
                    self.socketio.emit('status_update', {'status': 'ready'})
            
            threading.Thread(target=load, daemon=True).start()
        
        @self.socketio.on('send_message')
        def handle_message(data):
            if self.is_processing:
                emit('error', {'message': 'Processing...'})
                return
                
            question = data.get('message', '').strip()
            model_key = data.get('model_key')  # Optional model override
            
            if not question or self.current_frame is None:
                return
            
            self.conversation_counter += 1
            conversation_id = f"conv_{self.conversation_counter}"
            
            # Capture frame for chat
            captured_frame = self.current_frame.copy()
            frame_resized = cv2.resize(captured_frame, (356, CONFIG['CHAT_FRAME_HEIGHT']))
            _, buffer = cv2.imencode('.jpg', frame_resized, 
                                   [cv2.IMWRITE_JPEG_QUALITY, CONFIG['WEBP_QUALITY']])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send user message
            emit('chat_message', {
                'sender': 'You',
                'message': question,
                'timestamp': time.strftime("%H:%M:%S"),
                'frame': frame_base64,
                'frame_format': 'jpeg',
                'frame_width': 356,
                'frame_height': CONFIG['CHAT_FRAME_HEIGHT'],
                'conversation_id': conversation_id
            }, broadcast=True)
            
            self.is_processing = True
            emit('status_update', {'status': 'processing'})
            
            # Process in background
            def process():
                try:
                    self.processor.process_image_streaming(
                        captured_frame, question, self.socketio, 
                        conversation_id, model_key
                    )
                finally:
                    self.is_processing = False
                    self.socketio.emit('status_update', {'status': 'ready'})
            
            threading.Thread(target=process, daemon=True).start()
    
    def run(self):
        # Load default model
        if not self.processor.load_model(CONFIG['DEFAULT_MODEL']):
            print("Failed to load default model")
            return
            
        if not self.setup_camera():
            print("Failed to setup camera")
            return
        
        # Start video thread
        self.is_running = True
        threading.Thread(target=self.video_worker, daemon=True).start()
        
        print(f"\nMulti-Model VLM interface ready at http://localhost:{CONFIG['PORT']}")
        print(f"Available models: {', '.join(CONFIG['MODELS'].keys())}")
        print("Features: Model switching, real-time streaming, markdown support")
        
        self.socketio.run(self.app, host=CONFIG['HOST'], port=CONFIG['PORT'], debug=False)

if __name__ == "__main__":
    app = VLMWebApp()
    app.run()

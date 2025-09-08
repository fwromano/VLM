#!/usr/bin/env python
# vlm_web.py - vLLM Web Interface for High-Performance VLM

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
import platform as _platform
if _platform.system() != 'Darwin':
    os.environ.update({
        'CUDA_HOME': '/usr',
        'CUDA_ROOT': '/usr', 
        'CUDA_PATH': '/usr',
        'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu',
        'CUDA_VISIBLE_DEVICES': '0',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'
    })

print("vLLM VLM Video Chat - Web Interface")
print("===================================")

try:
    import torch
    import cv2
    from flask import Flask, render_template, request, jsonify
    from flask_socketio import SocketIO, emit
    from PIL import Image
    import json
    import io
    import base64 as _b64
    import urllib.request as _urlreq
    import urllib.error as _urlerr
    import urllib.parse as _urlparse
except ImportError as e:
    print(f"Missing library: {e}")
    print("Install with: pip install flask flask-socketio")
    # Continue; we can operate in server-first mode

# Optional vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

@dataclass
class AnalysisRequest:
    image: np.ndarray
    question: str
    timestamp: float
    chat_frame: str = None
    chat_frame_width: int = None
    chat_frame_height: int = None

@dataclass
class AnalysisResponse:
    question: str
    response: str
    timestamp: float
    processing_time: float

class vLLMProcessor:
    """High-performance VLM processor using vLLM"""
    
    def __init__(self, gpu_available=False):
        self.gpu_available = gpu_available
        self.device_count = torch.cuda.device_count() if gpu_available else 0
        self.llm = None
        self.sampling_params = None
        self.is_loaded = False
        self.current_model_id = None
        
    def load_model(self, model_id="google/gemma-3-4b-it"):
        """Load the VLM model using vLLM"""
        try:
            # Clear previous model if switching
            if self.current_model_id and self.current_model_id != model_id:
                print(f"Switching from {self.current_model_id} to {model_id}...")
                print("Cleaning up previous model...")
                
                if self.llm:
                    del self.llm
                    self.llm = None
                
                import gc
                gc.collect()
                
                if self.gpu_available:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                
                time.sleep(1)
                self.is_loaded = False
                self.current_model_id = None
                print("Previous model cleaned up")
            
            # Skip if already loaded
            if self.current_model_id == model_id and self.is_loaded:
                return True
            
            model_size = "4B" if "4b" in model_id else "12B"
            print(f"Loading Gemma 3 {model_size} model with vLLM...")
            
            # Configure vLLM for optimal performance with multimodal support
            vllm_config = {
                "model": model_id,
                "tensor_parallel_size": 1,  # Single GPU
                "gpu_memory_utilization": 0.75,  # Use 75% of VRAM for safety with images
                "max_model_len": 4096,  # Increased for multimodal inputs
                "enforce_eager": False,  # Use CUDA graphs for speed
                "trust_remote_code": True,  # Required for Gemma 3
                "limit_mm_per_prompt": {"image": 1},  # Limit to 1 image per prompt
            }
            
            # Add quantization for 12B model if needed
            if "12b" in model_id.lower():
                print("Using quantization for 12B model...")
                vllm_config["quantization"] = "awq"  # Use AWQ quantization
            
            if not VLLM_AVAILABLE:
                raise RuntimeError("vLLM not available on this platform; use server-first mode or install vLLM on Linux/CUDA")
            print("Initializing vLLM engine...")
            self.llm = LLM(**vllm_config)
            
            # Configure sampling parameters for consistent responses
            self.sampling_params = SamplingParams(
                temperature=0.1,  # Low temperature for consistency
                top_p=0.9,
                max_tokens=256,  # Reasonable response length
                stop=["<eos>", "</s>", "<end>"],  # Stop tokens
            )
            
            self.is_loaded = True
            self.current_model_id = model_id
            print(f"vLLM model {model_size} loaded successfully")
            
            # Show memory usage
            if self.gpu_available:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            return True
            
        except Exception as e:
            print(f"vLLM model loading failed: {e}")
            if "out of memory" in str(e).lower():
                print("Note: Try reducing gpu_memory_utilization or use smaller model")
            return False
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def process_image(self, image: np.ndarray, question: str) -> str:
        """Process image (server-first, then local vLLM fallback)"""
        # Try server first
        server_url = os.environ.get('VLM_SERVER_URL', 'http://127.0.0.1:8080')
        if server_url:
            try:
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                pil_image.save(buf, format='JPEG', quality=90)
                payload = {
                    'image_base64': _b64.b64encode(buf.getvalue()).decode('utf-8'),
                    'question': question,
                }
                if os.environ.get('VLM_SERVER_MODEL'):
                    payload['model'] = os.environ['VLM_SERVER_MODEL']
                if os.environ.get('VLM_SERVER_BACKEND'):
                    payload['backend'] = os.environ['VLM_SERVER_BACKEND']
                if os.environ.get('VLM_SERVER_FAST', '0') in ('1','true','TRUE'):
                    payload['fast'] = True
                data = _urlparse.urlencode(payload).encode('utf-8')
                req = _urlreq.Request(server_url.rstrip('/') + '/v1/vision/analyze', data=data)
                with _urlreq.urlopen(req, timeout=10) as resp:
                    out = json.loads(resp.read().decode('utf-8'))
                    if 'text' in out:
                        return out['text']
                    if 'error' in out:
                        return f"[Server error] {out['error']}"
            except Exception:
                pass

        if not self.is_loaded:
            return "Model not loaded"
        
        try:
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Resize image to match Gemma 3's expected 896x896 resolution
            target_size = 896
            width, height = pil_image.size
            if width != target_size or height != target_size:
                max_dim = max(width, height)
                square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
                paste_x = (max_dim - width) // 2
                paste_y = (max_dim - height) // 2
                square_img.paste(pil_image, (paste_x, paste_y))
                pil_image = square_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            try:
                # For vLLM with Gemma 3, use proper multimodal input format
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},  # Pass PIL Image directly
                            {"type": "text", "text": question}
                        ]
                    }
                ]
                
                # Generate response using vLLM's chat format
                outputs = self.llm.chat(conversation, sampling_params=self.sampling_params)
                response = outputs[0].outputs[0].text.strip()
                
                return response
                
            except Exception as chat_error:
                # If chat method fails, try alternative approach
                try:
                    # Fallback: use generate with a simple text prompt
                    prompt = f"User: {question}\nAssistant:"
                    outputs = self.llm.generate([prompt], self.sampling_params)
                    response = outputs[0].outputs[0].text.strip()
                    return f"[Text-only response] {response}"
                    
                except Exception as fallback_error:
                    return f"vLLM error: {str(chat_error)} | Fallback error: {str(fallback_error)}"
            
        except Exception as e:
            return f"vLLM analysis error: {str(e)}"

class VLMWebApp:
    """Web-based VLM Video Chat Interface using vLLM backend"""
    
    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'vllm_vlm_video_chat_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize components
        self.setup_gpu()
        self.processor = vLLMProcessor(self.gpu_available)
        
        # Threading components
        self.request_queue = queue.Queue(maxsize=2)  # Slightly larger queue for vLLM
        self.response_queue = queue.Queue()
        self.processing_thread = None
        self.video_thread = None
        self.is_running = False
        self.is_processing = False
        
        # Camera
        self.cap = None
        self.current_frame = None
        
        # UI state
        self.selected_model = "4B"
        
        # Available models for vLLM
        self.available_models = {
            "4B": "google/gemma-3-4b-it",
            "12B": "google/gemma-3-12b-it"  # vLLM can handle quantization automatically
        }
        
        # Setup routes and socketio handlers
        self.setup_routes()
        
    def setup_gpu(self):
        """Test GPU availability"""
        self.gpu_available = False
        self.device = torch.device('cpu')
        
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                test_tensor = torch.tensor([1.0], device='cuda:0')
                self.device = torch.device('cuda:0')
                self.gpu_available = True
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
            else:
                print("Using CPU mode (not recommended for vLLM)")
        except Exception as e:
            print(f"GPU test failed, using CPU: {e}")
    
    def setup_camera(self):
        """Initialize camera"""
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Camera not found")
            return False
        
        # Optimize camera settings for Gemma 3
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera ready: {width}x{height}")
        
        return True
    
    def video_stream_worker(self):
        """Worker thread for video streaming"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    
                    # Resize for web display
                    display_height = 480
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    display_width = int(display_height * aspect_ratio)
                    
                    frame_resized = cv2.resize(frame, (display_width, display_height))
                    
                    # Encode to base64 for web transmission
                    _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit to all connected clients
                    self.socketio.emit('video_frame', {'frame': frame_base64})
            
            time.sleep(1/30)  # 30 FPS
    
    def processing_worker(self):
        """Worker thread for vLLM processing"""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=1.0)
                
                start_time = time.time()
                response_text = self.processor.process_image(request.image, request.question)
                processing_time = time.time() - start_time
                
                response = AnalysisResponse(
                    question=request.question,
                    response=response_text,
                    timestamp=time.time(),
                    processing_time=processing_time
                )
                
                # Emit response to all clients with the captured frame
                response_data = {
                    'response': response.response,
                    'processing_time': response.processing_time,
                    'timestamp': time.strftime("%H:%M:%S", time.localtime(response.timestamp))
                }
                
                # Add frame info if available
                if hasattr(request, 'chat_frame') and request.chat_frame:
                    response_data.update({
                        'frame': request.chat_frame,
                        'frame_width': request.chat_frame_width,
                        'frame_height': request.chat_frame_height
                    })
                
                self.socketio.emit('analysis_response', response_data)
                
                self.is_processing = False
                self.socketio.emit('status_update', {'status': 'ready'})
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                self.socketio.emit('analysis_response', {
                    'response': f"Processing error: {str(e)}",
                    'processing_time': 0,
                    'timestamp': time.strftime("%H:%M:%S")
                })
                self.is_processing = False
                continue
    
    def setup_routes(self):
        """Setup Flask routes and SocketIO handlers"""
        
        @self.app.route('/')
        def index():
            return render_template('vlm_chat.html', 
                                 gpu_available=self.gpu_available,
                                 vllm_available=VLLM_AVAILABLE)
        
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            emit('status_update', {
                'status': 'ready',
                'model': self.selected_model,
                'device': 'GPU (vLLM)' if self.gpu_available else 'CPU (vLLM)',
                'backend': 'vLLM'
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')
        
        @self.socketio.on('send_message')
        def handle_message(data):
            if self.is_processing:
                emit('error', {'message': 'Still processing previous request...'})
                return
            
            question = data.get('message', '').strip()
            if not question:
                return
            
            if self.current_frame is None:
                emit('error', {'message': 'No camera frame available'})
                return
            
            # Capture and encode the current frame for display in chat
            captured_frame = self.current_frame.copy()
            
            # Resize frame for chat display (smaller for efficiency)
            chat_height = 200
            aspect_ratio = captured_frame.shape[1] / captured_frame.shape[0]
            chat_width = int(chat_height * aspect_ratio)
            frame_resized = cv2.resize(captured_frame, (chat_width, chat_height))
            
            # Encode to base64 for chat display
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Add user message to chat with captured frame
            emit('chat_message', {
                'sender': 'You',
                'message': question,
                'timestamp': time.strftime("%H:%M:%S"),
                'frame': frame_base64,
                'frame_width': chat_width,
                'frame_height': chat_height
            }, broadcast=True)
            
            # Show processing
            emit('chat_message', {
                'sender': 'vLLM',
                'message': 'Analyzing with vLLM...',
                'timestamp': time.strftime("%H:%M:%S"),
                'processing': True,
                'frame': frame_base64,
                'frame_width': chat_width,
                'frame_height': chat_height
            }, broadcast=True)
            
            emit('status_update', {'status': 'processing'}, broadcast=True)
            self.is_processing = True
            
            # Add to processing queue with the full resolution frame
            try:
                request = AnalysisRequest(
                    image=captured_frame,
                    question=question,
                    timestamp=time.time()
                )
                # Store the chat frame for the response
                request.chat_frame = frame_base64
                request.chat_frame_width = chat_width
                request.chat_frame_height = chat_height
                self.request_queue.put_nowait(request)
            except queue.Full:
                emit('error', {'message': 'Processing queue full, please wait...'})
                self.is_processing = False
        
        @self.socketio.on('change_model')
        def handle_model_change(data):
            new_model = data.get('model')
            if new_model == self.selected_model and self.processor.is_loaded:
                return
            
            if new_model not in self.available_models:
                emit('error', {'message': f'Model {new_model} not available'})
                return
            
            self.selected_model = new_model
            model_id = self.available_models[new_model]
            
            emit('status_update', {'status': f'loading_{new_model}'}, broadcast=True)
            
            # Load model in background
            def load_model():
                success = self.processor.load_model(model_id)
                if success:
                    self.socketio.emit('status_update', {
                        'status': 'ready', 
                        'model': new_model,
                        'backend': 'vLLM'
                    })
                    self.socketio.emit('chat_message', {
                        'sender': 'System',
                        'message': f'Successfully loaded Gemma 3 {new_model} model with vLLM',
                        'timestamp': time.strftime("%H:%M:%S"),
                        'system': True
                    })
                else:
                    self.socketio.emit('status_update', {'status': 'error'})
                    self.socketio.emit('chat_message', {
                        'sender': 'System',
                        'message': f'Failed to load {new_model} model with vLLM',
                        'timestamp': time.strftime("%H:%M:%S"),
                        'system': True
                    })
            
            threading.Thread(target=load_model, daemon=True).start()
    
    def run(self):
        """Main application loop"""
        # Load initial model
        model_id = self.available_models["4B"]
        if not self.processor.load_model(model_id):
            return
        
        # Setup camera
        if not self.setup_camera():
            return
        
        # Start worker threads
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()
        
        self.video_thread = threading.Thread(target=self.video_stream_worker, daemon=True)
        self.video_thread.start()
        
        print("vLLM VLM Video Chat Web Interface ready")
        print("Backend: vLLM (High-Performance)")
        print("Camera: 1280x720 → 896x896 processing")
        print("Web interface: http://localhost:5000")
        
        # Run Flask app
        try:
            self.socketio.run(self.app, host='0.0.0.0', port=5000, debug=False)
        except KeyboardInterrupt:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        if self.gpu_available:
            torch.cuda.empty_cache()
        
        print("vLLM VLM Video Chat Web Interface ended")

if __name__ == "__main__":
    app = VLMWebApp()
    app.run()

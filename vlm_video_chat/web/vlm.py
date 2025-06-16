#!/usr/bin/env python
# vlm_web.py - VLM Video Chat Web Interface

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

print("VLM Video Chat - Web Interface")
print("==============================")

try:
    import torch
    import cv2
    from flask import Flask, render_template, request, jsonify
    from flask_socketio import SocketIO, emit
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor
    from PIL import Image
except ImportError as e:
    print(f"Missing library: {e}")
    print("Install with: pip install flask flask-socketio")
    sys.exit(1)

# Check for bitsandbytes availability
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not installed. 12B quantized model will not be available.")

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

class VLMProcessor:
    """Handles VLM inference in a separate thread"""
    
    def __init__(self, gpu_available=False):
        self.gpu_available = gpu_available
        self.device = torch.device("cuda" if gpu_available else "cpu")
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.current_model_id = None
        
    def load_model(self, model_id="google/gemma-3-4b-it"):
        """Load the VLM model"""
        try:
            # Clear previous model if switching
            if self.current_model_id and self.current_model_id != model_id:
                print(f"Switching from {self.current_model_id} to {model_id}...")
                print("Cleaning up previous model...")
                
                if self.model:
                    if hasattr(self.model, 'to') and self.gpu_available:
                        self.model.to('cpu')
                    del self.model
                    self.model = None
                
                if self.processor:
                    del self.processor
                    self.processor = None
                
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
            
            # Determine model size more accurately - check 12B first!
            if "12b" in model_id.lower():
                model_size = "12B"
            elif "4b" in model_id.lower():
                model_size = "4B"
            else:
                model_size = "Unknown"
            
            is_quantized = "bnb" in model_id or "4bit" in model_id
            print(f"Loading Gemma 3 {model_size}{' (4-bit quantized)' if is_quantized else ''} model on {self.device}...")
            
            # Load 4B model with optimized settings
            print("Loading Gemma 3 4B model with optimized settings...")
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.gpu_available else torch.float32,
                device_map={"": 0} if self.gpu_available else None,
                low_cpu_mem_usage=True
            )
            
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            
            # Override generation config
            self.model.generation_config.do_sample = False
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
            self.model.generation_config.temperature = None
            
            if not self.gpu_available and not is_quantized:
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            self.current_model_id = model_id
            print(f"Model {model_size} loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            if "Connection" in str(e) or "resolve" in str(e) or "404" in str(e):
                print("Note: Model download requires internet connection")
            elif "CUDA out of memory" in str(e):
                print("Note: Insufficient GPU memory for this model")
            return False
    
    def process_image(self, image: np.ndarray, question: str) -> str:
        """Process image with VLM"""
        if not self.is_loaded or self.model is None:
            return "Model not loaded or unavailable"
        
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
            
            # Create message format for Gemma 3
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question}
                ]
            }]
            
            # Apply chat template and process inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[pil_image], return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            input_ids = inputs["input_ids"][0]
            output_ids = outputs[0]
            
            # Extract only the new tokens
            new_tokens = output_ids[len(input_ids):]
            generated_text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error during image processing: {e}")
            return f"Analysis error: {str(e)}"

class VLMWebApp:
    """Web-based VLM Video Chat Interface"""
    
    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'vlm_video_chat_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize components
        self.setup_gpu()
        self.processor = VLMProcessor(self.gpu_available)
        
        # Threading components
        self.request_queue = queue.Queue(maxsize=1)
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
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("Using CPU mode")
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
        """Worker thread for VLM processing"""
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
                                 bitsandbytes_available=BITSANDBYTES_AVAILABLE)
        
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            emit('status_update', {
                'status': 'ready',
                'model': self.selected_model,
                'device': 'GPU' if self.gpu_available else 'CPU'
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
                'sender': 'VLM',
                'message': 'Analyzing image...',
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
            # Model switching disabled - only 4B model available
            # Just acknowledge the request but don't change anything
            emit('status_update', {'status': 'ready', 'model': '4B'})
    
    def run(self):
        """Main application loop"""
        # Load 4B model (only model available)
        model_id = "google/gemma-3-4b-it"
        if not self.processor.load_model(model_id):
            print("Failed to load Gemma 3 4B model")
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
        
        print("VLM Video Chat Web Interface ready")
        print("Open your browser to: http://localhost:5000")
        
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
        
        print("VLM Video Chat Web Interface ended")

if __name__ == "__main__":
    app = VLMWebApp()
    app.run()
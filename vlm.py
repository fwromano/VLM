#!/usr/bin/env python
# vlm.py - VLM Video Chat Interface

import os
import sys
import warnings
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional
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

print("VLM Video Chat")
print("==============")

try:
    import torch
    import cv2
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    from transformers import pipeline
    from PIL import Image, ImageTk
    import numpy as np
except ImportError as e:
    print(f"Missing library: {e}")
    sys.exit(1)

@dataclass
class AnalysisRequest:
    image: np.ndarray
    question: str
    timestamp: float

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
        self.device_str = "cuda" if gpu_available else "cpu"
        self.pipe = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the VLM model"""
        try:
            print(f"Loading InternVL3-2B model on {self.device_str.upper()}...")
            
            self.pipe = pipeline(
                "image-text-to-text",
                model="OpenGVLab/InternVL3-2B-hf",
                trust_remote_code=True,
                device=self.device_str
            )
            
            self.is_loaded = True
            print(f"Model loaded on {self.device_str}")
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False
    
    def process_image(self, image: np.ndarray, question: str) -> str:
        """Process image with VLM"""
        if not self.is_loaded:
            return "Model not loaded"
        
        try:
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Create message format for pipeline
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question}
                ]
            }]
            
            # Generate response
            outputs = self.pipe(
                text=messages,
                max_new_tokens=150,
                return_full_text=False
            )
            
            return outputs[0]["generated_text"]
            
        except Exception as e:
            return f"Analysis error: {str(e)[:100]}"

class VLMChatApp:
    """Video Chat Style VLM Interface"""
    
    def __init__(self):
        # Initialize components
        self.setup_gpu()
        self.processor = VLMProcessor(self.gpu_available)
        
        # Threading components
        self.request_queue = queue.Queue(maxsize=1)
        self.response_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        self.is_processing = False
        
        # Camera
        self.cap = None
        self.current_frame = None
        
        # GUI
        self.root = None
        self.video_label = None
        self.chat_text = None
        self.input_entry = None
        self.send_button = None
        self.status_label = None
        
        # Quick prompts
        self.quick_prompts = [
            "What do you see in this image?",
            "Describe this scene in detail.",
            "What objects are visible?",
            "What colors are prominent?",
            "Is this environment safe?",
            "What actions are happening?"
        ]
        
    def setup_gpu(self):
        """Test GPU availability"""
        self.gpu_available = False
        self.device = torch.device('cpu')
        
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                test_tensor = torch.tensor([1.0], device='cuda:0')
                result = (test_tensor + 1).cpu().item()
                self.device = torch.device('cuda:0')
                self.gpu_available = True
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("Using CPU mode")
        except Exception as e:
            print(f"GPU test failed, using CPU: {e}")
    
    def setup_gui(self):
        """Setup the main GUI window"""
        self.root = tk.Tk()
        self.root.title("VLM Video Chat")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Video
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = tk.Label(video_frame, bg="black", text="Initializing camera...", 
                                   fg="white", font=("Arial", 16))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Status frame under video
        video_status_frame = ttk.Frame(video_frame)
        video_status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(video_status_frame, text="Initializing...", 
                                     font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT)
        
        # Processing indicator
        self.processing_label = ttk.Label(video_status_frame, text="", 
                                         font=("Arial", 10), foreground="orange")
        self.processing_label.pack(side=tk.RIGHT)
        
        # Right side - Chat interface
        chat_frame = ttk.LabelFrame(main_frame, text="VLM Chat", padding="10")
        chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))
        chat_frame.configure(width=400)
        
        # Chat history
        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            width=50,
            height=25,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Arial", 10),
            bg="#ffffff",
            fg="#2c3e50"
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Configure text tags for styling
        self.chat_text.tag_configure("user", foreground="#2980b9", font=("Arial", 10, "bold"))
        self.chat_text.tag_configure("assistant", foreground="#27ae60", font=("Arial", 10, "bold"))
        self.chat_text.tag_configure("timestamp", foreground="#7f8c8d", font=("Arial", 8))
        self.chat_text.tag_configure("processing", foreground="#e67e22", font=("Arial", 9, "italic"))
        
        # Input frame
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Text input
        self.input_entry = tk.Text(input_frame, height=3, wrap=tk.WORD, font=("Arial", 10))
        self.input_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Bind Enter key (Ctrl+Enter for newline)
        self.input_entry.bind("<Return>", self.on_enter_key)
        self.input_entry.bind("<Control-Return>", lambda e: self.input_entry.insert(tk.INSERT, "\n"))
        
        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X)
        
        self.send_button = ttk.Button(button_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Label(button_frame, text="Press Enter to send, Ctrl+Enter for new line", 
                 font=("Arial", 8), foreground="gray").pack(side=tk.LEFT)
        
        # Quick prompts frame
        quick_frame = ttk.LabelFrame(chat_frame, text="Quick Prompts", padding="5")
        quick_frame.pack(fill=tk.X)
        
        # Create grid of quick prompt buttons
        for i, prompt in enumerate(self.quick_prompts):
            row = i // 2
            col = i % 2
            btn = ttk.Button(quick_frame, text=prompt[:20] + "...", 
                           command=lambda p=prompt: self.use_quick_prompt(p))
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
        
        # Configure grid weights
        quick_frame.columnconfigure(0, weight=1)
        quick_frame.columnconfigure(1, weight=1)
        
        # Focus on input
        self.input_entry.focus_set()
        
        # Add welcome message
        self.add_message("System", "Welcome to VLM Video Chat! Type a question about what you see, or use the quick prompts below.", "assistant")
    
    def on_enter_key(self, event):
        """Handle Enter key press"""
        if not event.state & 0x4:  # If Ctrl is not pressed
            self.send_message()
            return "break"  # Prevent default behavior
    
    def use_quick_prompt(self, prompt):
        """Use a quick prompt"""
        self.input_entry.delete(1.0, tk.END)
        self.input_entry.insert(1.0, prompt)
        self.send_message()
    
    def add_message(self, sender, message, tag=""):
        """Add a message to the chat"""
        self.chat_text.configure(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        self.chat_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        
        # Add sender
        self.chat_text.insert(tk.END, f"{sender}: ", tag)
        
        # Add message
        self.chat_text.insert(tk.END, f"{message}\n\n")
        
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)
    
    def send_message(self):
        """Send a message for analysis"""
        if self.is_processing:
            messagebox.showwarning("Please Wait", "Still processing previous request...")
            return
        
        question = self.input_entry.get(1.0, tk.END).strip()
        if not question:
            return
        
        if self.current_frame is None:
            messagebox.showerror("No Camera", "No camera frame available")
            return
        
        # Clear input
        self.input_entry.delete(1.0, tk.END)
        
        # Add user message to chat
        self.add_message("You", question, "user")
        
        # Show processing message
        self.add_message("VLM", "Analyzing image...", "processing")
        self.processing_label.config(text="Processing...")
        self.is_processing = True
        
        # Add request to queue
        try:
            request = AnalysisRequest(
                image=self.current_frame.copy(),
                question=question,
                timestamp=time.time()
            )
            self.request_queue.put_nowait(request)
        except queue.Full:
            self.add_message("System", "Processing queue full, please wait...", "assistant")
            self.is_processing = False
            self.processing_label.config(text="")
    
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
                
                self.response_queue.put(response)
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def setup_camera(self):
        """Initialize camera"""
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Camera not found")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera ready: {width}x{height}")
        
        return True
    
    def update_video_frame(self):
        """Update video display"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                # Convert for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to fit display
                display_height = 480
                aspect_ratio = frame.shape[1] / frame.shape[0]
                display_width = int(display_height * aspect_ratio)
                
                frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
                
                # Convert to PhotoImage
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)
                
                # Update label
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo  # Keep a reference
        
        # Check for responses
        try:
            while True:
                response = self.response_queue.get_nowait()
                
                # Remove processing message
                self.chat_text.configure(state=tk.NORMAL)
                # Remove last two lines (processing message)
                content = self.chat_text.get(1.0, tk.END)
                lines = content.split('\n')
                if len(lines) >= 3 and "Analyzing image" in lines[-3]:
                    # Remove the processing message
                    for _ in range(3):
                        self.chat_text.delete(f"{len(lines)-2}.0", tk.END)
                        lines.pop()
                self.chat_text.configure(state=tk.DISABLED)
                
                # Add actual response
                response_with_time = f"{response.response}\n(Processing time: {response.processing_time:.2f}s)"
                self.add_message("VLM", response_with_time, "assistant")
                
                self.is_processing = False
                self.processing_label.config(text="")
                self.response_queue.task_done()
                
        except queue.Empty:
            pass
        
        # Update status
        device_text = f"{'GPU' if self.gpu_available else 'CPU'}: InternVL3-2B"
        queue_size = self.request_queue.qsize()
        status_text = f"{device_text} | Queue: {queue_size}"
        
        self.status_label.config(text=status_text)
        
        # Schedule next update
        if self.is_running:
            self.root.after(33, self.update_video_frame)  # ~30 FPS
    
    def run(self):
        """Main application loop"""
        # Load model
        if not self.processor.load_model():
            return
        
        # Setup camera
        if not self.setup_camera():
            return
        
        # Setup GUI
        self.setup_gui()
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()
        
        # Start video updates
        self.root.after(100, self.update_video_frame)
        
        print("VLM Video Chat ready")
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start GUI
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.cleanup()
    
    def on_closing(self):
        """Handle window closing"""
        self.cleanup()
        self.root.destroy()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        if self.gpu_available:
            torch.cuda.empty_cache()
        
        print("VLM Video Chat ended")

if __name__ == "__main__":
    app = VLMChatApp()
    app.run()
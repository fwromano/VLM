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
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor
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
        self.device = torch.device("cuda" if gpu_available else "cpu")
        self.model = None
        self.processor = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the VLM model"""
        try:
            print(f"Loading Gemma 3 4B model on {self.device}...")
            
            # Load model and processor with optimizations (skip flash attention for now)
            print("Loading with optimized standard attention...")
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                "google/gemma-3-4b-it",
                torch_dtype=torch.bfloat16 if self.gpu_available else torch.float32,
                device_map={"": 0} if self.gpu_available else None,  # Explicit device mapping
                low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
            )
            
            self.processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it", use_fast=True)
            
            # Override generation config to remove sampling parameters
            self.model.generation_config.do_sample = False
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
            self.model.generation_config.temperature = None
            
            if not self.gpu_available:
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            print(f"Model loaded on {self.device}")
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
            
            # Resize image for faster VLM processing while maintaining quality
            max_size = 512  # Reduced from original size for speed
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
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
            
            # Generate response with optimizations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,  # Increased to allow longer detailed responses
                    do_sample=False,
                    use_cache=True,  # Enable KV cache for faster generation
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode output - get the full generated response
            input_ids = inputs["input_ids"][0]
            output_ids = outputs[0]
            
            # Extract only the new tokens (everything after the input)
            new_tokens = output_ids[len(input_ids):]
            generated_text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Return the complete response without any truncation
            return generated_text.strip()
            
        except Exception as e:
            return f"Analysis error: {str(e)}"

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
            "What actions are happening?",
            "Count the people in this image",
            "What text or signs do you see?",
            "Is this indoors or outdoors?",
            "What time of day does this look like?"
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
        """Setup modern, clean GUI interface"""
        self.root = tk.Tk()
        self.root.title("VLM")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1a1a1a")
        self.root.minsize(1000, 700)
        
        # Configure modern style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark theme colors
        colors = {
            'bg': '#1a1a1a',
            'surface': '#2d2d2d', 
            'accent': '#0066cc',
            'text': '#ffffff',
            'text_secondary': '#b0b0b0',
            'border': '#404040'
        }
        
        # Configure ttk styles for dark theme
        style.configure('Modern.TFrame', background=colors['surface'])
        style.configure('Modern.TLabel', background=colors['surface'], foreground=colors['text'])
        style.configure('Modern.TButton', 
                       background=colors['accent'], 
                       foreground=colors['text'],
                       borderwidth=0,
                       focuscolor='none')
        style.map('Modern.TButton',
                 background=[('active', '#0080ff')])
        
        # Main container with dark background
        main_frame = tk.Frame(self.root, bg=colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Video section (left side)
        video_container = tk.Frame(main_frame, bg=colors['surface'], relief='flat', bd=0)
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video header
        video_header = tk.Frame(video_container, bg=colors['surface'], height=40)
        video_header.pack(fill=tk.X, padx=15, pady=(15, 0))
        video_header.pack_propagate(False)
        
        tk.Label(video_header, text="Live Camera", 
                font=("SF Pro Display", 16, "bold"), 
                bg=colors['surface'], fg=colors['text']).pack(side=tk.LEFT, anchor='w')
        
        self.status_label = tk.Label(video_header, text="Ready", 
                                   font=("SF Pro Display", 11), 
                                   bg=colors['surface'], fg=colors['text_secondary'])
        self.status_label.pack(side=tk.RIGHT, anchor='e')
        
        # Video display
        video_frame = tk.Frame(video_container, bg='#000000', relief='flat', bd=0)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(10, 15))
        
        self.video_label = tk.Label(video_frame, bg="#000000", text="Initializing...", 
                                   fg="#666666", font=("SF Pro Display", 14))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Chat section (right side)
        chat_container = tk.Frame(main_frame, bg=colors['surface'], relief='flat', bd=0)
        chat_container.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        chat_container.configure(width=450)
        
        # Chat header
        chat_header = tk.Frame(chat_container, bg=colors['surface'], height=40)
        chat_header.pack(fill=tk.X, padx=15, pady=(15, 0))
        chat_header.pack_propagate(False)
        
        tk.Label(chat_header, text="AI Analysis", 
                font=("SF Pro Display", 16, "bold"), 
                bg=colors['surface'], fg=colors['text']).pack(side=tk.LEFT, anchor='w')
        
        self.processing_label = tk.Label(chat_header, text="", 
                                       font=("SF Pro Display", 11), 
                                       bg=colors['surface'], fg='#ff6b35')
        self.processing_label.pack(side=tk.RIGHT, anchor='e')
        
        # Chat messages area
        chat_frame = tk.Frame(chat_container, bg=colors['surface'])
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(10, 0))
        
        # Custom scrolled text with modern styling
        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("SF Pro Display", 11),
            bg='#1a1a1a',
            fg=colors['text'],
            insertbackground=colors['text'],
            selectbackground=colors['accent'],
            borderwidth=0,
            relief='flat',
            padx=15,
            pady=15
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text styling
        self.chat_text.tag_configure("user", 
                                   foreground="#0066cc", 
                                   font=("SF Pro Display", 11, "bold"))
        self.chat_text.tag_configure("assistant", 
                                   foreground="#00cc66", 
                                   font=("SF Pro Display", 11, "bold"))
        self.chat_text.tag_configure("timestamp", 
                                   foreground="#666666", 
                                   font=("SF Pro Display", 9))
        self.chat_text.tag_configure("processing", 
                                   foreground="#ff6b35", 
                                   font=("SF Pro Display", 10, "italic"))
        
        # Input section
        input_section = tk.Frame(chat_container, bg=colors['surface'])
        input_section.pack(fill=tk.X, padx=15, pady=15)
        
        # Input field with modern styling
        input_frame = tk.Frame(input_section, bg='#1a1a1a', relief='flat', bd=1)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.input_entry = tk.Text(input_frame, 
                                 height=2, 
                                 wrap=tk.WORD, 
                                 font=("SF Pro Display", 11),
                                 bg='#1a1a1a',
                                 fg=colors['text'],
                                 insertbackground=colors['text'],
                                 selectbackground=colors['accent'],
                                 borderwidth=0,
                                 relief='flat',
                                 padx=12,
                                 pady=8)
        self.input_entry.pack(fill=tk.X)
        
        # Placeholder text
        self.input_entry.insert("1.0", "Ask about what you see...")
        self.input_entry.bind("<FocusIn>", self.clear_placeholder)
        self.input_entry.bind("<FocusOut>", self.restore_placeholder)
        self.input_entry.bind("<Return>", self.on_enter_key)
        self.input_entry.bind("<Control-Return>", lambda e: self.input_entry.insert(tk.INSERT, "\n"))
        
        # Send button
        self.send_button = tk.Button(input_section,
                                   text="Send",
                                   font=("SF Pro Display", 11, "bold"),
                                   bg=colors['accent'],
                                   fg=colors['text'],
                                   activebackground='#0080ff',
                                   activeforeground=colors['text'],
                                   relief='flat',
                                   borderwidth=0,
                                   padx=20,
                                   pady=8,
                                   command=self.send_message)
        self.send_button.pack(anchor='e')
        
        # Quick actions (simplified)
        actions_frame = tk.Frame(input_section, bg=colors['surface'])
        actions_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Essential quick prompts only
        essential_prompts = [
            "What do you see?",
            "Describe the scene",
            "Count objects",
            "Read any text"
        ]
        
        for i, prompt in enumerate(essential_prompts):
            btn = tk.Button(actions_frame,
                          text=prompt,
                          font=("SF Pro Display", 9),
                          bg='#404040',
                          fg=colors['text_secondary'],
                          activebackground='#4a4a4a',
                          activeforeground=colors['text'],
                          relief='flat',
                          borderwidth=0,
                          padx=8,
                          pady=4,
                          command=lambda p=prompt: self.use_quick_prompt(p))
            btn.pack(side=tk.LEFT, padx=(0, 5) if i < len(essential_prompts)-1 else 0)
        
        # Focus on input
        self.input_entry.focus_set()
        
        # Welcome message
        self.add_message("AI", "Ready to analyze your camera feed. Ask me anything about what you see.", "assistant")
    
    def clear_placeholder(self, event):
        """Clear placeholder text when focused"""
        if self.input_entry.get("1.0", tk.END).strip() == "Ask about what you see...":
            self.input_entry.delete("1.0", tk.END)
    
    def restore_placeholder(self, event):
        """Restore placeholder text if empty"""
        if not self.input_entry.get("1.0", tk.END).strip():
            self.input_entry.insert("1.0", "Ask about what you see...")
    
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
        
        # Optimize camera settings for speed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced from 1280 for faster processing
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced from 720 for faster processing
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set higher FPS for smoother video
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
        
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
                
                # Resize to fit display (optimized)
                display_height = 400  # Slightly smaller for better performance
                aspect_ratio = frame.shape[1] / frame.shape[0]
                display_width = int(display_height * aspect_ratio)
                
                frame_resized = cv2.resize(frame_rgb, (display_width, display_height), 
                                         interpolation=cv2.INTER_LINEAR)  # Faster interpolation
                
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
                
                # Add actual response - ensure full response is displayed
                full_response = response.response.strip()
                response_with_time = f"{full_response}\n(Processing time: {response.processing_time:.2f}s)"
                self.add_message("VLM", response_with_time, "assistant")
                
                self.is_processing = False
                self.processing_label.config(text="")
                self.response_queue.task_done()
                
        except queue.Empty:
            pass
        
        # Update status
        device_text = f"{'GPU' if self.gpu_available else 'CPU'}: Gemma-3-4B"
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
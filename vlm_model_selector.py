#!/usr/bin/env python3
"""
InternVL3 Model Selector with Performance Comparison
Allows you to choose between different InternVL3 models and see performance tradeoffs
"""

import cv2
import torch
import numpy as np
import threading
import queue
import time
import json
from datetime import datetime
from transformers import pipeline
import argparse
import sys
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelInfo:
    name: str
    full_name: str
    description: str
    parameters: str
    vram_gb: float
    speed_rating: int  # 1-5, 5 being fastest
    quality_rating: int  # 1-5, 5 being best quality
    recommended_use: str

# Available InternVL3 models with their specifications
AVAILABLE_MODELS = {
    "2B": ModelInfo(
        name="InternVL3-2B-hf",
        full_name="lmms-lab/InternVL3-2B-hf", 
        description="Smallest InternVL3 model, fastest inference",
        parameters="2B",
        vram_gb=4.0,
        speed_rating=5,
        quality_rating=3,
        recommended_use="Real-time applications, edge devices"
    ),
    "8B": ModelInfo(
        name="InternVL3-8B-hf",
        full_name="lmms-lab/InternVL3-8B-hf",
        description="Balanced model, good speed/quality tradeoff", 
        parameters="8B",
        vram_gb=12.0,
        speed_rating=3,
        quality_rating=4,
        recommended_use="General purpose, most applications"
    ),
    "26B": ModelInfo(
        name="InternVL3-26B-hf", 
        full_name="lmms-lab/InternVL3-26B-hf",
        description="Large model, best quality but slower",
        parameters="26B", 
        vram_gb=24.0,
        speed_rating=2,
        quality_rating=5,
        recommended_use="High-quality analysis, research"
    ),
    "72B": ModelInfo(
        name="InternVL3-72B-hf",
        full_name="lmms-lab/InternVL3-72B-hf", 
        description="Largest model, highest quality, requires multiple GPUs",
        parameters="72B",
        vram_gb=48.0,
        speed_rating=1,
        quality_rating=5,
        recommended_use="Research, maximum quality needed"
    )
}

# Fallback models if InternVL3 not available
FALLBACK_MODELS = {
    "blip-base": ModelInfo(
        name="blip-image-captioning-base",
        full_name="Salesforce/blip-image-captioning-base",
        description="BLIP image captioning (fallback)",
        parameters="200M", 
        vram_gb=1.0,
        speed_rating=5,
        quality_rating=2,
        recommended_use="Fallback when InternVL3 unavailable"
    ),
    "blip-large": ModelInfo(
        name="blip-image-captioning-large", 
        full_name="Salesforce/blip-image-captioning-large",
        description="BLIP large image captioning (fallback)",
        parameters="400M",
        vram_gb=2.0, 
        speed_rating=4,
        quality_rating=3,
        recommended_use="Better fallback option"
    )
}

class VLMModelSelector:
    def __init__(self, model_key="2B", device='cuda', camera_device=0):
        self.model_key = model_key
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.camera_device = camera_device
        
        # Get model info
        if model_key in AVAILABLE_MODELS:
            self.model_info = AVAILABLE_MODELS[model_key]
        elif model_key in FALLBACK_MODELS:
            self.model_info = FALLBACK_MODELS[model_key]
        else:
            raise ValueError(f"Unknown model: {model_key}")
        
        print(f"🚀 InternVL3 Model Selector")
        print(f"Selected Model: {self.model_info.name}")
        print(f"Parameters: {self.model_info.parameters}")
        print(f"Description: {self.model_info.description}")
        print(f"Device: {self.device}")
        print(f"VRAM Required: {self.model_info.vram_gb}GB")
        print("")
        
        # Check VRAM if using GPU
        if self.device == 'cuda':
            self.check_vram()
        
        # Initialize model
        self.init_model()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_device)
        if not self.cap.isOpened():
            print(f"❌ Error: Cannot open camera {camera_device}")
            sys.exit(1)
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Analysis tracking
        self.analysis_queue = queue.Queue(maxsize=10)
        self.results_queue = queue.Queue()
        self.performance_stats = {
            'total_analyses': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0
        }
        
        # Control variables
        self.running = True
        self.continuous_mode = True
        self.analysis_rate = 0.5
        self.default_prompt = "What do you see in this image?"
        
        # Start threads
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.analysis_thread = threading.Thread(target=self.analysis_loop)
        self.camera_thread.start()
        self.analysis_thread.start()
    
    def check_vram(self):
        """Check if GPU has enough VRAM for selected model"""
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            required_gb = self.model_info.vram_gb
            
            print(f"📊 GPU Memory Check:")
            print(f"   Available: {gpu_memory_gb:.1f}GB")
            print(f"   Required: {required_gb}GB")
            
            if gpu_memory_gb < required_gb:
                print(f"⚠️  Warning: May not have enough VRAM")
                print(f"   Consider using a smaller model or CPU")
            else:
                print(f"✅ Sufficient VRAM available")
            print("")
        except Exception as e:
            print(f"Could not check GPU memory: {e}")
    
    def init_model(self):
        """Initialize the selected VLM model"""
        print(f"🔄 Loading {self.model_info.name}...")
        start_time = time.time()
        
        try:
            # Try to load the specified model
            if "InternVL3" in self.model_info.full_name:
                # For InternVL3 models, try VQA pipeline first
                try:
                    self.model = pipeline(
                        "visual-question-answering",
                        model=self.model_info.full_name,
                        device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1
                    )
                    self.model_type = "vqa"
                except Exception as e:
                    print(f"VQA pipeline failed: {e}")
                    print("Trying image-to-text pipeline...")
                    self.model = pipeline(
                        "image-to-text", 
                        model=self.model_info.full_name,
                        device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1
                    )
                    self.model_type = "caption"
            else:
                # For fallback models
                self.model = pipeline(
                    "image-to-text",
                    model=self.model_info.full_name,
                    device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1
                )
                self.model_type = "caption"
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.1f}s")
            print(f"   Type: {self.model_type}")
            print("")
            
        except Exception as e:
            print(f"❌ Error loading {self.model_info.name}: {e}")
            print("🔄 Falling back to BLIP model...")
            
            try:
                self.model_info = FALLBACK_MODELS["blip-base"]
                self.model = pipeline(
                    "image-to-text",
                    model=self.model_info.full_name,
                    device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1
                )
                self.model_type = "caption"
                load_time = time.time() - start_time
                print(f"✅ Fallback model loaded in {load_time:.1f}s")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                sys.exit(1)
    
    def camera_loop(self):
        """Continuously capture frames"""
        last_analysis_time = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Add performance overlay
            self.draw_overlay(frame)
            
            # Show frame
            cv2.imshow(f'VLM Camera - {self.model_info.name}', frame)
            
            # Check if we should analyze
            current_time = time.time()
            if self.continuous_mode and (current_time - last_analysis_time) >= (1.0 / self.analysis_rate):
                if not self.analysis_queue.full():
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.analysis_queue.put({
                        'image': rgb_frame,
                        'prompt': self.default_prompt,
                        'timestamp': datetime.now()
                    })
                    last_analysis_time = current_time
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('p'):
                self.continuous_mode = not self.continuous_mode
                print(f"🔄 Continuous mode: {'ON' if self.continuous_mode else 'OFF'}")
            elif key == ord('s'):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.analysis_queue.put({
                    'image': rgb_frame,
                    'prompt': self.default_prompt,
                    'timestamp': datetime.now()
                })
            elif key == ord('r'):
                self.reset_stats()
    
    def draw_overlay(self, frame):
        """Draw performance information on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Model info
        cv2.putText(frame, f"Model: {self.model_info.name}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Params: {self.model_info.parameters}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Performance stats
        if self.performance_stats['total_analyses'] > 0:
            cv2.putText(frame, f"Avg: {self.performance_stats['avg_time']:.2f}s", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Count: {self.performance_stats['total_analyses']}", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "Q:Quit P:Pause S:Single R:Reset", (20, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def analysis_loop(self):
        """Process analysis requests and track performance"""
        while self.running:
            try:
                task = self.analysis_queue.get(timeout=1.0)
                
                # Perform analysis
                start_time = time.time()
                result = self.analyze_image(task['image'], task['prompt'])
                processing_time = time.time() - start_time
                
                # Update performance stats
                self.update_stats(processing_time)
                
                # Create analysis message
                analysis_msg = {
                    'timestamp': task['timestamp'].isoformat(),
                    'model': self.model_info.name,
                    'prompt': task['prompt'],
                    'text': result,
                    'processing_time': processing_time,
                    'device': self.device
                }
                
                # Print result
                print(f"\n🔍 === {self.model_info.name} Analysis ===")
                print(f"⏰ Time: {task['timestamp'].strftime('%H:%M:%S')}")
                print(f"❓ Prompt: {task['prompt']}")
                print(f"💬 Response: {result}")
                print(f"⚡ Processing: {processing_time:.3f}s")
                print(f"📊 Avg Speed: {self.performance_stats['avg_time']:.3f}s")
                print("=" * 50)
                
                self.results_queue.put(analysis_msg)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Analysis error: {e}")
    
    def analyze_image(self, image, prompt):
        """Run VLM inference on image"""
        try:
            from PIL import Image as PILImage
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            
            # Use appropriate pipeline
            if self.model_type == "vqa":
                result = self.model(pil_image, prompt)
                return result[0]['answer'] if result else "No response"
            else:
                result = self.model(pil_image)
                return result[0]['generated_text'] if result else "No description"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def update_stats(self, processing_time):
        """Update performance statistics"""
        self.performance_stats['total_analyses'] += 1
        self.performance_stats['total_time'] += processing_time
        self.performance_stats['avg_time'] = (
            self.performance_stats['total_time'] / self.performance_stats['total_analyses']
        )
        self.performance_stats['min_time'] = min(self.performance_stats['min_time'], processing_time)
        self.performance_stats['max_time'] = max(self.performance_stats['max_time'], processing_time)
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_analyses': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0
        }
        print("📊 Performance stats reset")
    
    def set_prompt(self, prompt, rate=None):
        """Change analysis prompt and rate"""
        self.default_prompt = prompt
        if rate is not None:
            self.analysis_rate = rate
        print(f"🔄 Updated: Prompt='{prompt}', Rate={self.analysis_rate}Hz")
    
    def cleanup(self):
        """Clean up resources and save results"""
        self.running = False
        self.camera_thread.join()
        self.analysis_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save results and performance stats
        if not self.results_queue.empty():
            results = []
            while not self.results_queue.empty():
                results.append(self.results_queue.get())
            
            # Add performance summary
            summary = {
                'model_info': {
                    'name': self.model_info.name,
                    'parameters': self.model_info.parameters,
                    'device': self.device,
                    'vram_gb': self.model_info.vram_gb,
                    'speed_rating': self.model_info.speed_rating,
                    'quality_rating': self.model_info.quality_rating
                },
                'performance_stats': self.performance_stats,
                'analyses': results
            }
            
            filename = f"vlm_analysis_{self.model_info.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n📁 Results saved to: {filename}")
            print(f"📊 Performance Summary:")
            print(f"   Model: {self.model_info.name}")
            print(f"   Total Analyses: {self.performance_stats['total_analyses']}")
            print(f"   Average Time: {self.performance_stats['avg_time']:.3f}s")
            print(f"   Min Time: {self.performance_stats['min_time']:.3f}s")
            print(f"   Max Time: {self.performance_stats['max_time']:.3f}s")

def show_model_menu():
    """Display available models and their specs"""
    print("\n🎯 Available InternVL3 Models:")
    print("=" * 80)
    
    for key, model in AVAILABLE_MODELS.items():
        print(f"🔹 {key}: {model.name}")
        print(f"   Parameters: {model.parameters}")
        print(f"   VRAM: {model.vram_gb}GB")
        print(f"   Speed: {'⭐' * model.speed_rating} ({model.speed_rating}/5)")
        print(f"   Quality: {'⭐' * model.quality_rating} ({model.quality_rating}/5)")
        print(f"   Use Case: {model.recommended_use}")
        print(f"   Description: {model.description}")
        print()
    
    print("🔹 Fallback Models:")
    for key, model in FALLBACK_MODELS.items():
        print(f"   {key}: {model.name} ({model.parameters})")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='InternVL3 Model Selector with Performance Comparison')
    parser.add_argument('--model', default='2B', 
                       choices=list(AVAILABLE_MODELS.keys()) + list(FALLBACK_MODELS.keys()),
                       help='Model to use (default: 2B)')
    parser.add_argument('--list-models', action='store_true',
                       help='Show available models and exit')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--rate', type=float, default=0.5,
                       help='Analysis rate in Hz')
    parser.add_argument('--prompt', default="What do you see in this image?",
                       help='Default analysis prompt')
    parser.add_argument('--no-continuous', action='store_true',
                       help='Disable continuous analysis')
    
    args = parser.parse_args()
    
    if args.list_models:
        show_model_menu()
        return
    
    print("🚀 InternVL3 Model Selector")
    print("Compare performance across different InternVL3 models!")
    print("=" * 60)
    
    # Check if VLM environment is active
    if 'CONDA_PREFIX' in os.environ and 'vlm' in os.environ['CONDA_PREFIX']:
        print("✅ VLM conda environment detected")
    else:
        print("⚠️  Warning: VLM conda environment not active")
        print("   Run: conda activate vlm")
    
    print("\n🎮 Controls:")
    print("   q - Quit")
    print("   p - Toggle continuous mode")
    print("   s - Single analysis")
    print("   r - Reset performance stats")
    print("")
    
    # Initialize model selector
    try:
        vlm = VLMModelSelector(
            model_key=args.model,
            device=args.device,
            camera_device=args.camera
        )
        
        vlm.set_prompt(args.prompt, args.rate)
        if args.no_continuous:
            vlm.continuous_mode = False
        
        # Keep main thread alive
        while vlm.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'vlm' in locals():
            vlm.cleanup()

if __name__ == '__main__':
    main()
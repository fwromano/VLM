#!/usr/bin/env python3
"""
Standalone VLM ROS2-style interface without ROS2 dependencies
This provides similar functionality for testing without full ROS2 setup
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

class VLMStandalone:
    def __init__(self, device='cuda', model_name='InternVL3-2B-hf', camera_device=0):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.camera_device = camera_device
        
        print(f"Initializing VLM Standalone Mode")
        print(f"Device: {self.device}")
        print(f"Model: {self.model_name}")
        
        # Initialize model
        self.init_model()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_device)
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {camera_device}")
            sys.exit(1)
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Analysis queue
        self.analysis_queue = queue.Queue(maxsize=10)
        self.results_queue = queue.Queue()
        
        # Control variables
        self.running = True
        self.continuous_mode = True
        self.analysis_rate = 0.5  # Hz
        self.default_prompt = "What do you see in this image?"
        
        # Start threads
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.analysis_thread = threading.Thread(target=self.analysis_loop)
        self.camera_thread.start()
        self.analysis_thread.start()
    
    def init_model(self):
        """Initialize the VLM model"""
        print("Loading VLM model...")
        try:
            # Use HuggingFace pipeline which handles the model loading
            self.model = pipeline(
                "visual-question-answering",
                model=f"lmms-lab/{self.model_name}",
                device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to image captioning model...")
            try:
                self.model = pipeline(
                    "image-to-text",
                    model="Salesforce/blip-image-captioning-base",
                    device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1
                )
                print("Fallback model loaded!")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                sys.exit(1)
    
    def camera_loop(self):
        """Continuously capture frames"""
        last_analysis_time = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Show frame
            cv2.imshow('VLM Camera Feed', frame)
            
            # Check if we should analyze
            current_time = time.time()
            if self.continuous_mode and (current_time - last_analysis_time) >= (1.0 / self.analysis_rate):
                if not self.analysis_queue.full():
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.analysis_queue.put({
                        'image': rgb_frame,
                        'prompt': self.default_prompt,
                        'timestamp': datetime.now()
                    })
                    last_analysis_time = current_time
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('p'):
                # Pause/resume continuous mode
                self.continuous_mode = not self.continuous_mode
                print(f"Continuous mode: {'ON' if self.continuous_mode else 'OFF'}")
            elif key == ord('s'):
                # Single analysis
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.analysis_queue.put({
                    'image': rgb_frame,
                    'prompt': self.default_prompt,
                    'timestamp': datetime.now()
                })
    
    def analysis_loop(self):
        """Process analysis requests"""
        while self.running:
            try:
                task = self.analysis_queue.get(timeout=1.0)
                
                # Perform analysis
                start_time = time.time()
                result = self.analyze_image(task['image'], task['prompt'])
                processing_time = time.time() - start_time
                
                # Print result (simulating ROS2 topic publish)
                analysis_msg = {
                    'timestamp': task['timestamp'].isoformat(),
                    'prompt': task['prompt'],
                    'text': result,
                    'processing_time': processing_time,
                    'model_name': self.model_name
                }
                
                print(f"\n=== VLM Analysis ===")
                print(f"Time: {task['timestamp'].strftime('%H:%M:%S')}")
                print(f"Prompt: {task['prompt']}")
                print(f"Response: {result}")
                print(f"Processing: {processing_time:.2f}s")
                print("==================\n")
                
                # Store result
                self.results_queue.put(analysis_msg)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {e}")
    
    def analyze_image(self, image, prompt):
        """Run VLM inference on image"""
        try:
            # Convert numpy array to PIL Image
            from PIL import Image as PILImage
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            
            # Use the pipeline
            if hasattr(self.model, 'task') and 'visual-question-answering' in self.model.task:
                # VQA model
                result = self.model(pil_image, prompt)
                return result[0]['answer'] if result else "No response"
            else:
                # Image captioning model
                result = self.model(pil_image)
                return result[0]['generated_text'] if result else "No description available"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def set_prompt(self, prompt, rate=None):
        """Change the analysis prompt and rate"""
        self.default_prompt = prompt
        if rate is not None:
            self.analysis_rate = rate
        print(f"Updated: Prompt='{prompt}', Rate={self.analysis_rate}Hz")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.camera_thread.join()
        self.analysis_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save results
        if not self.results_queue.empty():
            results = []
            while not self.results_queue.empty():
                results.append(self.results_queue.get())
            
            filename = f"vlm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='VLM Standalone Mode')
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
    
    print("VLM Standalone Mode (ROS2-style interface without ROS2)")
    print("======================================================")
    print("Controls:")
    print("  q - Quit")
    print("  p - Toggle continuous mode")
    print("  s - Single analysis")
    print("")
    
    # Check if VLM environment is active
    if 'CONDA_PREFIX' in os.environ and 'vlm' in os.environ['CONDA_PREFIX']:
        print("✓ VLM conda environment detected")
    else:
        print("! Warning: VLM conda environment not active")
        print("  Run: conda activate vlm")
    
    vlm = VLMStandalone(
        device=args.device,
        camera_device=args.camera
    )
    
    vlm.set_prompt(args.prompt, args.rate)
    if args.no_continuous:
        vlm.continuous_mode = False
    
    try:
        # Keep main thread alive
        while vlm.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        vlm.cleanup()


if __name__ == '__main__':
    main()
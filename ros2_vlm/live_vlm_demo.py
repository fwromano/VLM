#!/usr/bin/env python3
"""
Live VLM Demo - Integrated Camera + Analysis
Shows live camera feed with real-time VLM analysis overlay
"""

import cv2
import numpy as np
import threading
import queue
import time
import subprocess
import tempfile
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk

class LiveVLMDemo:
    def __init__(self):
        print("🎥 Live VLM Demo Starting...")
        
        # Camera setup
        self.cap = None
        self.camera_running = False
        
        # VLM setup
        self.conda_env_path = "/home/fwromano/anaconda3/envs/vlm"
        self.vlm_script_path = str(Path(__file__).parent / "vlm_processor.py")
        
        # Analysis state
        self.current_prompt = "What do you see in this image?"
        self.last_analysis = "Starting analysis..."
        self.analysis_time = 0
        self.processing = False
        
        # Threading
        self.analysis_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue()
        
        # Predefined prompts for quick access
        self.prompts = {
            '1': "What objects do you see?",
            '2': "Describe the colors and scene",
            '3': "Is this environment safe?",
            '4': "Count the people in this image",
            '5': "What actions are happening?",
            '6': "Describe the lighting and mood",
            '7': "What text or signs do you see?",
            '8': "Is this indoors or outdoors?",
            '9': "What time of day does this look like?"
        }
        
        print("✓ Demo initialized")
    
    def setup_camera(self):
        """Initialize camera"""
        print("📹 Setting up camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Camera not found!")
            return False
        
        # Camera settings for good performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera ready: {width}x{height}")
        
        self.camera_running = True
        return True
    
    def vlm_worker(self):
        """Background thread for VLM processing"""
        print("🤖 VLM worker started")
        
        while self.camera_running:
            try:
                # Get image from queue
                image_data = self.analysis_queue.get(timeout=1.0)
                if image_data is None:  # Shutdown signal
                    break
                
                image, prompt = image_data
                
                # Process with VLM
                start_time = time.time()
                result = self.process_with_vlm(image, prompt)
                processing_time = time.time() - start_time
                
                # Send result back
                self.result_queue.put((result, processing_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"VLM worker error: {e}")
                continue
        
        print("🤖 VLM worker stopped")
    
    def process_with_vlm(self, image, prompt):
        """Process image with VLM using conda environment"""
        try:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image)
                temp_image_path = tmp_file.name
            
            # Run VLM processing
            cmd = [
                f"{self.conda_env_path}/bin/python",
                self.vlm_script_path,
                temp_image_path,
                prompt
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,  # Shorter timeout for responsiveness
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            )
            
            # Clean up temp file
            os.unlink(temp_image_path)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error: {result.stderr[:100]}"
                
        except Exception as e:
            return f"Error: {str(e)[:100]}"
    
    def draw_overlay(self, frame):
        """Draw analysis overlay on frame"""
        overlay = frame.copy()
        
        # Create semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (630, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Current prompt
        prompt_text = f"Prompt: {self.current_prompt}"
        cv2.putText(frame, prompt_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Analysis result
        analysis_lines = self.wrap_text(self.last_analysis, 70)
        for i, line in enumerate(analysis_lines[:3]):  # Max 3 lines
            y_pos = 60 + (i * 25)
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Processing time
        if self.analysis_time > 0:
            time_text = f"Analysis time: {self.analysis_time:.1f}s"
            cv2.putText(frame, time_text, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Processing indicator
        if self.processing:
            cv2.circle(frame, (600, 30), 10, (0, 165, 255), -1)  # Orange circle
            cv2.putText(frame, "ANALYZING", (520, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        return frame
    
    def wrap_text(self, text, width):
        """Wrap text to fit in specified width"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= width:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def show_help(self, frame):
        """Show help overlay"""
        overlay = frame.copy()
        
        # Create help background
        cv2.rectangle(overlay, (50, 200), (590, 450), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        help_text = [
            "LIVE VLM DEMO - KEYBOARD CONTROLS",
            "",
            "NUMBER KEYS - Quick prompts:",
            "  1: What objects do you see?",
            "  2: Describe colors and scene", 
            "  3: Is this environment safe?",
            "  4: Count people in image",
            "  5: What actions are happening?",
            "  6: Describe lighting and mood",
            "  7: What text or signs do you see?",
            "  8: Indoors or outdoors?",
            "  9: What time of day?",
            "",
            "OTHER KEYS:",
            "  H: Toggle this help",
            "  Q/ESC: Quit demo",
            "  SPACE: Trigger analysis now"
        ]
        
        for i, line in enumerate(help_text):
            y_pos = 220 + (i * 15)
            font_scale = 0.5 if line.startswith(" ") else 0.6
            color = (255, 255, 255) if not line.startswith("LIVE VLM") else (0, 255, 255)
            thickness = 1 if not line.startswith("LIVE VLM") else 2
            cv2.putText(frame, line, (60, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return frame
    
    def run_demo(self):
        """Main demo loop"""
        if not self.setup_camera():
            return
        
        # Start VLM worker thread
        vlm_thread = threading.Thread(target=self.vlm_worker, daemon=True)
        vlm_thread.start()
        
        print("\n🎥 Live VLM Demo Running!")
        print("=" * 40)
        print("📱 Press 'H' for help")
        print("🔢 Press 1-9 for quick prompts")
        print("❌ Press 'Q' or ESC to quit")
        print("=" * 40)
        
        show_help = False
        last_analysis_time = 0
        analysis_interval = 2.0  # Analyze every 2 seconds
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Check for VLM results
                try:
                    while True:
                        result, proc_time = self.result_queue.get_nowait()
                        self.last_analysis = result
                        self.analysis_time = proc_time
                        self.processing = False
                        print(f"🤖 Analysis: {result[:50]}...")
                except queue.Empty:
                    pass
                
                # Trigger analysis periodically
                current_time = time.time()
                if (current_time - last_analysis_time) > analysis_interval and not self.processing:
                    try:
                        self.analysis_queue.put_nowait((frame.copy(), self.current_prompt))
                        self.processing = True
                        last_analysis_time = current_time
                    except queue.Full:
                        pass  # Skip if queue is full
                
                # Draw overlay
                if show_help:
                    frame = self.show_help(frame)
                else:
                    frame = self.draw_overlay(frame)
                
                # Show frame
                cv2.imshow('Live VLM Demo', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('h'):  # Toggle help
                    show_help = not show_help
                elif key == ord(' '):  # Force analysis
                    if not self.processing:
                        try:
                            self.analysis_queue.put_nowait((frame.copy(), self.current_prompt))
                            self.processing = True
                            last_analysis_time = current_time
                        except queue.Full:
                            pass
                elif chr(key) in self.prompts:  # Number key prompts
                    self.current_prompt = self.prompts[chr(key)]
                    print(f"🎯 Prompt changed to: {self.current_prompt}")
                    # Trigger immediate analysis with new prompt
                    if not self.processing:
                        try:
                            self.analysis_queue.put_nowait((frame.copy(), self.current_prompt))
                            self.processing = True
                            last_analysis_time = current_time
                        except queue.Full:
                            pass
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        self.camera_running = False
        
        # Signal VLM worker to stop
        try:
            self.analysis_queue.put_nowait(None)
        except queue.Full:
            pass
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("✅ Demo stopped")

def main():
    # Clean environment (remove conda interference)
    os.environ['PATH'] = ':'.join([p for p in os.environ.get('PATH', '').split(':') 
                                  if 'anaconda' not in p and 'miniconda' not in p])
    
    demo = LiveVLMDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
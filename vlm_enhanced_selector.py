#!/usr/bin/env python3
"""
Enhanced VLM Model Selector with Structured Output and Long Responses
Supports longer responses, JSON output, reasoning chains, and structured analysis
"""

import cv2
import torch
import numpy as np
import threading
import queue
import time
import json
from datetime import datetime
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM
import argparse
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import re

# Enhanced model configurations with response length support
@dataclass
class EnhancedModelInfo:
    name: str
    full_name: str
    description: str
    parameters: str
    vram_gb: float
    speed_rating: int
    quality_rating: int
    max_tokens: int  # NEW: Maximum response length
    supports_reasoning: bool  # NEW: Supports chain-of-thought
    recommended_use: str

# Available models with enhanced capabilities (CORRECTED NAMES)
ENHANCED_MODELS = {
    "InternVL3-2B": EnhancedModelInfo(
        name="InternVL3-2B",
        full_name="OpenGVLab/InternVL3-2B",
        description="Latest InternVL3 2B model with enhanced capabilities",
        parameters="2B",
        vram_gb=4.0,
        speed_rating=5,
        quality_rating=4,
        max_tokens=512,
        supports_reasoning=True,
        recommended_use="Real-time applications with detailed analysis"
    ),
    "InternVL3-8B": EnhancedModelInfo(
        name="InternVL3-8B", 
        full_name="OpenGVLab/InternVL3-8B",
        description="Latest InternVL3 8B model with advanced reasoning",
        parameters="8B",
        vram_gb=12.0,
        speed_rating=3,
        quality_rating=5,
        max_tokens=1024,
        supports_reasoning=True,
        recommended_use="Best balanced model for detailed analysis"
    ),
    "InternVL2-26B": EnhancedModelInfo(
        name="InternVL2-26B",
        full_name="OpenGVLab/InternVL2-26B", 
        description="Large InternVL2 model with comprehensive reasoning",
        parameters="26B",
        vram_gb=24.0,
        speed_rating=2,
        quality_rating=5,
        max_tokens=2048,
        supports_reasoning=True,
        recommended_use="Research-grade analysis with full reasoning chains"
    ),
    "InternVL-Chat": EnhancedModelInfo(
        name="InternVL-Chat-V1-5",
        full_name="OpenGVLab/InternVL-Chat-V1-5",
        description="Conversational InternVL with chat capabilities", 
        parameters="Unknown",
        vram_gb=16.0,
        speed_rating=3,
        quality_rating=4,
        max_tokens=2048,
        supports_reasoning=True,
        recommended_use="Interactive analysis and conversation"
    )
}

# Enhanced fallback models (VERIFIED WORKING)
ENHANCED_FALLBACKS = {
    "blip-large": EnhancedModelInfo(
        name="blip-image-captioning-large",
        full_name="Salesforce/blip-image-captioning-large",
        description="BLIP large with better descriptions",
        parameters="400M",
        vram_gb=2.0,
        speed_rating=4,
        quality_rating=3,
        max_tokens=256,
        supports_reasoning=False,
        recommended_use="Reliable fallback with good descriptions"
    ),
    "blip-base": EnhancedModelInfo(
        name="blip-image-captioning-base",
        full_name="Salesforce/blip-image-captioning-base", 
        description="Fast BLIP captioning",
        parameters="200M",
        vram_gb=1.0,
        speed_rating=5,
        quality_rating=2,
        max_tokens=128,
        supports_reasoning=False,
        recommended_use="Fastest fallback for basic descriptions"
    )
}

# Structured output templates
STRUCTURED_TEMPLATES = {
    "json": """Analyze this image and provide a detailed JSON response with the following structure:
{
  "scene_description": "Overall description of the scene",
  "objects": [
    {"name": "object_name", "location": "position", "description": "details"}
  ],
  "people": [
    {"count": number, "actions": ["action1", "action2"], "appearance": "description"}
  ],
  "setting": {
    "location_type": "indoor/outdoor/etc",
    "lighting": "description",
    "mood": "description"
  },
  "analysis": "Detailed analysis and insights"
}

Question: {prompt}""",

    "reasoning": """Analyze this image step by step using structured reasoning:

🔍 OBSERVATION:
- What do I see immediately?
- What stands out most?

🧠 ANALYSIS:
- What is happening in this scene?
- What are the relationships between elements?
- What context clues are present?

💭 REASONING:
- Why might this scene exist?
- What story does this image tell?
- What can I infer beyond what's visible?

📝 CONCLUSION:
- Summary of key findings
- Answer to the specific question

Question: {prompt}""",

    "detailed": """Provide an extremely detailed analysis of this image covering:

VISUAL ELEMENTS:
- Colors, lighting, composition
- Textures, patterns, materials
- Spatial relationships and perspective

CONTENT ANALYSIS:
- Objects and their properties
- People and their activities
- Setting and environment

CONTEXTUAL UNDERSTANDING:
- Purpose or function of the scene
- Time of day, season, weather
- Cultural or social context

SPECIFIC RESPONSE:
{prompt}

Please be thorough and descriptive in your analysis.""",

    "ultrathink": """🧠 ULTRA-DETAILED THINKING ANALYSIS 🧠

Step 1 - IMMEDIATE PERCEPTION:
What hits my visual cortex first? List everything I notice in the first 3 seconds.

Step 2 - SYSTEMATIC SCAN:
Methodically examine:
- Foreground elements (closest to camera)
- Middle ground elements  
- Background elements
- Left to right scan
- Top to bottom scan

Step 3 - DETAIL ANALYSIS:
For each significant element:
- Physical properties (size, color, texture, condition)
- Position and orientation
- Relationship to other elements
- Function or purpose

Step 4 - CONTEXTUAL REASONING:
- Where is this? (location type, setting)
- When is this? (time of day, season, era)
- What's happening? (activities, events, processes)
- Why does this scene exist? (purpose, intention)

Step 5 - INFERENCE AND INSIGHTS:
- What story does this tell?
- What happened before this moment?
- What might happen next?
- What emotions or atmosphere does this convey?

Step 6 - SPECIFIC QUESTION RESPONSE:
{prompt}

Step 7 - CONFIDENCE ASSESSMENT:
Rate confidence (1-10) for each major claim and explain reasoning.

Please think through each step thoroughly and provide rich, detailed analysis."""
}

class EnhancedVLMSelector:
    def __init__(self, model_key="8B", device='cuda', camera_device=0, 
                 output_format="standard", max_tokens=None, temperature=0.7):
        self.model_key = model_key
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.camera_device = camera_device
        self.output_format = output_format
        self.temperature = temperature
        
        # Get model info
        if model_key in ENHANCED_MODELS:
            self.model_info = ENHANCED_MODELS[model_key]
        elif model_key in ENHANCED_FALLBACKS:
            self.model_info = ENHANCED_FALLBACKS[model_key]
        else:
            raise ValueError(f"Unknown model: {model_key}")
        
        # Set response length
        self.max_tokens = max_tokens or self.model_info.max_tokens
        
        print(f"🚀 Enhanced VLM Model Selector")
        print(f"Selected Model: {self.model_info.name}")
        print(f"Parameters: {self.model_info.parameters}")
        print(f"Max Response Length: {self.max_tokens} tokens")
        print(f"Output Format: {output_format}")
        print(f"Supports Reasoning: {'Yes' if self.model_info.supports_reasoning else 'No'}")
        print(f"Device: {self.device}")
        print("")
        
        # Check VRAM
        if self.device == 'cuda':
            self.check_vram()
        
        # Initialize model
        self.init_enhanced_model()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_device)
        if not self.cap.isOpened():
            print(f"❌ Error: Cannot open camera {camera_device}")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Enhanced tracking
        self.analysis_queue = queue.Queue(maxsize=10)
        self.results_queue = queue.Queue()
        self.performance_stats = {
            'total_analyses': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_response_length': 0.0,
            'total_tokens': 0
        }
        
        # Control variables
        self.running = True
        self.continuous_mode = True
        self.analysis_rate = 0.3  # Slower for detailed analysis
        self.default_prompt = "Analyze this image in detail"
        
        # Start threads
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.analysis_thread = threading.Thread(target=self.analysis_loop)
        self.camera_thread.start()
        self.analysis_thread.start()
    
    def check_vram(self):
        """Enhanced VRAM checking"""
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            required_gb = self.model_info.vram_gb
            
            print(f"📊 GPU Memory Analysis:")
            print(f"   Available: {gpu_memory_gb:.1f}GB")
            print(f"   Required: {required_gb}GB")
            print(f"   Response Length: {self.max_tokens} tokens")
            
            if gpu_memory_gb < required_gb:
                print(f"⚠️  Warning: May not have enough VRAM for max quality")
                print(f"   Consider using CPU mode or smaller max_tokens")
            else:
                print(f"✅ Sufficient VRAM for enhanced processing")
            print("")
        except Exception as e:
            print(f"Could not check GPU memory: {e}")
    
    def init_enhanced_model(self):
        """Initialize model with enhanced capabilities"""
        print(f"🔄 Loading enhanced {self.model_info.name}...")
        start_time = time.time()
        
        try:
            # Try advanced VLM models first
            if "InternVL3" in self.model_info.full_name:
                try:
                    # Try direct model loading for more control
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_info.full_name,
                        trust_remote_code=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_info.full_name,
                        torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                        device_map=self.device,
                        trust_remote_code=True
                    )
                    self.model_type = "advanced_vlm"
                    print(f"✅ Advanced VLM model loaded")
                except Exception as e:
                    print(f"Advanced loading failed: {e}")
                    raise e
            else:
                # Use pipeline for fallback models
                self.model = pipeline(
                    "image-to-text",
                    model=self.model_info.full_name,
                    device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1,
                    max_new_tokens=self.max_tokens
                )
                self.model_type = "pipeline"
            
            load_time = time.time() - start_time
            print(f"✅ Enhanced model loaded in {load_time:.1f}s")
            print(f"   Type: {self.model_type}")
            print(f"   Max tokens: {self.max_tokens}")
            print("")
            
        except Exception as e:
            print(f"❌ Error loading {self.model_info.name}: {e}")
            print("🔄 Falling back to enhanced BLIP...")
            
            try:
                self.model_info = ENHANCED_FALLBACKS["blip-large"]
                self.model = pipeline(
                    "image-to-text",
                    model=self.model_info.full_name,
                    device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1,
                    max_new_tokens=self.max_tokens
                )
                self.model_type = "pipeline"
                print(f"✅ Enhanced fallback loaded")
            except Exception as e2:
                print(f"❌ Enhanced fallback failed: {e2}")
                sys.exit(1)
    
    def format_prompt(self, prompt):
        """Format prompt based on output format"""
        if self.output_format in STRUCTURED_TEMPLATES:
            return STRUCTURED_TEMPLATES[self.output_format].format(prompt=prompt)
        return prompt
    
    def camera_loop(self):
        """Enhanced camera loop with format info"""
        last_analysis_time = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Enhanced overlay
            self.draw_enhanced_overlay(frame)
            
            # Show frame
            cv2.imshow(f'Enhanced VLM - {self.model_info.name} [{self.output_format}]', frame)
            
            # Analysis timing
            current_time = time.time()
            if self.continuous_mode and (current_time - last_analysis_time) >= (1.0 / self.analysis_rate):
                if not self.analysis_queue.full():
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    formatted_prompt = self.format_prompt(self.default_prompt)
                    self.analysis_queue.put({
                        'image': rgb_frame,
                        'prompt': formatted_prompt,
                        'raw_prompt': self.default_prompt,
                        'timestamp': datetime.now()
                    })
                    last_analysis_time = current_time
            
            # Enhanced controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('p'):
                self.continuous_mode = not self.continuous_mode
                print(f"🔄 Continuous mode: {'ON' if self.continuous_mode else 'OFF'}")
            elif key == ord('s'):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                formatted_prompt = self.format_prompt(self.default_prompt)
                self.analysis_queue.put({
                    'image': rgb_frame,
                    'prompt': formatted_prompt,
                    'raw_prompt': self.default_prompt,
                    'timestamp': datetime.now()
                })
            elif key == ord('r'):
                self.reset_stats()
            elif key == ord('f'):
                self.cycle_format()
    
    def draw_enhanced_overlay(self, frame):
        """Enhanced overlay with format and token info"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Model info
        cv2.putText(frame, f"Model: {self.model_info.name}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Format: {self.output_format}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Max Tokens: {self.max_tokens}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Performance stats
        if self.performance_stats['total_analyses'] > 0:
            cv2.putText(frame, f"Avg Time: {self.performance_stats['avg_time']:.2f}s", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Avg Length: {self.performance_stats['avg_response_length']:.0f} chars", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Enhanced controls
        cv2.putText(frame, "Q:Quit P:Pause S:Single R:Reset F:Format", (20, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def cycle_format(self):
        """Cycle through output formats"""
        formats = ["standard", "json", "reasoning", "detailed", "ultrathink"]
        current_idx = formats.index(self.output_format) if self.output_format in formats else 0
        next_idx = (current_idx + 1) % len(formats)
        self.output_format = formats[next_idx]
        print(f"🔄 Switched to format: {self.output_format}")
    
    def analyze_image_enhanced(self, image, prompt):
        """Enhanced image analysis with longer responses"""
        try:
            from PIL import Image as PILImage
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            
            if self.model_type == "advanced_vlm":
                # Use advanced model with full control
                inputs = self.processor(
                    text=prompt,
                    images=pil_image,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                # Clean up response
                if prompt in response:
                    response = response.split(prompt)[-1].strip()
                
                return response
            else:
                # Use pipeline
                result = self.model(pil_image, max_new_tokens=self.max_tokens)
                return result[0]['generated_text'] if result else "No description"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analysis_loop(self):
        """Enhanced analysis loop with token tracking"""
        while self.running:
            try:
                task = self.analysis_queue.get(timeout=1.0)
                
                # Perform enhanced analysis
                start_time = time.time()
                result = self.analyze_image_enhanced(task['image'], task['prompt'])
                processing_time = time.time() - start_time
                
                # Update enhanced stats
                self.update_enhanced_stats(processing_time, result)
                
                # Create enhanced analysis message
                analysis_msg = {
                    'timestamp': task['timestamp'].isoformat(),
                    'model': self.model_info.name,
                    'output_format': self.output_format,
                    'raw_prompt': task['raw_prompt'],
                    'formatted_prompt': task['prompt'],
                    'response': result,
                    'processing_time': processing_time,
                    'response_length': len(result),
                    'max_tokens': self.max_tokens,
                    'device': self.device
                }
                
                # Enhanced output display
                print(f"\n🧠 === Enhanced {self.model_info.name} Analysis ===")
                print(f"⏰ Time: {task['timestamp'].strftime('%H:%M:%S')}")
                print(f"🎯 Format: {self.output_format}")
                print(f"❓ Prompt: {task['raw_prompt']}")
                print(f"📝 Response ({len(result)} chars):")
                print("-" * 60)
                print(result)
                print("-" * 60)
                print(f"⚡ Processing: {processing_time:.3f}s")
                print(f"📊 Avg Speed: {self.performance_stats['avg_time']:.3f}s")
                print(f"📏 Avg Length: {self.performance_stats['avg_response_length']:.0f} chars")
                print("=" * 60)
                
                self.results_queue.put(analysis_msg)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Enhanced analysis error: {e}")
    
    def update_enhanced_stats(self, processing_time, response):
        """Update enhanced performance statistics"""
        self.performance_stats['total_analyses'] += 1
        self.performance_stats['total_time'] += processing_time
        self.performance_stats['avg_time'] = (
            self.performance_stats['total_time'] / self.performance_stats['total_analyses']
        )
        self.performance_stats['min_time'] = min(self.performance_stats['min_time'], processing_time)
        self.performance_stats['max_time'] = max(self.performance_stats['max_time'], processing_time)
        
        # Response length tracking
        response_length = len(response)
        self.performance_stats['total_tokens'] += response_length
        self.performance_stats['avg_response_length'] = (
            self.performance_stats['total_tokens'] / self.performance_stats['total_analyses']
        )
    
    def reset_stats(self):
        """Reset enhanced statistics"""
        self.performance_stats = {
            'total_analyses': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_response_length': 0.0,
            'total_tokens': 0
        }
        print("📊 Enhanced performance stats reset")
    
    def set_prompt(self, prompt, rate=None):
        """Set prompt with format awareness"""
        self.default_prompt = prompt
        if rate is not None:
            self.analysis_rate = rate
        print(f"🔄 Updated: Prompt='{prompt}', Rate={self.analysis_rate}Hz, Format={self.output_format}")
    
    def cleanup(self):
        """Enhanced cleanup with detailed logging"""
        self.running = False
        self.camera_thread.join()
        self.analysis_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save enhanced results
        if not self.results_queue.empty():
            results = []
            while not self.results_queue.empty():
                results.append(self.results_queue.get())
            
            # Enhanced summary
            summary = {
                'session_info': {
                    'model': self.model_info.name,
                    'parameters': self.model_info.parameters,
                    'device': self.device,
                    'output_format': self.output_format,
                    'max_tokens': self.max_tokens,
                    'temperature': self.temperature,
                    'supports_reasoning': self.model_info.supports_reasoning
                },
                'performance_stats': self.performance_stats,
                'analyses': results
            }
            
            filename = f"enhanced_vlm_{self.model_info.name}_{self.output_format}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n📁 Enhanced results saved to: {filename}")
            print(f"📊 Enhanced Performance Summary:")
            print(f"   Model: {self.model_info.name}")
            print(f"   Output Format: {self.output_format}")
            print(f"   Total Analyses: {self.performance_stats['total_analyses']}")
            print(f"   Average Time: {self.performance_stats['avg_time']:.3f}s")
            print(f"   Average Response Length: {self.performance_stats['avg_response_length']:.0f} chars")
            print(f"   Min Time: {self.performance_stats['min_time']:.3f}s")
            print(f"   Max Time: {self.performance_stats['max_time']:.3f}s")

def show_enhanced_menu():
    """Display enhanced models and capabilities"""
    print("\n🧠 Enhanced InternVL3 Models with Long Responses:")
    print("=" * 90)
    
    for key, model in ENHANCED_MODELS.items():
        print(f"🔹 {key}: {model.name}")
        print(f"   Parameters: {model.parameters}")
        print(f"   VRAM: {model.vram_gb}GB")
        print(f"   Max Response: {model.max_tokens} tokens")
        print(f"   Speed: {'⭐' * model.speed_rating} ({model.speed_rating}/5)")
        print(f"   Quality: {'⭐' * model.quality_rating} ({model.quality_rating}/5)")
        print(f"   Reasoning: {'Yes' if model.supports_reasoning else 'No'}")
        print(f"   Use Case: {model.recommended_use}")
        print()
    
    print("🎯 Output Formats:")
    for fmt, template in STRUCTURED_TEMPLATES.items():
        print(f"   {fmt}: {template.split('Question:')[0][:100]}...")
    print()
    
    print("🔹 Enhanced Fallback Models:")
    for key, model in ENHANCED_FALLBACKS.items():
        print(f"   {key}: {model.name} ({model.parameters}, {model.max_tokens} tokens)")
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description='Enhanced VLM with Long Responses and Structured Output')
    parser.add_argument('--model', default='8B',
                       choices=list(ENHANCED_MODELS.keys()) + list(ENHANCED_FALLBACKS.keys()),
                       help='Model to use (default: 8B)')
    parser.add_argument('--list-models', action='store_true',
                       help='Show enhanced models and exit')
    parser.add_argument('--format', default='standard', 
                       choices=['standard', 'json', 'reasoning', 'detailed', 'ultrathink'],
                       help='Output format (default: standard)')
    parser.add_argument('--max-tokens', type=int,
                       help='Override max response length')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Response creativity (0.0-1.0)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--rate', type=float, default=0.3,
                       help='Analysis rate in Hz (slower for detailed analysis)')
    parser.add_argument('--prompt', default="Analyze this image in detail",
                       help='Default analysis prompt')
    
    args = parser.parse_args()
    
    if args.list_models:
        show_enhanced_menu()
        return
    
    print("🧠 Enhanced VLM Model Selector")
    print("Long responses, structured output, and reasoning chains!")
    print("=" * 70)
    
    # Check environment
    if 'CONDA_PREFIX' in os.environ and 'vlm' in os.environ['CONDA_PREFIX']:
        print("✅ VLM conda environment detected")
    else:
        print("⚠️  Warning: VLM conda environment not active")
    
    print("\n🎮 Enhanced Controls:")
    print("   q - Quit")
    print("   p - Toggle continuous mode")
    print("   s - Single analysis")
    print("   r - Reset performance stats")
    print("   f - Cycle output format")
    print("")
    
    # Initialize enhanced selector
    try:
        vlm = EnhancedVLMSelector(
            model_key=args.model,
            device=args.device,
            camera_device=args.camera,
            output_format=args.format,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        vlm.set_prompt(args.prompt, args.rate)
        
        # Keep main thread alive
        while vlm.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down enhanced VLM...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'vlm' in locals():
            vlm.cleanup()

if __name__ == '__main__':
    main()
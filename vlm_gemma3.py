#!/usr/bin/env python3
"""
VLM with Gemma3 - Fail Hard Implementation
No fallbacks - uses Gemma3 or fails completely
"""

import cv2
import torch
import numpy as np
import threading
import queue
import time
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse
import sys
import os
from dataclasses import dataclass
from typing import Dict, List
import gc

@dataclass
class Gemma3ModelInfo:
    name: str
    full_name: str
    description: str
    parameters: str
    vram_gb: float
    max_tokens: int
    context_length: int

# Gemma3 Models (FAIL HARD - NO FALLBACKS)
GEMMA3_MODELS = {
    "gemma3-2b": Gemma3ModelInfo(
        name="Gemma3-2B",
        full_name="google/gemma-2-2b",
        description="Latest Gemma3 2B model with instruction following",
        parameters="2B",
        vram_gb=4.0,
        max_tokens=2048,
        context_length=8192
    ),
    "gemma3-9b": Gemma3ModelInfo(
        name="Gemma3-9B", 
        full_name="google/gemma-2-9b",
        description="Gemma3 9B model for detailed analysis",
        parameters="9B",
        vram_gb=16.0,
        max_tokens=4096,
        context_length=8192
    ),
    "gemma3-27b": Gemma3ModelInfo(
        name="Gemma3-27B",
        full_name="google/gemma-2-27b",
        description="Largest Gemma3 model for maximum quality",
        parameters="27B", 
        vram_gb=32.0,
        max_tokens=8192,
        context_length=8192
    )
}

class VLMGemma3:
    def __init__(self, model_key="gemma3-2b", device='cuda', camera_device=0, 
                 max_tokens=None, temperature=0.7, use_vision_encoder="blip"):
        """
        Initialize VLM with Gemma3 - FAIL HARD MODE
        
        Args:
            model_key: Which Gemma3 model to use
            device: cuda or cpu
            camera_device: camera index
            max_tokens: max response length
            temperature: sampling temperature
            use_vision_encoder: vision encoder to pair with Gemma3
        """
        
        self.model_key = model_key
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.camera_device = camera_device
        self.temperature = temperature
        self.use_vision_encoder = use_vision_encoder
        
        # Get model info - FAIL if model doesn't exist
        if model_key not in GEMMA3_MODELS:
            self.fail_hard(f"Unknown Gemma3 model: {model_key}. Available: {list(GEMMA3_MODELS.keys())}")
        
        self.model_info = GEMMA3_MODELS[model_key]
        self.max_tokens = max_tokens or self.model_info.max_tokens
        
        print(f"🚀 VLM with Gemma3 (Fail Hard Mode)")
        print(f"Selected Model: {self.model_info.name}")
        print(f"Parameters: {self.model_info.parameters}")
        print(f"Max Tokens: {self.max_tokens}")
        print(f"Context Length: {self.model_info.context_length}")
        print(f"Vision Encoder: {use_vision_encoder}")
        print(f"Device: {self.device}")
        print("")
        
        # HARD REQUIREMENT CHECK
        self.check_requirements_or_fail()
        
        # Initialize models - FAIL HARD if any step fails
        self.init_gemma3_model()
        self.init_vision_encoder()
        
        # Initialize camera - FAIL HARD if camera not available
        self.init_camera()
        
        # Performance tracking
        self.analysis_queue = queue.Queue(maxsize=10)
        self.results_queue = queue.Queue()
        self.performance_stats = {
            'total_analyses': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_response_length': 0.0,
            'total_chars': 0,
            'model_name': self.model_info.name
        }
        
        # Control variables
        self.running = True
        self.continuous_mode = True
        self.analysis_rate = 0.3  # Slower for detailed analysis
        self.default_prompt = "Analyze this image in detail and provide comprehensive insights"
        
        print("✅ Gemma3 VLM initialized successfully - NO FALLBACKS!")
        print("   This system will FAIL HARD if anything goes wrong")
        print("   Gemma3 model loaded and ready for vision analysis")
        print("")
        
        # Start threads
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.analysis_thread = threading.Thread(target=self.analysis_loop)
        self.camera_thread.start()
        self.analysis_thread.start()
    
    def fail_hard(self, message):
        """FAIL HARD - no recovery, no fallbacks"""
        print(f"\n💥 HARD FAILURE - SYSTEM STOPPING")
        print(f"❌ ERROR: {message}")
        print(f"🚫 NO FALLBACKS AVAILABLE")
        print(f"🛑 Fix the issue and try again")
        print("")
        sys.exit(1)
    
    def check_requirements_or_fail(self):
        """Check all requirements - FAIL HARD if any missing"""
        print("🔍 Checking requirements (FAIL HARD MODE)...")
        
        # Check transformers version
        try:
            import transformers
            version = transformers.__version__
            major, minor = map(int, version.split('.')[:2])
            if major < 4 or (major == 4 and minor < 38):
                self.fail_hard(f"transformers version {version} too old. Need >= 4.38.0")
            print(f"✅ transformers {version} OK")
        except Exception as e:
            self.fail_hard(f"transformers import failed: {e}")
        
        # Check torch version
        try:
            torch_version = torch.__version__
            print(f"✅ torch {torch_version} OK")
        except Exception as e:
            self.fail_hard(f"torch import failed: {e}")
        
        # Check VRAM if using GPU
        if self.device == 'cuda':
            if not torch.cuda.is_available():
                self.fail_hard("CUDA requested but not available")
            
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                required_gb = self.model_info.vram_gb
                
                print(f"📊 GPU Memory Check:")
                print(f"   Available: {gpu_memory_gb:.1f}GB")
                print(f"   Required: {required_gb}GB")
                
                if gpu_memory_gb < required_gb:
                    self.fail_hard(f"Insufficient VRAM: need {required_gb}GB, have {gpu_memory_gb:.1f}GB")
                
                print(f"✅ VRAM check passed")
            except Exception as e:
                self.fail_hard(f"GPU memory check failed: {e}")
        
        print("✅ All requirements satisfied")
    
    def init_gemma3_model(self):
        """Initialize Gemma3 model - FAIL HARD if it doesn't work"""
        print(f"🔄 Loading Gemma3 model: {self.model_info.full_name}")
        start_time = time.time()
        
        try:
            # Load tokenizer
            print("   Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_info.full_name,
                trust_remote_code=True
            )
            
            # Ensure tokenizer has proper tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            print("   Loading model...")
            if self.device == 'cuda':
                self.gemma3_model = AutoModelForCausalLM.from_pretrained(
                    self.model_info.full_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
                )
            else:
                self.gemma3_model = AutoModelForCausalLM.from_pretrained(
                    self.model_info.full_name,
                    torch_dtype=torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
            
            load_time = time.time() - start_time
            print(f"✅ Gemma3 model loaded in {load_time:.1f}s")
            
            # Test the model
            print("   Testing model...")
            test_input = self.tokenizer("Hello", return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.gemma3_model.generate(
                    **test_input,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            test_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"✅ Model test passed: '{test_response[:50]}...'")
            
        except Exception as e:
            self.fail_hard(f"Failed to load Gemma3 model: {e}")
    
    def init_vision_encoder(self):
        """Initialize vision encoder - FAIL HARD if it doesn't work"""
        print(f"🔄 Loading vision encoder: {self.use_vision_encoder}")
        
        try:
            if self.use_vision_encoder == "blip":
                self.vision_model = pipeline(
                    "image-to-text",
                    model="Salesforce/blip-image-captioning-large",
                    device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1
                )
            elif self.use_vision_encoder == "git":
                self.vision_model = pipeline(
                    "image-to-text",
                    model="microsoft/git-large-coco",
                    device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1
                )
            else:
                self.fail_hard(f"Unknown vision encoder: {self.use_vision_encoder}")
            
            print(f"✅ Vision encoder loaded: {self.use_vision_encoder}")
            
            # Test vision encoder
            print("   Testing vision encoder...")
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            from PIL import Image as PILImage
            test_pil = PILImage.fromarray(test_image)
            test_caption = self.vision_model(test_pil)
            print(f"✅ Vision encoder test passed")
            
        except Exception as e:
            self.fail_hard(f"Failed to load vision encoder: {e}")
    
    def init_camera(self):
        """Initialize camera - FAIL HARD if camera not available"""
        print(f"🔄 Initializing camera {self.camera_device}")
        
        try:
            self.cap = cv2.VideoCapture(self.camera_device)
            if not self.cap.isOpened():
                self.fail_hard(f"Cannot open camera {self.camera_device}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test camera
            ret, frame = self.cap.read()
            if not ret:
                self.fail_hard(f"Camera {self.camera_device} not working - cannot read frames")
            
            print(f"✅ Camera initialized: {frame.shape}")
            
        except Exception as e:
            self.fail_hard(f"Camera initialization failed: {e}")
    
    def camera_loop(self):
        """Camera loop with Gemma3 overlay"""
        last_analysis_time = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.fail_hard("Camera stopped working during operation")
                
                # Draw Gemma3 overlay
                self.draw_gemma3_overlay(frame)
                
                # Show frame
                cv2.imshow(f'Gemma3 VLM - {self.model_info.name} [FAIL HARD]', frame)
                
                # Analysis timing
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
                
                # Controls
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
                
            except Exception as e:
                self.fail_hard(f"Camera loop error: {e}")
    
    def draw_gemma3_overlay(self, frame):
        """Draw Gemma3-specific overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Model info
        cv2.putText(frame, f"Gemma3: {self.model_info.name}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Mode: FAIL HARD", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Tokens: {self.max_tokens}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Performance stats
        if self.performance_stats['total_analyses'] > 0:
            cv2.putText(frame, f"Avg: {self.performance_stats['avg_time']:.2f}s", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Length: {self.performance_stats['avg_response_length']:.0f} chars", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "Q:Quit P:Pause S:Single R:Reset [FAIL HARD]", (20, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def analyze_with_gemma3(self, image, prompt):
        """Analyze image using vision encoder + Gemma3 - FAIL HARD on errors"""
        try:
            # Step 1: Get image description from vision encoder
            from PIL import Image as PILImage
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            
            vision_result = self.vision_model(pil_image)
            if not vision_result or len(vision_result) == 0:
                self.fail_hard("Vision encoder returned empty result")
            
            image_description = vision_result[0]['generated_text']
            
            # Step 2: Create enhanced prompt for Gemma3
            enhanced_prompt = f"""<start_of_turn>user
Image Description: {image_description}

User Request: {prompt}

Please provide a detailed analysis based on the image description above. Be comprehensive, insightful, and specific in your response.
<end_of_turn>
<start_of_turn>model
"""
            
            # Step 3: Use Gemma3 for detailed analysis
            inputs = self.tokenizer(
                enhanced_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.model_info.context_length - self.max_tokens
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.gemma3_model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the model's response
            if "<start_of_turn>model" in full_response:
                response = full_response.split("<start_of_turn>model")[-1].strip()
            else:
                response = full_response[len(enhanced_prompt):].strip()
            
            if len(response) < 10:
                self.fail_hard(f"Gemma3 response too short: '{response}'")
            
            return response
            
        except Exception as e:
            self.fail_hard(f"Gemma3 analysis failed: {e}")
    
    def analysis_loop(self):
        """Analysis loop with Gemma3 - FAIL HARD on errors"""
        while self.running:
            try:
                task = self.analysis_queue.get(timeout=1.0)
                
                # Perform Gemma3 analysis
                start_time = time.time()
                result = self.analyze_with_gemma3(task['image'], task['prompt'])
                processing_time = time.time() - start_time
                
                # Update stats
                self.update_stats(processing_time, result)
                
                # Create analysis message
                analysis_msg = {
                    'timestamp': task['timestamp'].isoformat(),
                    'model': self.model_info.name,
                    'prompt': task['prompt'],
                    'response': result,
                    'processing_time': processing_time,
                    'response_length': len(result),
                    'max_tokens': self.max_tokens,
                    'device': self.device,
                    'mode': 'FAIL_HARD'
                }
                
                # Display result
                print(f"\n🧠 === Gemma3 Analysis (FAIL HARD) ===")
                print(f"⏰ Time: {task['timestamp'].strftime('%H:%M:%S')}")
                print(f"❓ Prompt: {task['prompt']}")
                print(f"📝 Gemma3 Response ({len(result)} chars):")
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
                self.fail_hard(f"Analysis loop error: {e}")
    
    def update_stats(self, processing_time, response):
        """Update performance statistics"""
        self.performance_stats['total_analyses'] += 1
        self.performance_stats['total_time'] += processing_time
        self.performance_stats['avg_time'] = (
            self.performance_stats['total_time'] / self.performance_stats['total_analyses']
        )
        self.performance_stats['min_time'] = min(self.performance_stats['min_time'], processing_time)
        self.performance_stats['max_time'] = max(self.performance_stats['max_time'], processing_time)
        
        response_length = len(response)
        self.performance_stats['total_chars'] += response_length
        self.performance_stats['avg_response_length'] = (
            self.performance_stats['total_chars'] / self.performance_stats['total_analyses']
        )
    
    def reset_stats(self):
        """Reset performance statistics"""
        old_model = self.performance_stats['model_name']
        self.performance_stats = {
            'total_analyses': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_response_length': 0.0,
            'total_chars': 0,
            'model_name': old_model
        }
        print(f"📊 Gemma3 performance stats reset")
    
    def set_prompt(self, prompt, rate=None):
        """Set prompt and rate"""
        self.default_prompt = prompt
        if rate is not None:
            self.analysis_rate = rate
        print(f"🔄 Gemma3 Updated: Prompt='{prompt}', Rate={self.analysis_rate}Hz")
    
    def cleanup(self):
        """Clean up resources"""
        try:
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
                
                summary = {
                    'session_info': {
                        'model': self.model_info.name,
                        'parameters': self.model_info.parameters,
                        'device': self.device,
                        'mode': 'FAIL_HARD',
                        'max_tokens': self.max_tokens,
                        'vision_encoder': self.use_vision_encoder
                    },
                    'performance_stats': self.performance_stats,
                    'analyses': results
                }
                
                filename = f"gemma3_vlm_{self.model_info.name.replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"\n📁 Gemma3 results saved to: {filename}")
                print(f"📊 Gemma3 Performance Summary:")
                print(f"   Model: {self.model_info.name}")
                print(f"   Analyses: {self.performance_stats['total_analyses']}")
                print(f"   Avg Time: {self.performance_stats['avg_time']:.3f}s")
                print(f"   Avg Length: {self.performance_stats['avg_response_length']:.0f} chars")
                print(f"   Mode: FAIL HARD (no fallbacks)")
                
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

def show_gemma3_models():
    """Display available Gemma3 models"""
    print("\n🔥 Gemma3 Models (FAIL HARD MODE):")
    print("=" * 60)
    
    for key, model in GEMMA3_MODELS.items():
        print(f"🔹 {key}: {model.name}")
        print(f"   Parameters: {model.parameters}")
        print(f"   VRAM: {model.vram_gb}GB")
        print(f"   Max Response: {model.max_tokens} tokens")
        print(f"   Context: {model.context_length} tokens")
        print(f"   Description: {model.description}")
        print()
    
    print("⚠️  FAIL HARD MODE: No fallbacks, no recovery")
    print("   System will exit immediately on any error")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='VLM with Gemma3 - Fail Hard Mode')
    parser.add_argument('--model', default='gemma3-2b',
                       choices=list(GEMMA3_MODELS.keys()),
                       help='Gemma3 model to use (default: gemma3-2b)')
    parser.add_argument('--list-models', action='store_true',
                       help='Show Gemma3 models and exit')
    parser.add_argument('--max-tokens', type=int,
                       help='Override max response length')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (0.0-1.0)')
    parser.add_argument('--vision-encoder', default='blip', choices=['blip', 'git'],
                       help='Vision encoder to use with Gemma3')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--rate', type=float, default=0.3,
                       help='Analysis rate in Hz')
    parser.add_argument('--prompt', default="Provide a comprehensive analysis of this image",
                       help='Default analysis prompt')
    
    args = parser.parse_args()
    
    if args.list_models:
        show_gemma3_models()
        return
    
    print("🔥 VLM with Gemma3 - FAIL HARD MODE")
    print("=" * 50)
    print("⚠️  WARNING: This system has NO FALLBACKS")
    print("   Any error will cause immediate termination")
    print("   Gemma3 must work perfectly or system fails")
    print("")
    
    # Check environment
    if 'CONDA_PREFIX' in os.environ and 'vlm' in os.environ['CONDA_PREFIX']:
        print("✅ VLM conda environment detected")
    else:
        print("⚠️  Warning: VLM conda environment not active")
    
    print("\n🎮 Controls:")
    print("   q - Quit")
    print("   p - Toggle continuous mode")  
    print("   s - Single analysis")
    print("   r - Reset performance stats")
    print("")
    
    # Initialize Gemma3 VLM - FAIL HARD
    try:
        vlm = VLMGemma3(
            model_key=args.model,
            device=args.device,
            camera_device=args.camera,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_vision_encoder=args.vision_encoder
        )
        
        vlm.set_prompt(args.prompt, args.rate)
        
        print("🚀 Gemma3 VLM ready - operating in FAIL HARD mode!")
        print("   Any error will terminate the system immediately")
        print("")
        
        # Keep main thread alive
        while vlm.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 User interrupted Gemma3 VLM")
    except Exception as e:
        print(f"\n💥 GEMMA3 VLM HARD FAILURE: {e}")
        print("🚫 System terminated - no recovery possible")
        sys.exit(1)
    finally:
        if 'vlm' in locals():
            vlm.cleanup()

if __name__ == '__main__':
    main()
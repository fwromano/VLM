#!/usr/bin/env python3
"""
Simple VLM Program using vLLM and Gemma 3
=========================================

A minimal VLM (Vision-Language Model) implementation that:
- Uses vLLM for high-performance inference
- Runs Gemma 3 multimodal models
- Processes images and answers questions about them
- Optimized for RTX 5000 Ada GPU

Usage:
    python vlm_simple.py --image path/to/image.jpg --question "What do you see?"
    python vlm_simple.py --interactive  # Interactive mode with camera
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Optional, List
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# Set CUDA environment
os.environ.update({
    'CUDA_HOME': '/usr',
    'CUDA_ROOT': '/usr', 
    'CUDA_PATH': '/usr',
    'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu',
    'CUDA_VISIBLE_DEVICES': '0',
    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'
})

try:
    import torch
    from vllm import LLM, SamplingParams
    from PIL import Image
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install vllm pillow opencv-python")
    sys.exit(1)

class SimpleVLM:
    """Simple VLM using vLLM for high-performance inference"""
    
    def __init__(self, model_name: str = "google/gemma-3-4b-it"):
        """
        Initialize the VLM
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.llm = None
        self.sampling_params = None
        self.setup_gpu()
        
    def setup_gpu(self):
        """Check GPU availability and setup"""
        if torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {self.gpu_memory:.1f}GB")
        else:
            print("No GPU detected, using CPU (not recommended)")
            self.device_count = 0
            self.gpu_memory = 0
    
    def load_model(self):
        """Load the model using vLLM"""
        print(f"Loading {self.model_name} with vLLM...")
        
        try:
            # Configure vLLM for optimal performance with multimodal support
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,  # Single GPU
                gpu_memory_utilization=0.80,  # Use 80% of VRAM (leave room for images)
                max_model_len=4096,  # Increased for multimodal inputs
                enforce_eager=False,  # Use CUDA graphs for speed
                trust_remote_code=True,  # Required for Gemma 3
                limit_mm_per_prompt={"image": 1},  # Limit to 1 image per prompt
            )
            
            # Configure sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.1,  # Low temperature for consistent responses
                top_p=0.9,
                max_tokens=256,  # Reasonable response length
                stop=["<eos>", "</s>"],  # Stop tokens
            )
            
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def preprocess_image(self, image_input) -> Image.Image:
        """
        Preprocess image for the model
        
        Args:
            image_input: Path to image file, PIL Image, or numpy array
            
        Returns:
            Preprocessed PIL Image
        """
        # Handle different input types
        if isinstance(image_input, str) or isinstance(image_input, Path):
            # File path
            image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # Numpy array (from camera)
            if image_input.dtype != np.uint8:
                image_input = (image_input * 255).astype(np.uint8)
            # Convert BGR to RGB if needed
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            # Already a PIL Image
            image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to optimal size for Gemma 3 (896x896)
        target_size = 896
        width, height = image.size
        
        if width != target_size or height != target_size:
            # Create square image with padding to maintain aspect ratio
            max_dim = max(width, height)
            square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
            
            # Paste original image centered
            paste_x = (max_dim - width) // 2
            paste_y = (max_dim - height) // 2
            square_img.paste(image, (paste_x, paste_y))
            
            # Resize to target resolution
            image = square_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        return image
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def ask_about_image(self, image_input, question: str) -> str:
        """
        Ask a question about an image
        
        Args:
            image_input: Image input (path, PIL Image, or numpy array)
            question: Question to ask about the image
            
        Returns:
            Model's response
        """
        # Try server first if configured
        server_url = os.environ.get('VLM_SERVER_URL', '')
        if server_url:
            try:
                # Convert to PIL
                if isinstance(image_input, np.ndarray):
                    arr = image_input
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).astype(np.uint8)
                    pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                elif isinstance(image_input, Image.Image):
                    pil_img = image_input.convert('RGB')
                else:
                    pil_img = Image.open(image_input).convert('RGB')
                from io import BytesIO
                buf = BytesIO()
                pil_img.save(buf, format='JPEG', quality=90)
                payload = {
                    'image_base64': base64.b64encode(buf.getvalue()).decode('utf-8'),
                    'question': question,
                }
                import urllib.request as _urlreq, urllib.parse as _urlparse, json as _json
                if os.environ.get('VLM_SERVER_MODEL'):
                    payload['model'] = os.environ['VLM_SERVER_MODEL']
                if os.environ.get('VLM_SERVER_BACKEND'):
                    payload['backend'] = os.environ['VLM_SERVER_BACKEND']
                if os.environ.get('VLM_SERVER_FAST', '0') in ('1','true','TRUE'):
                    payload['fast'] = True
                data = _urlparse.urlencode(payload).encode('utf-8')
                req = _urlreq.Request(server_url.rstrip('/') + '/v1/vision/analyze', data=data)
                with _urlreq.urlopen(req, timeout=10) as resp:
                    out = _json.loads(resp.read().decode('utf-8'))
                    if 'text' in out:
                        return out['text']
            except Exception:
                pass

        if not self.llm:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        image = self.preprocess_image(image_input)
        
        try:
            # For vLLM multimodal, use the prompt with image data
            # Note: vLLM's multimodal API format depends on the specific model
            
            # Method 1: Try vLLM's multimodal input format
            try:
                from vllm.inputs import TextPrompt, ImagePrompt
                
                # Create multimodal prompt
                prompt = TextPrompt(prompt=f"<image>\nUser: {question}\nAssistant:")
                image_prompt = ImagePrompt(prompt=prompt, multi_modal_data={"image": image})
                
                outputs = self.llm.generate([image_prompt], self.sampling_params)
                response = outputs[0].outputs[0].text.strip()
                return response
                
            except (ImportError, AttributeError):
                # Method 2: Try direct generate with multimodal data
                from vllm import MultiModalData
                
                mm_data = MultiModalData(type="image", data=image)
                outputs = self.llm.generate(
                    prompts=[f"<image>\nUser: {question}\nAssistant:"],
                    sampling_params=self.sampling_params,
                    multi_modal_data=[mm_data]
                )
                response = outputs[0].outputs[0].text.strip()
                return response
                
        except Exception as e:
            # If all multimodal methods fail, provide text-only response
            try:
                print(f"Multimodal processing failed: {str(e)}")
                print("Falling back to text-only mode...")
                
                prompt = f"User: {question}\nAssistant: I can see you're asking about an image, but I'm currently running in text-only mode. "
                outputs = self.llm.generate([prompt], self.sampling_params)
                response = outputs[0].outputs[0].text.strip()
                return f"[Text-only mode] {response}"
                
            except Exception as e2:
                return f"Both multimodal and text processing failed. Multimodal error: {str(e)} | Text error: {str(e2)}"
    
    def interactive_camera_mode(self):
        """Interactive mode using camera input"""
        print("\n🎥 Starting interactive camera mode...")
        print("Commands:")
        print("  - Type a question and press Enter")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Press 'c' to capture current frame")
        print("\n")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized")
        current_frame = None
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if ret:
                    current_frame = frame.copy()
                    
                    # Show frame (optional - comment out if running headless)
                    try:
                        cv2.imshow('VLM Camera Feed', frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('c'):
                            print("Frame captured!")
                    except:
                        # Ignore display errors for headless systems
                        pass
                
                # Check for user input (non-blocking)
                try:
                    import select
                    if select.select([sys.stdin], [], [], 0)[0]:
                        question = input("Ask about the image: ").strip()
                        
                        if question.lower() in ['quit', 'exit']:
                            break
                        
                        if question and current_frame is not None:
                            print("Analyzing...")
                            response = self.ask_about_image(current_frame, question)
                            print(f"VLM: {response}\n")
                        elif current_frame is None:
                            print("No frame captured yet")
                        
                except (ImportError, OSError):
                    # Fallback for systems without select
                    question = input("Ask about the image (or 'quit' to exit): ").strip()
                    
                    if question.lower() in ['quit', 'exit']:
                        break
                    
                    if question and current_frame is not None:
                        print("Analyzing...")
                        response = self.ask_about_image(current_frame, question)
                        print(f"VLM: {response}\n")
                    elif current_frame is None:
                        print("No frame captured yet")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera mode ended")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Simple VLM using vLLM and Gemma 3")
    
    # Model selection
    parser.add_argument(
        '--model', 
        default='google/gemma-3-4b-it',
        help='Model to use (default: google/gemma-3-4b-it)'
    )
    
    # Image and question mode
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--question', help='Question to ask about the image')
    
    # Interactive mode
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='Start interactive camera mode'
    )
    
    # Quick test mode
    parser.add_argument(
        '--test',
        action='store_true', 
        help='Run a quick test with a sample image'
    )
    
    args = parser.parse_args()
    
    # Initialize VLM
    print("Initializing Simple VLM...")
    vlm = SimpleVLM(model_name=args.model)
    
    if not vlm.load_model():
        sys.exit(1)
    
    # Choose mode based on arguments
    if args.interactive:
        vlm.interactive_camera_mode()
    
    elif args.image and args.question:
        if not Path(args.image).exists():
            print(f"Image file not found: {args.image}")
            sys.exit(1)
        
        print(f"Image: {args.image}")
        print(f"Question: {args.question}")
        print("Analyzing...")
        
        response = vlm.ask_about_image(args.image, args.question)
        print(f"VLM: {response}")
    
    elif args.test:
        # Create a simple test image
        print("🧪 Running test mode...")
        test_image = Image.new('RGB', (400, 300), color='blue')
        response = vlm.ask_about_image(test_image, "What color is this image?")
        print(f"VLM: {response}")
    
    else:
        print("Please specify either --interactive, --image + --question, or --test")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

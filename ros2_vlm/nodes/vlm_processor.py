#!/usr/bin/env python3
"""
VLM Processor - Runs in conda environment
Called by ROS2 node via subprocess
"""

import sys
import os
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def setup_cuda():
    """Setup CUDA environment"""
    os.environ.update({
        'CUDA_HOME': '/usr',
        'CUDA_ROOT': '/usr', 
        'CUDA_PATH': '/usr',
        'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu',
        'CUDA_VISIBLE_DEVICES': '0',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'
    })

def load_vlm_model():
    """Load VLM model"""
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_available = torch.cuda.is_available()
    
    print(f"Loading Gemma 3 4B on {device}...", file=sys.stderr)
    
    if gpu_available:
        # Clear GPU cache first
        torch.cuda.empty_cache()
        
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory: {gpu_memory:.1f}GB", file=sys.stderr)
        
        # For RTX 5000 Ada (16GB), use explicit device mapping
        if gpu_memory >= 15:  # Use full GPU for 16GB+ cards
            device_map = {"": 0}  # Put entire model on GPU 0
            torch_dtype = torch.bfloat16
        else:  # Fallback for smaller GPUs
            device_map = "auto"
            torch_dtype = torch.float16
    else:
        device_map = None
        torch_dtype = torch.float32
    
    # Load model and processor
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it",
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    
    # For CPU fallback, move model explicitly
    if not gpu_available:
        model = model.to(device)
    
    print(f"Model loaded on {device}", file=sys.stderr)
    return model, processor, device

def process_image(model, processor, device, image_path, question):
    """Process image with VLM"""
    # Load image
    pil_image = Image.open(image_path).convert('RGB')
    
    # Create message format for Gemma 3
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": question}
        ]
    }]
    
    # Apply chat template and process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[pil_image], return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.1,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode output
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove input)
    input_length = len(processor.decode(inputs["input_ids"][0], skip_special_tokens=True))
    generated_text = response[input_length:].strip()
    
    return generated_text

def main():
    if len(sys.argv) != 3:
        print("Usage: vlm_processor.py <image_path> <question>", file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    question = sys.argv[2]
    
    try:
        # Setup environment
        setup_cuda()
        
        # Load model (cache for subsequent calls)
        if not hasattr(main, 'model'):
            main.model, main.processor, main.device = load_vlm_model()
        
        # Process image
        result = process_image(main.model, main.processor, main.device, image_path, question)
        
        # Output result (stdout will be captured by ROS2 node)
        print(result)
        
    except Exception as e:
        print(f"VLM processing failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
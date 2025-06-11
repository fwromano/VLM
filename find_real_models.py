#!/usr/bin/env python3
"""
Find Real Available VLM Models on HuggingFace
"""

import requests
from transformers import pipeline
import torch

def check_model_exists(model_name):
    """Check if a model exists on HuggingFace"""
    try:
        response = requests.get(f"https://huggingface.co/api/models/{model_name}")
        return response.status_code == 200
    except:
        return False

def test_model_loading(model_name, task="image-to-text"):
    """Test if we can actually load a model"""
    try:
        print(f"Testing {model_name}...")
        model = pipeline(task, model=model_name, device=-1)  # CPU test
        print(f"✅ {model_name} - WORKS")
        return True
    except Exception as e:
        print(f"❌ {model_name} - FAILED: {str(e)[:100]}")
        return False

print("🔍 Finding Real Available VLM Models")
print("=" * 50)

# Potential InternVL models to check
intern_models = [
    "OpenGVLab/InternVL3-2B",
    "OpenGVLab/InternVL3-8B", 
    "OpenGVLab/InternVL3-26B",
    "OpenGVLab/InternVL3-72B",
    "OpenGVLab/InternVL2-2B",
    "OpenGVLab/InternVL2-8B",
    "OpenGVLab/InternVL2-26B",
    "lmms-lab/InternVL2-2B",
    "lmms-lab/InternVL2-8B",
    "OpenGVLab/InternVL-Chat-V1-5",
    "OpenGVLab/InternVL-Chat-V1-2-Plus"
]

# BLIP models (known to work)
blip_models = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large", 
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-opt-6.7b",
    "Salesforce/blip2-opt-2.7b-coco",
    "Salesforce/blip2-flan-t5-xl"
]

# Other VLM models
other_vlm_models = [
    "microsoft/git-base",
    "microsoft/git-large",
    "microsoft/git-large-coco",
    "nlpconnect/vit-gpt2-image-captioning",
    "Salesforce/instructblip-vicuna-7b",
    "Salesforce/instructblip-flan-t5-xl"
]

print("\n🔍 Checking InternVL Models:")
working_intern = []
for model in intern_models:
    if check_model_exists(model):
        print(f"✅ EXISTS: {model}")
        working_intern.append(model)
    else:
        print(f"❌ NOT FOUND: {model}")

print("\n🔍 Testing BLIP Models:")
working_blip = []
for model in blip_models:
    if test_model_loading(model):
        working_blip.append(model)

print("\n🔍 Testing Other VLM Models:")
working_other = []
for model in other_vlm_models:
    if test_model_loading(model):
        working_other.append(model)

print("\n" + "=" * 50)
print("📋 SUMMARY OF WORKING MODELS:")
print("=" * 50)

print("\n✅ Working InternVL Models:")
for model in working_intern:
    print(f"   {model}")

print("\n✅ Working BLIP Models:")
for model in working_blip:
    print(f"   {model}")

print("\n✅ Working Other VLM Models:")
for model in working_other:
    print(f"   {model}")

print(f"\nTotal working models found: {len(working_intern + working_blip + working_other)}")

# Test GPU availability
print(f"\n🖥️  GPU Info:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"   GPU Name: {torch.cuda.get_device_properties(0).name}")
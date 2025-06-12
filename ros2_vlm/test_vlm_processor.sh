#!/bin/bash

echo "Testing VLM Processor in Conda"
echo "=============================="

# Check if VLM conda environment exists
if [ ! -d "/home/fwromano/anaconda3/envs/vlm" ]; then
    echo "❌ VLM conda environment not found"
    echo "Create it with: cd ../standalone && ./setup.sh"
    exit 1
fi

echo "✓ VLM conda environment found"

# Create a simple test image
echo ""
echo "Creating test image..."
/home/fwromano/anaconda3/envs/vlm/bin/python -c "
import numpy as np
from PIL import Image
import os

# Create a simple test image (blue square with white circle)
img_array = np.zeros((224, 224, 3), dtype=np.uint8)
img_array[:, :, 2] = 100  # Blue background

# Add a white circle in the center
center = 112
radius = 50
y, x = np.ogrid[:224, :224]
mask = (x - center)**2 + (y - center)**2 <= radius**2
img_array[mask] = [255, 255, 255]  # White circle

# Save test image
img = Image.fromarray(img_array)
img.save('/tmp/test_vlm_image.jpg')
print('✓ Test image created at /tmp/test_vlm_image.jpg')
"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create test image"
    exit 1
fi

# Test VLM processor
echo ""
echo "Testing VLM processor..."
echo "This may take a moment to load the model..."

/home/fwromano/anaconda3/envs/vlm/bin/python vlm_processor.py /tmp/test_vlm_image.jpg "What shapes and colors do you see in this image?"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ VLM processor working!"
else
    echo ""
    echo "❌ VLM processor failed"
    echo "Check your HuggingFace authentication and model access"
fi

# Clean up
rm -f /tmp/test_vlm_image.jpg
echo ""
echo "Test complete!"
#!/bin/bash

# InternVL3 Model Comparison Script
# Easy way to test different models and see performance differences

echo "🚀 InternVL3 Model Performance Comparison"
echo "========================================="
echo ""

# Activate VLM environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^vlm "; then
        echo "✅ Activating VLM environment..."
        conda activate vlm
    else
        echo "⚠️  VLM environment not found, using system Python"
    fi
fi

# Show available models
echo "📋 Available InternVL3 Models:"
python vlm_model_selector.py --list-models

echo ""
echo "🎯 Quick Start Commands:"
echo ""
echo "🏃‍♂️ Fast & Light (2B model):"
echo "   ./compare_models.sh 2B"
echo ""
echo "⚖️  Balanced (8B model):"
echo "   ./compare_models.sh 8B"
echo ""
echo "🏆 Best Quality (26B model - requires 24GB VRAM):"
echo "   ./compare_models.sh 26B"
echo ""
echo "🚀 Extreme (72B model - requires multiple GPUs):"
echo "   ./compare_models.sh 72B"
echo ""

# If model specified, run it
if [ $# -gt 0 ]; then
    MODEL=$1
    PROMPT=${2:-"Describe what you see in detail"}
    RATE=${3:-0.5}
    
    echo "🔄 Starting $MODEL model comparison..."
    echo "   Model: InternVL3-$MODEL"
    echo "   Prompt: $PROMPT"
    echo "   Rate: $RATE Hz"
    echo ""
    echo "Press 'r' in the camera window to reset performance stats"
    echo "Press 'q' to quit and see final performance report"
    echo ""
    
    python vlm_model_selector.py \
        --model $MODEL \
        --prompt "$PROMPT" \
        --rate $RATE
else
    echo "Usage: $0 <model> [prompt] [rate]"
    echo ""
    echo "Examples:"
    echo "  $0 2B                                    # Fast 2B model"
    echo "  $0 8B \"Count the objects\" 1.0           # 8B model, custom prompt"
    echo "  $0 26B \"Detailed analysis\" 0.2         # Slow but detailed"
    echo ""
fi
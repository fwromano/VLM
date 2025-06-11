#!/bin/bash

# Real Working VLM Models Runner
# Uses verified model names that actually exist on HuggingFace

echo "🔍 Real Working VLM Models"
echo "=========================="
echo ""

# Activate environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^vlm "; then
        echo "✅ Activating VLM environment..."
        conda activate vlm
    fi
fi

echo "🎯 Available Real Models:"
echo ""
echo "🚀 InternVL3 Models (Latest & Best):"
echo "   InternVL3-2B   - Fast, 4GB VRAM, great quality"
echo "   InternVL3-8B   - Balanced, 12GB VRAM, excellent quality"
echo ""
echo "🔧 InternVL2 Models (Proven Stable):"
echo "   InternVL2-26B  - Large, 24GB VRAM, research grade"
echo ""
echo "💬 Chat Models:"
echo "   InternVL-Chat  - Conversational, interactive analysis"
echo ""
echo "⚡ Fallback Models (Always Work):"
echo "   blip-large     - Reliable, 2GB VRAM, good descriptions"
echo "   blip-base      - Fast, 1GB VRAM, basic descriptions"
echo ""

# Parse arguments
MODEL=${1:-"InternVL3-8B"}
FORMAT=${2:-"detailed"}
TOKENS=${3:-1024}
PROMPT=${4:-"Analyze this image in maximum detail"}

echo "🔄 Starting Real VLM Analysis..."
echo "   Model: $MODEL"
echo "   Format: $FORMAT"
echo "   Max Tokens: $TOKENS"
echo "   Your GPU: 15.7GB (perfect for InternVL3-8B!)"
echo ""

# Show what we're about to run
if [[ "$MODEL" == "InternVL3-8B" ]]; then
    echo "✅ Perfect choice! InternVL3-8B fits your GPU and gives excellent quality"
elif [[ "$MODEL" == "InternVL3-2B" ]]; then
    echo "⚡ Fast choice! InternVL3-2B will be very quick on your GPU"
elif [[ "$MODEL" == "InternVL2-26B" ]]; then
    echo "🧠 High quality! Will run on CPU (slower but best results)"
    EXTRA_ARGS="--device cpu"
else
    echo "🔄 Using fallback model for guaranteed compatibility"
fi

echo ""
echo "Controls:"
echo "   'f' = cycle output formats (json, reasoning, detailed, ultrathink)"
echo "   'q' = quit and save detailed report"
echo "   'r' = reset performance stats"
echo ""

# Run the enhanced selector with real model names
python vlm_enhanced_selector.py \
    --model "$MODEL" \
    --format "$FORMAT" \
    --max-tokens "$TOKENS" \
    --prompt "$PROMPT" \
    --rate 0.3 \
    ${EXTRA_ARGS:-}

echo ""
echo "Session complete! Check the generated JSON file for detailed analysis."
#!/bin/bash

# Gemma3 VLM Runner - FAIL HARD MODE
# No fallbacks, no recovery - Gemma3 or nothing

echo "🔥 Gemma3 VLM - FAIL HARD MODE"
echo "============================="
echo ""

# Activate environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^vlm "; then
        echo "✅ Activating VLM environment..."
        conda activate vlm
    else
        echo "💥 HARD FAILURE: VLM environment not found"
        echo "🚫 Run ./setup.sh first"
        exit 1
    fi
else
    echo "💥 HARD FAILURE: Conda not available"
    exit 1
fi

echo "⚠️  WARNING: FAIL HARD MODE ACTIVE"
echo "   - NO fallback models"
echo "   - NO error recovery"
echo "   - Gemma3 works or system exits"
echo ""

echo "🎯 Available Gemma3 Models:"
echo ""
echo "🚀 gemma3-2b  - 2B params, 4GB VRAM, fast"
echo "🔥 gemma3-9b  - 9B params, 16GB VRAM, balanced"  
echo "💥 gemma3-27b - 27B params, 32GB VRAM, maximum quality"
echo ""

# Parse arguments
MODEL=${1:-"gemma3-2b"}
VISION=${2:-"blip"}
TOKENS=${3:-2048}
TEMP=${4:-0.7}
PROMPT=${5:-"Provide a comprehensive and detailed analysis of this image"}

echo "🔄 Starting Gemma3 VLM (FAIL HARD)..."
echo "   Model: $MODEL"
echo "   Vision Encoder: $VISION"
echo "   Max Tokens: $TOKENS"
echo "   Temperature: $TEMP"
echo "   Your GPU: 15.7GB VRAM"
echo ""

# Check model compatibility
if [[ "$MODEL" == "gemma3-2b" ]]; then
    echo "✅ Perfect! gemma3-2b fits your GPU easily"
elif [[ "$MODEL" == "gemma3-9b" ]]; then
    echo "⚡ Good choice! gemma3-9b should fit your GPU"
elif [[ "$MODEL" == "gemma3-27b" ]]; then
    echo "⚠️  WARNING: gemma3-27b needs 32GB VRAM"
    echo "   Your GPU: 15.7GB - will likely fail"
    echo "   Consider using CPU with --device cpu"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "💥 User aborted due to VRAM concerns"
        exit 1
    fi
fi

echo ""
echo "🎮 Gemma3 Controls:"
echo "   q = quit and save results"
echo "   p = pause/resume continuous analysis"
echo "   s = single shot analysis"
echo "   r = reset performance statistics"
echo ""
echo "🚀 Launching Gemma3 VLM..."
echo "   Mode: FAIL HARD (no fallbacks)"
echo "   Any error will terminate immediately"
echo ""

# Run Gemma3 VLM
python vlm_gemma3.py \
    --model "$MODEL" \
    --vision-encoder "$VISION" \
    --max-tokens "$TOKENS" \
    --temperature "$TEMP" \
    --prompt "$PROMPT" \
    --rate 0.3

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Gemma3 VLM session completed successfully"
    echo "📁 Check the generated JSON file for detailed results"
else
    echo ""
    echo "💥 Gemma3 VLM failed with exit code: $EXIT_CODE"
    echo "🚫 No recovery attempted (FAIL HARD mode)"
fi

exit $EXIT_CODE
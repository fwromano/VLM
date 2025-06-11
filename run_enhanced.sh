#!/bin/bash

# Enhanced VLM Runner with Long Responses and Structured Output

echo "🧠 Enhanced VLM with Long Responses & Structured Output"
echo "=================================================="
echo ""

# Activate environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^vlm "; then
        echo "✅ Activating VLM environment..."
        conda activate vlm
    fi
fi

# Show available options
echo "🎯 Quick Commands:"
echo ""
echo "🔍 Standard Analysis (longer responses):"
echo "   ./run_enhanced.sh standard"
echo ""
echo "📋 JSON Structured Output:"
echo "   ./run_enhanced.sh json"
echo ""
echo "🧠 Step-by-Step Reasoning:"
echo "   ./run_enhanced.sh reasoning"
echo ""
echo "📖 Ultra-Detailed Analysis:"
echo "   ./run_enhanced.sh detailed"
echo ""
echo "🚀 Ultra-Think Mode (maximum detail):"
echo "   ./run_enhanced.sh ultrathink"
echo ""

# Show all models
if [[ "$1" == "models" ]]; then
    python vlm_enhanced_selector.py --list-models
    exit 0
fi

# Run with specific format
if [ $# -gt 0 ]; then
    FORMAT=$1
    MODEL=${2:-"8B"}
    TOKENS=${3:-1024}
    PROMPT=${4:-"Analyze this image in maximum detail"}
    
    echo "🔄 Starting Enhanced VLM..."
    echo "   Format: $FORMAT"
    echo "   Model: $MODEL"
    echo "   Max Tokens: $TOKENS"
    echo "   Prompt: $PROMPT"
    echo ""
    echo "Enhanced Controls:"
    echo "   'f' = cycle through output formats"
    echo "   'q' = quit and save detailed report"
    echo ""
    
    python vlm_enhanced_selector.py \
        --model $MODEL \
        --format $FORMAT \
        --max-tokens $TOKENS \
        --prompt "$PROMPT" \
        --rate 0.3
else
    echo "Usage: $0 <format> [model] [max_tokens] [prompt]"
    echo ""
    echo "Formats: standard, json, reasoning, detailed, ultrathink"
    echo "Models: 2B, 8B, 26B, 72B"
    echo ""
    echo "Examples:"
    echo "  $0 json 8B 2048                                    # JSON output"
    echo "  $0 ultrathink 8B 4096 \"Analyze safety hazards\"    # Ultra detailed"
    echo "  $0 reasoning 2B 1024 \"What story does this tell?\" # Step-by-step"
    echo ""
fi
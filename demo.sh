#!/bin/bash
# demo.sh
# Canonical demo for fine-tuning and running inference with Gemma 4 using Unsloth

set -e

echo "=========================================================="
echo " Starting Gemma 4 Fine-Tuning Demo (LaTeX OCR Vision Task)"
echo "=========================================================="

echo "[1/4] Setting up environment with uv..."
# Ensure uv is available in path
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create a virtual environment using uv
uv venv
source .venv/bin/activate

# Install dependencies lightning-fast with uv
uv pip install unsloth trl peft transformers datasets bitsandbytes accelerate
echo "Dependencies installed."

echo "[2/4] Running fine-tuning pipeline..."
# Disable hf_transfer and xet to prevent silent download hangs or partial corruptions
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
uv pip uninstall -y hf-transfer hf-xet

# We run a quick training of 30 steps on the canonical dataset
python finetune_gemma4.py \
    --model_name "unsloth/gemma-4-E4B-unsloth-bnb-4bit" \
    --dataset_name "unsloth/LaTeX_OCR" \
    --modality "vision" \
    --max_steps 30 \
    --output_dir "gemma-4-latex-demo"
echo "Fine-tuning complete. Checkpoints are saved in 'gemma-4-latex-demo'."

echo "[3/4] Running inference demo on a sample..."
# This will load the adapter and generate LaTeX for a sample image from the dataset
python inference_demo.py --adapter_dir "gemma-4-latex-demo"
echo "Inference demo complete."

echo "=========================================================="
echo " Demo completed successfully!"
echo "=========================================================="

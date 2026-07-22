# Gemma 4 Fine-Tuning Pipeline with Unsloth

This repository contains a generalized, scalable pipeline for fine-tuning Google's **Gemma 4** models (text and vision) using the [Unsloth](https://unsloth.ai) framework. Unsloth is chosen as the backend due to its ~1.5x faster training speeds and ~60% reduction in VRAM, enabling fine-tuning of advanced models locally on standard consumer GPUs.

## Overview

The `finetune_gemma4.py` script serves as a complete demonstration and boilerplate for fine-tuning Gemma 4 using `SFTTrainer` and PEFT (Parameter-Efficient Fine-Tuning) via LoRA (Low-Rank Adaptation). It supports:
- **Vision Fine-Tuning** (e.g., Gemma 4 E2B, E4B)
- **Text Fine-Tuning**
- **4-bit Quantization** via BitsAndBytes out of the box
- **GGUF Export** for optimal local inference (e.g., using Ollama or llama.cpp)

## Prerequisites

To use this pipeline, it is recommended to use [`uv`](https://github.com/astral-sh/uv) for fast, isolated dependency management. Since Unsloth interacts closely with your specific CUDA and torch versions, please refer to the official [Unsloth Installation Guide](https://github.com/unslothai/unsloth) for complex setups, or simply install the general dependencies using uv:

```bash
uv venv
source .venv/bin/activate
uv pip install unsloth trl peft transformers datasets bitsandbytes
```

## How to Use the Pipeline

### Running the Canonical Demo
We provide a 1-click execution script that acts as a canonical demonstration. It installs the required dependencies, fine-tunes a Gemma 4 Vision model (`unsloth/gemma-4-e4b-bnb-4bit`) on a LaTeX OCR task for a quick 30 steps, and runs an inference test displaying the model's generation stream alongside the ground truth.

```bash
./demo.sh
```

### Manual Usage

You can run the script directly with default parameters (which will run a quick vision fine-tuning demo on a LaTeX OCR dataset):

```bash
python finetune_gemma4.py
```

### CLI Arguments

The script is highly customizable through command-line arguments to adapt to your specific dataset and environment:

- `--model_name`: The path or Hugging Face repo of the Gemma 4 model (Default: `unsloth/gemma-4-e4b-bnb-4bit`).
- `--dataset_name`: Hugging Face dataset to fine-tune on (Default: `unsloth/LaTeX_OCR`).
- `--modality`: Select the modality to fine-tune. Options are `text` or `vision` (Default: `vision`).
- `--max_seq_length`: Maximum sequence length (Default: `2048`).
- `--batch_size`: Per-device training batch size (Default: `1`).
- `--grad_accum`: Gradient accumulation steps (Default: `4`).
- `--max_steps`: Total training steps. Set to `0` to use `--epochs` instead (Default: `60`).
- `--epochs`: Number of training epochs, active only if `--max_steps 0`.
- `--learning_rate`: Training learning rate (Default: `2e-4`).
- `--save_gguf`: Pass this flag to automatically quantize and export the fine-tuned model into `.gguf` format for inference.

### Example: Text Fine-Tuning

To fine-tune a text-only Gemma 4 model for 1 epoch and save to GGUF:

```bash
python finetune_gemma4.py --model_name "unsloth/gemma-4-12b-bnb-4bit" \
                          --dataset_name "your_text_dataset" \
                          --modality "text" \
                          --max_steps 0 \
                          --epochs 1 \
                          --save_gguf
```

## Preparing Custom Datasets

The script includes a `prepare_dataset` function. Unsloth's `SFTTrainer` setup expects datasets in a structured multi-turn conversation format. 

### Text-Only Format Example
For standard text-only fine-tuning, you simply pass `type: "text"` entries.
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is the capital of France?"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "The capital of France is Paris."}
      ]
    }
  ]
}
```

### Vision + Text Format Example
For image-to-text fine-tuning, pass a PIL Image object or a path to the image alongside the text. This natively supports interleaving multiple images and text!
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "<PIL.Image object 1>"},
        {"type": "text", "text": "What is the total of this receipt?"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "The total is $19.99."}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Now look at this second receipt:"},
        {"type": "image", "image": "<PIL.Image object 2>"},
        {"type": "text", "text": "What is the merchant name?"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "The merchant is Example Store."}
      ]
    }
  ]
}
```

### Video + Text Format Example
For video processing or fine-tuning, pass a video file path or a sequence of extracted frames along with your text prompt. Gemma 4 processes video temporally alongside text instructions.
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "<Video object or path>"},
        {"type": "text", "text": "Describe the main action occurring in this video clip."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "A person is teaching a dog how to fetch a frisbee in a park."}
      ]
    }
  ]
}
```

### Audio Format Example
Gemma 4 also supports audio transcription or reasoning.
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "audio", "audio": "<Audio object or path>"},
        {"type": "text", "text": "Transcribe the following speech segment in English."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Hello everyone and welcome back."}
      ]
    }
  ]
}
```

If you are using a custom dataset, update the mapping inside the `prepare_dataset` function in `finetune_gemma4.py` to align your raw dataset features to these expected output schemas.

## References & Resources

- [Unsloth Gemma 4 Documentation](https://unsloth.ai/docs/models/gemma-4)
- [Gemma 4 Fine-Tuning Guide](https://unsloth.ai/docs/models/gemma-4/train)

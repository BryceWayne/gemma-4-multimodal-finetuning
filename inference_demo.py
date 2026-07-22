import argparse
from datasets import load_dataset
from unsloth import FastVisionModel, get_chat_template

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Demo for Fine-Tuned Gemma 4")
    parser.add_argument("--base_model", type=str, default="unsloth/gemma-4-E4B-unsloth-bnb-4bit", help="Base model name")
    parser.add_argument("--adapter_dir", type=str, default="gemma-4-latex-demo", help="Directory where the fine-tuned adapter is saved")
    parser.add_argument("--dataset_name", type=str, default="unsloth/LaTeX_OCR", help="Dataset to fetch a test sample from")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading base model '{args.base_model}' and adapter from '{args.adapter_dir}'...")
    # Load model and apply PEFT adapter natively through Unsloth
    model, processor = FastVisionModel.from_pretrained(
        model_name = args.base_model,
        load_in_4bit = True,
    )
    
    # Load the fine-tuned adapter
    model.load_adapter(args.adapter_dir)
    FastVisionModel.for_inference(model) # Enable native 2x faster inference
    
    processor = get_chat_template(processor, "gemma-4-thinking")

    print(f"Loading a test sample from '{args.dataset_name}'...")
    dataset = load_dataset(args.dataset_name, split="train")
    # Grab the 3rd sample from the dataset as a test
    test_sample = dataset[2]
    image = test_sample["image"]
    true_latex = test_sample["text"]
    
    instruction = "Write the LaTeX representation for this image."
    
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    
    print("\nPreparing prompt...")
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to("cuda")

    print("\n--- Generating Response ---")
    from transformers import TextStreamer
    text_streamer = TextStreamer(processor, skip_prompt=True)
    
    # Generate the prediction
    _ = model.generate(
        **inputs, 
        streamer=text_streamer, 
        max_new_tokens=128,
        use_cache=True, 
        temperature=0.1, # Low temperature for more deterministic output
    )
    
    print("\n--- Expected Output (Ground Truth) ---")
    print(true_latex)
    
if __name__ == "__main__":
    main()

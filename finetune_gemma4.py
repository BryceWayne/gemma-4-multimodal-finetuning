import os
import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel, FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="General Pipeline for Fine-Tuning Gemma 4 with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-4-e4b-bnb-4bit", help="Model name or path")
    parser.add_argument("--dataset_name", type=str, default="unsloth/LaTeX_OCR", help="Hugging Face dataset name")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--modality", type=str, choices=["text", "vision"], default="vision", help="Fine-tuning modality")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for model checkpoints")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=60, help="Max training steps (set to 0 to use epochs)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (if max_steps is 0)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save_gguf", action="store_true", help="Whether to save the model in GGUF format after training")
    return parser.parse_args()

def prepare_dataset(dataset, modality, instruction=""):
    """
    Converts dataset to the standard conversation format expected by Unsloth & SFTTrainer.
    Customize this function based on your specific dataset's structure.
    """
    def convert_vision_to_conversation(sample):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": sample["image"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
            ]
        }
        
    def convert_text_to_conversation(sample):
        return {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": sample.get("instruction", "")}]},
                {"role": "assistant", "content": [{"type": "text", "text": sample.get("output", "")}]},
            ]
        }

    if modality == "vision":
        return [convert_vision_to_conversation(sample) for sample in dataset]
    else:
        return [convert_text_to_conversation(sample) for sample in dataset]

def main():
    args = parse_args()
    
    print(f"Loading {args.modality} model: {args.model_name}")
    if args.modality == "vision":
        model, processor = FastVisionModel.from_pretrained(
            model_name = args.model_name,
            load_in_4bit = True,
            max_seq_length = args.max_seq_length,
        )
        
        # Apply PEFT
        model = FastVisionModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )
        
        processor = get_chat_template(processor, "gemma-4-thinking")
        tokenizer = processor.tokenizer
        data_collator = UnslothVisionDataCollator(model, processor)
        
    else: # text
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_name,
            max_seq_length = args.max_seq_length,
            load_in_4bit = True,
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )
        
        tokenizer = get_chat_template(tokenizer, "gemma-4-thinking")
        data_collator = None # SFTTrainer uses default for text

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    
    # Example instruction for vision tasks
    instruction = "Write the LaTeX representation for this image." if args.modality == "vision" else ""
    converted_dataset = prepare_dataset(dataset, args.modality, instruction)

    # Configure Training arguments
    training_args = SFTConfig(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        warmup_ratio = 0.03,
        max_grad_norm = 0.3,
        learning_rate = args.learning_rate,
        logging_steps = 1,
        save_strategy = "steps",
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = args.output_dir,
        report_to = "none",
        
        # Unsloth specific SFT configurations
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = args.max_seq_length,
    )

    if args.max_steps > 0:
        training_args.max_steps = args.max_steps
    else:
        training_args.num_train_epochs = args.epochs

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model = model,
        train_dataset = converted_dataset,
        processing_class = tokenizer,
        data_collator = data_collator,
        args = training_args
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print("Training complete!")

    print(f"Saving final model adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_gguf:
        print("Exporting model to GGUF format...")
        # Save to typical GGUF formats (Q4_K_M is recommended for local inference)
        if args.modality == "vision":
            model.save_pretrained_gguf(args.output_dir, processor, quantization_method = "q4_k_m")
        else:
            model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method = "q4_k_m")
        print("GGUF export complete!")

if __name__ == "__main__":
    main()

# Autonomous Fine-Tuning Agent Instructions

Welcome to the `gemma-4-multimodal-finetuning` autoresearch setup!

Your goal is to autonomously discover the optimal fine-tuning hyperparameters and architectures for Gemma 4 (4B) on a single GPU.

## How it works

1. **`finetune_gemma4.py`**: This is the core script you will modify. It contains the model loading, Unsloth configuration, and the `SFTTrainer`.
2. **`demo.sh`**: Use this script to execute a training run. It guarantees environment setup and executes `finetune_gemma4.py`.
3. **Time Budget**: To ensure fair comparison between architectures and hyperparameter choices, the training loop is strictly bounded to exactly 5 minutes (300 seconds) of wall-clock time. 

## Your Task

1. Read `finetune_gemma4.py`.
2. Propose a single, focused change. You may experiment with:
   - **LoRA Configuration**: Rank (`r`), `lora_alpha`, `lora_dropout`, target modules.
   - **Hyperparameters**: `learning_rate`, `weight_decay`, `warmup_steps`, `lr_scheduler_type` (e.g. cosine, linear).
   - **Optimizer**: Try different optimizers supported by Hugging Face (e.g., `adamw_8bit`, `adamw_torch_fused`, `paged_adamw_32bit`).
   - **Batch Sizing**: `per_device_train_batch_size`, `gradient_accumulation_steps`.
3. Apply the modification to `finetune_gemma4.py`.
4. Execute `./demo.sh`.
5. Observe the final training loss at the end of the 5-minute budget.
6. If the loss improved compared to the baseline (or previous run), keep the change and document it in an `experiments.log` file. If it degraded, revert it.
7. Repeat this process indefinitely.

## Important Constraints
- **Never modify `demo.sh`** unless absolutely necessary for environment stability.
- **Never modify the 5-minute TimeLimitCallback** in `finetune_gemma4.py`. The time budget must remain strictly fixed so experiments are comparable.
- Always output the exact final loss value in your experiment log.

Let the autonomous research begin!

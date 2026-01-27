#!/usr/bin/env python3
"""
AstroLlama Fine-tuning Script
Fine-tune Llama 3.1 70B for astronomy tasks using QLoRA.

Usage:
    # Local fine-tuning (requires 2x A100 or similar)
    python scripts/fine_tune.py --local
    
    # Fine-tune and upload to Together.ai
    python scripts/fine_tune.py --upload-together
    
    # Fine-tune using Together.ai's service
    python scripts/fine_tune.py --together-finetune
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "base_model": "meta-llama/Llama-3.1-70B-Instruct",
    "output_dir": "./outputs/astro-llama-70b",
    "train_file": "./data/training/combined_train.jsonl",
    "val_file": "./data/training/combined_val.jsonl",
    
    # LoRA config
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Training config
    "num_epochs": 3,
    "batch_size": 1,  # Small for 70B
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "max_seq_length": 4096,
    "logging_steps": 10,
    "save_steps": 100,
    
    # Quantization
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data(train_file: str, val_file: str = None, tokenizer=None):
    """Load and prepare training data from JSONL files."""
    
    print(f"Loading training data from: {train_file}")
    
    def load_jsonl(filepath):
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file) if val_file and os.path.exists(val_file) else None
    
    # Convert to chat format for Llama 3.1
    def format_chat(example):
        messages = example.get('messages', [])
        
        # Format as Llama 3.1 chat template
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == 'user':
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == 'assistant':
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        return {"text": formatted}
    
    train_dataset = Dataset.from_list([format_chat(ex) for ex in train_data])
    val_dataset = Dataset.from_list([format_chat(ex) for ex in val_data]) if val_data else None
    
    print(f"Training examples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def prepare_combined_dataset(data_dir: str = "./data/training"):
    """Combine all training data files into a single JSONL."""
    
    all_data = []
    data_path = Path(data_dir)
    
    for jsonl_file in data_path.glob("*.jsonl"):
        if "combined" in jsonl_file.name:
            continue
        
        print(f"Loading: {jsonl_file}")
        with open(jsonl_file, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))
    
    # Shuffle
    import random
    random.shuffle(all_data)
    
    # Split 90/10
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Save
    train_file = data_path / "combined_train.jsonl"
    val_file = data_path / "combined_val.jsonl"
    
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(val_file, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Combined dataset created:")
    print(f"  Train: {len(train_data)} examples -> {train_file}")
    print(f"  Val: {len(val_data)} examples -> {val_file}")
    
    return str(train_file), str(val_file)


# =============================================================================
# LOCAL FINE-TUNING (QLoRA)
# =============================================================================

def fine_tune_local(config: dict):
    """Fine-tune locally using QLoRA."""
    
    print("\n" + "="*60)
    print("AstroLlama Local Fine-tuning (QLoRA)")
    print("="*60)
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Local fine-tuning requires GPU.")
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, config["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
    )
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print(f"Loading model: {config['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    train_dataset, val_dataset = load_and_prepare_data(
        config["train_file"],
        config.get("val_file"),
        tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",  # or "wandb" if you have it
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=False,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save
    final_path = f"{config['output_dir']}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\nTraining complete! Model saved to: {final_path}")
    return final_path


# =============================================================================
# TOGETHER.AI FINE-TUNING
# =============================================================================

def fine_tune_together(config: dict):
    """Submit fine-tuning job to Together.ai."""
    
    try:
        import together
    except ImportError:
        print("Please install: pip install together")
        return
    
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("Error: TOGETHER_API_KEY not set")
        return
    
    together.api_key = api_key
    
    print("\n" + "="*60)
    print("Together.ai Fine-tuning")
    print("="*60)
    
    # Upload training file
    print(f"\nUploading training file: {config['train_file']}")
    
    file_response = together.Files.upload(file=config["train_file"])
    file_id = file_response["id"]
    print(f"File uploaded: {file_id}")
    
    # Create fine-tuning job
    print("\nCreating fine-tuning job...")
    
    job_response = together.FineTuning.create(
        training_file=file_id,
        model=config["base_model"],
        n_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        suffix="astro-llama",
    )
    
    job_id = job_response["id"]
    print(f"Fine-tuning job created: {job_id}")
    print(f"Model will be: {job_response.get('output_name', 'TBD')}")
    
    # Check status
    print("\nTo check status:")
    print(f"  together fine-tuning retrieve {job_id}")
    
    return job_id


# =============================================================================
# MERGE AND UPLOAD
# =============================================================================

def merge_and_upload(adapter_path: str, base_model: str, output_name: str):
    """Merge LoRA adapter with base model and upload."""
    
    from peft import PeftModel
    
    print("\n" + "="*60)
    print("Merging LoRA adapter with base model")
    print("="*60)
    
    # Load base model
    print(f"\nLoading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load adapter
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge
    print("Merging weights...")
    model = model.merge_and_unload()
    
    # Save merged model
    merged_path = f"./outputs/{output_name}"
    print(f"Saving merged model to: {merged_path}")
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    
    # Upload to HuggingFace Hub (optional)
    upload = input("\nUpload to HuggingFace Hub? (y/n): ").strip().lower()
    if upload == 'y':
        hf_repo = input("Enter repo name (username/model-name): ").strip()
        print(f"Uploading to: {hf_repo}")
        model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)
        print("Upload complete!")
    
    return merged_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AstroLlama Fine-tuning")
    
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--local", action="store_true", help="Fine-tune locally with QLoRA")
    parser.add_argument("--together-finetune", action="store_true", help="Fine-tune using Together.ai")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare combined training dataset")
    parser.add_argument("--merge", type=str, help="Path to LoRA adapter to merge")
    
    # Override config options
    parser.add_argument("--model", type=str, help="Base model to fine-tune")
    parser.add_argument("--train-file", type=str, help="Training data file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f).get('finetuning', DEFAULT_CONFIG)
    else:
        config = DEFAULT_CONFIG.copy()
    
    # Override with CLI args
    if args.model:
        config["base_model"] = args.model
    if args.train_file:
        config["train_file"] = args.train_file
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Execute requested action
    if args.prepare_data:
        prepare_combined_dataset()
    
    elif args.local:
        fine_tune_local(config)
    
    elif args.together_finetune:
        fine_tune_together(config)
    
    elif args.merge:
        merge_and_upload(
            args.merge,
            config["base_model"],
            "astro-llama-70b-merged"
        )
    
    else:
        parser.print_help()
        print("\n\nQuick start:")
        print("  1. Prepare data:     python scripts/fine_tune.py --prepare-data")
        print("  2. Local training:   python scripts/fine_tune.py --local")
        print("  3. Or use Together:  python scripts/fine_tune.py --together-finetune")


if __name__ == "__main__":
    main()

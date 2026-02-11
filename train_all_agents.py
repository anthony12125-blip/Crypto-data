#!/usr/bin/env python3
"""
Train all 3 crypto agents simultaneously on RunPod using Hugging Face Transformers + PEFT (LoRA)
Saves models to /workspace/models/

Uses:
- PyTorch for GPU training on NVIDIA
- Transformers for model loading
- PEFT (LoRA) for efficient fine-tuning
- BitsAndBytes for 4-bit quantization
"""

import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import bitsandbytes as bnb

# Config
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TRAINING_DIR = "data/training"
OUTPUT_DIR = "/workspace/models"

def setup_model_and_tokenizer():
    """Load model with 4-bit quantization for memory efficiency"""
    print(f"Loading {BASE_MODEL} with 4-bit quantization...")
    
    # Check for GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires NVIDIA GPU.")
        sys.exit(1)
    
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 4-bit quantization config
    bnb_config = bnb.transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print(f"✅ Model loaded successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    
    return model, tokenizer

def setup_lora_model(model):
    """Setup LoRA configuration"""
    print("Setting up LoRA configuration...")
    
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling parameter
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def load_training_data(data_path, tokenizer):
    """Load and tokenize training data"""
    print(f"Loading training data from {data_path}...")
    
    dataset = load_dataset(
        "json",
        data_files={"train": f"{data_path}/train.jsonl"},
        split="train"
    )
    
    print(f"   Loaded {len(dataset)} training examples")
    
    def tokenize_function(examples):
        # Tokenize with padding and truncation
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_overflowing_tokens=False,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    return tokenized_dataset

def train_agent(agent_name, model, tokenizer):
    """Train a single agent"""
    print(f"\n{'='*60}")
    print(f"Training Agent {agent_name.upper()}")
    print(f"{'='*60}\n")
    
    data_path = f"{TRAINING_DIR}/agent_{agent_name}"
    output_path = f"{OUTPUT_DIR}/crypto_agent_{agent_name}"
    
    if not os.path.exists(f"{data_path}/train.jsonl"):
        print(f"❌ No training data found at {data_path}")
        return False
    
    try:
        # Setup LoRA
        model = setup_lora_model(model)
        
        # Load training data
        train_dataset = load_training_data(data_path, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            bf16=True,  # Use bfloat16 for training
            optim="paged_adamw_8bit",  # 8-bit Adam optimizer
            weight_decay=0.01,
            max_grad_norm=0.3,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="none",
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Trainer
        from transformers import Trainer
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print(f"Starting training for Agent {agent_name.upper()}...")
        trainer.train()
        
        # Save model
        print(f"Saving model to {output_path}...")
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        
        print(f"✅ Agent {agent_name} training complete!")
        print(f"   Model saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error training agent {agent_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Starting Crypto LLM Training - All 3 Agents")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model and tokenizer once
    model, tokenizer = setup_model_and_tokenizer()
    
    # Train agents sequentially (GPU memory constraint)
    agents = ["a", "b", "c"]
    results = {}
    
    for agent in agents:
        # Reload base model for each agent to avoid LoRA stacking
        if agent != agents[0]:
            print(f"\nReloading base model for Agent {agent.upper()}...")
            # Clear CUDA cache
            torch.cuda.empty_cache()
            model, tokenizer = setup_model_and_tokenizer()
        
        results[agent] = train_agent(agent, model, tokenizer)
        
        # Clear memory after training
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    for agent, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"Agent {agent.upper()}: {status}")
    
    # List output files
    print(f"\n📁 Models saved in: {OUTPUT_DIR}")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith('.safetensors') or file.endswith('.bin'):
                path = os.path.join(root, file)
                size = os.path.getsize(path) / (1024*1024)
                print(f"   {path}: {size:.1f} MB")
    
    print("\n✨ Training complete! Models ready for download.")

if __name__ == "__main__":
    main()

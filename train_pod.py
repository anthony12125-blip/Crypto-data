#!/usr/bin/env python3
"""
RunPod Training Script - Single Agent Training
Called by deploy_runpod_final.py for each agent pod
"""

import os
import sys
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import bitsandbytes as bnb

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--agent', required=True, choices=['a', 'b', 'c'])
parser.add_argument('--data-path', default='/workspace/train.jsonl')
parser.add_argument('--output-dir', default='/workspace/models')
args = parser.parse_args()

AGENT_NAMES = {'a': 'Technician', 'b': 'Fundamentalist', 'c': 'RiskManager'}
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def main():
    agent_name = AGENT_NAMES[args.agent]
    print(f"🚀 Training Agent {args.agent.upper()} - {agent_name}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model with 4-bit quantization
    print(f"\nLoading {BASE_MODEL}...")
    
    bnb_config = bnb.transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load training data
    print(f"\nLoading training data from {args.data_path}...")
    
    if not os.path.exists(args.data_path):
        print(f"❌ Training data not found: {args.data_path}")
        sys.exit(1)
    
    dataset = load_dataset("json", data_files={"train": args.data_path}, split="train")
    print(f"   Loaded {len(dataset)} examples")
    
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Training arguments
    output_path = f"{args.output_dir}/crypto_agent_{args.agent}"
    os.makedirs(output_path, exist_ok=True)
    
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
        bf16=True,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting training for Agent {args.agent.upper()} - {agent_name}")
    print(f"{'='*60}\n")
    
    trainer.train()
    
    # Save
    print(f"\nSaving model to {output_path}...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✅ Agent {args.agent.upper()} training complete!")
    print(f"   Model saved to: {output_path}")
    
    # Upload to GCS if credentials available
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        print("\nUploading to GCS...")
        os.system(f"gcloud storage cp -r {output_path} gs://haley_chat/models/")
        print("✓ Upload complete")

if __name__ == "__main__":
    main()

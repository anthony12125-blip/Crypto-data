#!/usr/bin/env python3
"""
Local Mac Mini Training
Train on your Mac instead of RunPod
"""

import os
import subprocess
from pathlib import Path

REPO = "https://github.com/anthony12125-blip/Crypto-data"
AGENTS = ['agent_a', 'agent_b', 'agent_c']

def train_locally():
    print("=" * 70)
    print("LOCAL TRAINING ON MAC MINI")
    print("=" * 70)
    
    workspace = Path.home() / "iron_dragoon_training"
    workspace.mkdir(exist_ok=True)
    
    # Clone repo
    repo_dir = workspace / "repo"
    if not repo_dir.exists():
        print("\n📥 Downloading training data...")
        subprocess.run([
            "git", "clone", "--depth", "1", 
            "--filter=blob:none", "--sparse",
            REPO, str(repo_dir)
        ], check=True)
        subprocess.run(
            ["git", "sparse-checkout", "set", "data/training"],
            cwd=repo_dir, check=True
        )
    
    # Install deps
    print("\n📦 Installing dependencies...")
    subprocess.run([
        "pip3", "install", "-q",
        "torch", "transformers", "datasets", 
        "peft", "accelerate", "bitsandbytes"
    ], check=True)
    
    # Train each agent sequentially
    for agent in AGENTS:
        print(f"\n{'='*70}")
        print(f"🤖 Training {agent}")
        print(f"{'='*70}")
        
        train_file = repo_dir / "data" / "training" / agent / "train.jsonl"
        
        # Create training script
        script = f'''
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset

AGENT = "{agent}"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TRAIN_FILE = "{train_file}"
OUT_DIR = "{workspace}/models/{agent}"

print(f"Training {{AGENT}}")
if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    print("No GPU/MPS - training on CPU (slow)")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {{device}}")

ds = load_dataset("json", data_files={{"train": TRAIN_FILE}}, split="train")
print(f"Examples: {{len(ds)}}")

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token

def tokenize(ex):
    out = tok(ex["text"], truncation=True, max_length=512, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out

tok_ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    device_map="auto" if device != "mps" else None
)

if device == "mps":
    model = model.to("mps")

model = prepare_model_for_kbit_training(model) if device == "cuda" else model

lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora)

args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=1,  # Reduced for Mac
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=500,
    bf16=(device=="cuda"),
    fp16=(device=="mps"),
    optim="adamw_torch",
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=tok_ds, data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))
print("Starting training...")
trainer.train()
trainer.save_model(OUT_DIR)
print(f"Saved to {{OUT_DIR}}")
'''
        
        script_path = workspace / f"train_{agent}.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        # Run training
        result = subprocess.run(["python3", str(script_path)])
        if result.returncode != 0:
            print(f"❌ {agent} failed")
        else:
            print(f"✅ {agent} complete")
    
    print(f"\n{'='*70}")
    print("All training complete")
    print(f"Models in: {workspace}/models/")

if __name__ == "__main__":
    train_locally()

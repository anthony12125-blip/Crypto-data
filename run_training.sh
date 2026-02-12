#!/bin/bash
# IRON DRAGOON Training Script for RunPod
# Run this inside a RunPod pod after creation

set -e

AGENT_NAME="${AGENT_NAME:-agent_a}"
REPO="https://github.com/anthony12125-blip/Crypto-data"

echo "════════════════════════════════════════════════════════════"
echo "  IRON DRAGOON TRAINING - $AGENT_NAME"
echo "════════════════════════════════════════════════════════════"

cd /workspace

# Install git
echo ""
echo "📥 Step 1: Installing git..."
apt-get update -qq && apt-get install -y -qq git

# Download training data
echo ""
echo "📥 Step 2: Downloading training data..."
git clone --depth 1 --filter=blob:none --sparse $REPO /workspace/repo
cd /workspace/repo
git sparse-checkout set data/training
cd /workspace

case $AGENT_NAME in
  agent_a)
    cp /workspace/repo/data/training/agent_a/train.jsonl /workspace/train.jsonl
    ;;
  agent_b)
    cp /workspace/repo/data/training/agent_b/train.jsonl /workspace/train.jsonl
    ;;
  agent_c)
    cp /workspace/repo/data/training/agent_c/train.jsonl /workspace/train.jsonl
    ;;
  *)
    echo "❌ Unknown agent: $AGENT_NAME"
    exit 1
    ;;
esac

echo "✓ Training data ready: $(wc -l < /workspace/train.jsonl) examples"

# Install dependencies
echo ""
echo "📦 Step 3: Installing dependencies..."
pip install -q transformers==4.36.0 datasets peft accelerate bitsandbytes scipy huggingface-hub
echo "✓ Dependencies installed"

# Create training script
echo ""
echo "📝 Step 4: Setting up training..."

cat > /workspace/train.py << 'EOF'
import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset

AGENT = os.getenv("AGENT_NAME", "agent")
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TRAIN_FILE = "/workspace/train.jsonl"
OUT_DIR = "/workspace/models/" + AGENT

def main():
    print("\n🚀 Starting training for " + AGENT)
    print("   Base Model: " + BASE_MODEL)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)
    
    print("\n🎮 GPU: " + torch.cuda.get_device_name(0))
    print("   VRAM: " + str(torch.cuda.get_device_properties(0).total_memory / 1e9) + " GB")
    
    # Load dataset
    print("\n📊 Loading training data...")
    ds = load_dataset("json", data_files={"train": TRAIN_FILE}, split="train")
    print("   Examples: " + str(len(ds)))
    
    # Tokenizer
    print("\n🔤 Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    
    def tokenize(ex):
        out = tok(ex["text"], truncation=True, max_length=2048, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out
    
    print("\n🔄 Tokenizing...")
    tok_ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    
    # Load model
    print("\n🧠 Loading model (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print params
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot_p = sum(p.numel() for p in model.parameters())
    print("\n📈 Trainable: " + str(train_p) + " / " + str(tot_p))
    
    # Training args
    args = TrainingArguments(
        output_dir=OUT_DIR,
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
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
    )
    
    # Train
    print("\n" + "="*60)
    print("🎯 TRAINING STARTED")
    print("="*60)
    
    trainer.train()
    
    # Save
    print("\n💾 Saving model...")
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print("\nModel saved to: " + OUT_DIR)

if __name__ == "__main__":
    main()
EOF

echo "✓ Training script ready"

# Run training
echo ""
echo "════════════════════════════════════════════════════════════"
echo "🚀 STARTING TRAINING"
echo "════════════════════════════════════════════════════════════"

export PYTHONUNBUFFERED=1
cd /workspace
python /workspace/train.py 2>&1 | tee /workspace/training.log

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🏁 TRAINING FINISHED"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Model: /workspace/models/$AGENT_NAME/"
echo "Logs: /workspace/training.log"
echo ""
echo "To download:"
echo "  runpodctl send /local/path /workspace/models/$AGENT_NAME/"

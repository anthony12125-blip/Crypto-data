"""
LLM Fine-Tuning Pipeline (MLX + LoRA)
========================================
Fine-tunes a 7B language model on crypto market data using Apple MLX.
Runs locally on your M4 Mac Mini OR on a rented GPU server.

Two modes:
  1. LOCAL (M4): Slower but free. QLoRA 4-bit. ~8-24 hours.
  2. CLOUD (rented GPU): Fast. LoRA 16-bit. ~2-5 hours. ~$5-30.

The model learns to:
  - Analyze crypto news and predict market sentiment
  - Understand market regime transitions
  - Identify when altcoins will amplify Bitcoin moves
  - Detect FUD vs genuine risk events
  - Read on-chain signals and whale movements

Run: python3 finetune_llm.py [--mode local|cloud]
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("finetune")

# Directories
TRAINING_DIR = Path("data/training")
MODEL_DIR = Path("models/llm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def check_dependencies():
    """Verify MLX and required packages are installed."""
    missing = []

    try:
        import mlx
        logger.info(f"✅ MLX version: {mlx.__version__}")
    except ImportError:
        missing.append("mlx")

    try:
        import mlx_lm
        logger.info(f"✅ MLX-LM available")
    except ImportError:
        missing.append("mlx-lm")

    if missing:
        logger.error(f"Missing packages: {missing}")
        logger.info("Install with: pip3 install mlx mlx-lm --break-system-packages")
        return False

    return True


def check_training_data():
    """Verify training data exists."""
    required = ["train.jsonl", "valid.jsonl"]
    for f in required:
        path = TRAINING_DIR / f
        if not path.exists():
            logger.error(f"Missing: {path}")
            logger.info("Run scrape_historical.py first to build training data!")
            return False

        # Count lines
        with open(path) as fh:
            count = sum(1 for _ in fh)
        logger.info(f"  {f}: {count} training examples")

    return True


def download_base_model(model_id: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"):
    """Download the base model from HuggingFace."""
    logger.info(f"Downloading base model: {model_id}")
    logger.info("This may take 5-10 minutes on first run (~4.5GB download)...")

    try:
        subprocess.run(
            ["huggingface-cli", "download", model_id],
            check=True,
            capture_output=False,
        )
        logger.info(f"✅ Model downloaded: {model_id}")
        return model_id
    except FileNotFoundError:
        logger.info("huggingface-cli not found. Installing...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"],
            check=True,
        )
        subprocess.run(
            ["huggingface-cli", "download", model_id],
            check=True,
        )
        return model_id
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def create_training_config(
    model_id: str,
    output_dir: str,
    num_iters: int = 1000,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    lora_rank: int = 8,
    lora_layers: int = 16,
):
    """Create MLX-LM training configuration."""
    config = {
        "model": model_id,
        "data": str(TRAINING_DIR),
        "train": True,
        "fine_tune_type": "lora",
        "adapter_path": output_dir,
        "iters": num_iters,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_layers": lora_layers,
        "save_every": 200,
        "val_batches": 25,
        "steps_per_eval": 100,
        "steps_per_report": 10,
        "max_seq_length": 2048,
        "grad_checkpoint": True,
    }

    config_path = Path(output_dir) / "training_config.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Training config saved to {config_path}")
    return config


def train_local_mlx(
    model_id: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    num_iters: int = 1000,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    lora_rank: int = 8,
):
    """
    Fine-tune using MLX LoRA locally on Apple Silicon.
    
    Estimated times on M4 Mac Mini (16GB):
      - 500 iters:  ~4-8 hours
      - 1000 iters: ~8-16 hours
      - 2000 iters: ~16-32 hours
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    adapter_dir = str(MODEL_DIR / f"crypto_adapter_{timestamp}")

    logger.info("=" * 60)
    logger.info("  LOCAL TRAINING (MLX + LoRA on Apple Silicon)")
    logger.info("=" * 60)
    logger.info(f"  Model:         {model_id}")
    logger.info(f"  Iterations:    {num_iters}")
    logger.info(f"  Batch size:    {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  LoRA rank:     {lora_rank}")
    logger.info(f"  Output:        {adapter_dir}")
    logger.info("=" * 60)

    # Build the MLX-LM command
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_id,
        "--data", str(TRAINING_DIR),
        "--train",
        "--fine-tune-type", "lora",
        "--adapter-path", adapter_dir,
        "--batch-size", str(batch_size),
        "--iters", str(num_iters),
        "--learning-rate", str(learning_rate),
        "--lora-layers", "16",
        "--save-every", "200",
        "--val-batches", "25",
        "--steps-per-eval", "100",
        "--steps-per-report", "10",
        "--max-seq-length", "2048",
        "--grad-checkpoint",
    ]

    logger.info(f"\nRunning command:\n  {' '.join(cmd)}\n")
    logger.info("Training started. This will take several hours.")
    logger.info("You can monitor progress in this terminal.\n")

    try:
        process = subprocess.run(cmd, check=True)
        logger.info(f"\n✅ Training complete! Adapter saved to: {adapter_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return None
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted. Partial adapter may be saved.")
        return adapter_dir

    return adapter_dir


def fuse_adapter(
    base_model: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    adapter_path: str = None,
):
    """
    Fuse the LoRA adapter back into the base model.
    Creates a single standalone model file.
    """
    if not adapter_path:
        # Find most recent adapter
        adapters = sorted(MODEL_DIR.glob("crypto_adapter_*"))
        if not adapters:
            logger.error("No adapters found. Train first!")
            return None
        adapter_path = str(adapters[-1])

    fused_path = str(MODEL_DIR / "crypto_analyst_7b_fused")

    logger.info(f"Fusing adapter into base model...")
    logger.info(f"  Base:    {base_model}")
    logger.info(f"  Adapter: {adapter_path}")
    logger.info(f"  Output:  {fused_path}")

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", base_model,
        "--adapter-path", adapter_path,
        "--save-path", fused_path,
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"\n✅ Fused model saved to: {fused_path}")
        return fused_path
    except Exception as e:
        logger.error(f"Fusion failed: {e}")
        return None


def test_model(model_path: str = None):
    """Quick test of the fine-tuned model."""
    if not model_path:
        fused = MODEL_DIR / "crypto_analyst_7b_fused"
        adapters = sorted(MODEL_DIR.glob("crypto_adapter_*"))
        if fused.exists():
            model_path = str(fused)
        elif adapters:
            model_path = str(adapters[-1])
        else:
            logger.error("No model found!")
            return

    test_prompts = [
        "Bitcoin just dropped 15% in 24 hours and the Fear & Greed Index is at 12 (Extreme Fear). What should a trader do?",
        "Ethereum is up 45% this week while Bitcoin is up only 8%. BTC dominance is falling. What does this mean for altcoins?",
        "The SEC just announced a new regulatory framework for crypto exchanges. How will this affect the market?",
        "Bitcoin has been consolidating between $50,000 and $55,000 for 3 weeks. Volume is decreasing. What's likely next?",
        "A whale just moved 10,000 BTC from cold storage to Binance. What does this signal?",
    ]

    logger.info(f"\nTesting model: {model_path}\n")

    for prompt in test_prompts:
        cmd = [
            sys.executable, "-m", "mlx_lm.generate",
            "--model", model_path,
            "--prompt", f"[INST] {prompt} [/INST]",
            "--max-tokens", "200",
            "--temp", "0.7",
        ]

        logger.info(f"Q: {prompt}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"A: {result.stdout.strip()}\n")
        except Exception as e:
            logger.error(f"Generation failed: {e}")

        logger.info("-" * 60)


def generate_cloud_training_package():
    """
    Generate a zip file with everything needed to train on a rented GPU.
    Upload this to RunPod, Lambda Labs, or Vast.ai.
    """
    import shutil

    package_dir = Path("cloud_training_package")
    package_dir.mkdir(exist_ok=True)

    # Copy training data
    if TRAINING_DIR.exists():
        shutil.copytree(TRAINING_DIR, package_dir / "data", dirs_exist_ok=True)

    # Create cloud training script
    cloud_script = '''#!/bin/bash
# ============================================================
# CLOUD GPU TRAINING SCRIPT
# ============================================================
# Upload this package to your rented GPU server and run:
#   chmod +x train_cloud.sh && ./train_cloud.sh
# ============================================================

set -e

echo "Installing dependencies..."
pip install mlx-lm transformers datasets peft bitsandbytes accelerate torch

echo "Downloading base model..."
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3

echo "Starting LoRA fine-tuning..."
python -m mlx_lm.lora \\
  --model mistralai/Mistral-7B-Instruct-v0.3 \\
  --data ./data \\
  --train \\
  --fine-tune-type lora \\
  --adapter-path ./crypto_adapter \\
  --batch-size 4 \\
  --iters 1000 \\
  --learning-rate 2e-5 \\
  --lora-layers 16 \\
  --save-every 200 \\
  --val-batches 25 \\
  --steps-per-eval 100 \\
  --steps-per-report 10 \\
  --max-seq-length 2048

echo "Fusing adapter..."
python -m mlx_lm.fuse \\
  --model mistralai/Mistral-7B-Instruct-v0.3 \\
  --adapter-path ./crypto_adapter \\
  --save-path ./crypto_analyst_7b_fused

echo "Done! Download the crypto_analyst_7b_fused folder to your M4."
'''

    with open(package_dir / "train_cloud.sh", "w") as f:
        f.write(cloud_script)

    # Create PyTorch/HuggingFace alternative for non-MLX GPU servers
    torch_script = '''#!/usr/bin/env python3
"""
Alternative training script using PyTorch + PEFT (for NVIDIA GPUs).
Use this if the cloud server doesn't have MLX (most won't).
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Config
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./crypto_adapter"
DATA_DIR = "./data"

print("Loading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset("json", data_files={
    "train": f"{DATA_DIR}/train.jsonl",
    "validation": f"{DATA_DIR}/valid.jsonl",
})

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=100,
    bf16=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=2048,
    dataset_text_field="text",
)

print("Starting training...")
trainer.train()

print("Saving adapter...")
trainer.save_model(OUTPUT_DIR)

print(f"Done! Download {OUTPUT_DIR} to your M4.")
print("Then fuse with: python -m mlx_lm.fuse --model <base> --adapter-path <adapter> --save-path <output>")
'''

    with open(package_dir / "train_torch.py", "w") as f:
        f.write(torch_script)

    # Zip it
    shutil.make_archive("cloud_training_package", "zip", ".", package_dir)
    logger.info(f"☁️ Cloud training package: cloud_training_package.zip")
    logger.info("Upload to RunPod/Lambda/Vast.ai and run train_cloud.sh or train_torch.py")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on crypto data")
    parser.add_argument("--mode", choices=["local", "cloud", "test", "fuse", "package"],
                        default="local", help="Training mode")
    parser.add_argument("--model", default="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                        help="Base model ID")
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--adapter", default=None, help="Adapter path (for fuse/test)")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║   🧠 LLM FINE-TUNING PIPELINE                                    ║
║   Teaching the model to understand crypto markets                ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    if args.mode == "local":
        if not check_dependencies():
            sys.exit(1)
        if not check_training_data():
            sys.exit(1)

        model_id = download_base_model(args.model)
        if not model_id:
            sys.exit(1)

        adapter_path = train_local_mlx(
            model_id=model_id,
            num_iters=args.iters,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

        if adapter_path:
            fuse_adapter(args.model, adapter_path)
            test_model()

    elif args.mode == "fuse":
        fuse_adapter(args.model, args.adapter)

    elif args.mode == "test":
        test_model(args.adapter)

    elif args.mode == "package":
        if not check_training_data():
            sys.exit(1)
        generate_cloud_training_package()

    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train all 3 crypto agents simultaneously on RunPod
Saves models to /workspace/models/
"""

import os
import sys
import multiprocessing
from mlx_lm import load, train

# Config
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TRAINING_DIR = "data/training"
OUTPUT_DIR = "/workspace/models"

def train_agent(agent_name):
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
        # Load base model
        print(f"Loading {BASE_MODEL}...")
        model, tokenizer = load(BASE_MODEL)
        
        # Train
        print(f"Starting training...")
        train(
            model=model,
            tokenizer=tokenizer,
            data=data_path,
            output_dir=output_path,
            batch_size=4,
            num_epochs=3,
            learning_rate=1e-4,
            lora_rank=16,
            lora_layers=16,
            max_seq_length=2048
        )
        
        print(f"✅ Agent {agent_name} training complete!")
        print(f"   Model saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error training agent {agent_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Crypto LLM Training - All 3 Agents")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train agents sequentially (GPU memory constraint)
    agents = ["a", "b", "c"]
    results = {}
    
    for agent in agents:
        results[agent] = train_agent(agent)
    
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

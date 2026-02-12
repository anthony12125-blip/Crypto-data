#!/usr/bin/env python3
"""
RunPod Training Deployment - 3 Agent Ensemble
Deploys pods that download training data and start training automatically
"""

import os
import json
import requests

# RunPod API configuration
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_API_URL = "https://api.runpod.io/graphql"

GITHUB_REPO = "https://github.com/anthony12125-blip/Crypto-data"

# Training configuration
TRAINING_CONFIG = {
    'base_model': 'mistralai/Mistral-7B-Instruct-v0.3',
    'gpu_type': 'NVIDIA RTX A6000',
    'gpu_count': 1,
    'container_image': 'runpod/pytorch:2.2.0-py3.10-cuda12.1-devel-ubuntu22.04',
    'volume_size_gb': 100,
}

# Agent configurations
AGENTS = {
    'agent_a': {
        'name': 'Technician',
        'training_file': 'data/training/agent_a/train.jsonl',
    },
    'agent_b': {
        'name': 'Fundamentalist',
        'training_file': 'data/training/agent_b/train.jsonl',
    },
    'agent_c': {
        'name': 'RiskManager',
        'training_file': 'data/training/agent_c/train.jsonl',
    }
}

def create_start_script(agent_key, agent_name, training_file):
    """Create the startup script that runs when pod starts"""
    
    script = f'''#!/bin/bash
set -e

echo "════════════════════════════════════════════════════════════"
echo "  IRON DRAGOON - {agent_name} ({agent_key})"
echo "════════════════════════════════════════════════════════════"

# Set working directory
cd /workspace

# Step 1: Clone repo and get training data
echo ""
echo "📥 Step 1: Downloading training data..."
apt-get update -qq && apt-get install -y -qq git curl

# Clone just the training data directory
echo "  Cloning from {GITHUB_REPO}..."
git clone --depth 1 --filter=blob:none --sparse {GITHUB_REPO} /workspace/repo
cd /workspace/repo
git sparse-checkout set data/training

# Copy training file to working location
mkdir -p /workspace/data
cp /workspace/repo/{training_file} /workspace/train.jsonl
echo "  ✓ Training data ready: $(wc -l < /workspace/train.jsonl) examples"

# Set agent name for training
export AGENT_NAME="{agent_key}"

# Step 2: Install dependencies
echo ""
echo "📦 Step 2: Installing dependencies..."
pip install -q transformers==4.36.0 datasets peft accelerate bitsandbytes scipy huggingface-hub
echo "  ✓ Dependencies installed"

# Step 3: Create training script
echo ""
echo "📝 Step 3: Creating training script..."

cat > /workspace/train.py << 'PYTHON_EOF'
import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset

# Config
AGENT_NAME = os.getenv('AGENT_NAME', 'agent')
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TRAINING_FILE = "/workspace/train.jsonl"
OUTPUT_DIR = "/workspace/models/" + AGENT_NAME

def main():
    print(f"\\n🚀 Starting training for " + AGENT_NAME)
    print(f"   Base Model: " + BASE_MODEL)
    print(f"   Training File: " + TRAINING_FILE)
    print(f"   Output: " + OUTPUT_DIR)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)
    
    print(f"\\n🎮 GPU: " + torch.cuda.get_device_name(0))
    print(f"   VRAM: " + str(torch.cuda.get_device_properties(0).total_memory / 1e9) + " GB")
    
    # Load dataset
    print("\\n📊 Loading training data...")
    dataset = load_dataset("json", data_files={"train": TRAINING_FILE}, split="train")
    print(f"   ✓ Loaded " + str(len(dataset)) + " examples")
    
    # Load tokenizer
    print("\\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Tokenize
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    print("\\n🔄 Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Load model with 4-bit quantization
    print("\\n🧠 Loading model (4-bit quantized)...")
    from transformers import BitsAndBytesConfig
    
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
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\\n📈 Trainable parameters: {{trainable_params:,}} / {{total_params:,}} ({{100 * trainable_params / total_params:.2f}}%)")
    
    # Training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
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
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Train
    print("\\n" + "="*60)
    print("🎯 TRAINING STARTED")
    print("="*60)
    
    trainer.train()
    
    # Save
    print("\\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"\\nModel saved to: {{OUTPUT_DIR}}")
    
    # List saved files
    import os
    if os.path.exists(OUTPUT_DIR):
        files = os.listdir(OUTPUT_DIR)
        print(f"\\nSaved files ({{len(files)}} total):")
        for f in files[:10]:
            print(f"  - {{f}}")

if __name__ == "__main__":
    main()
PYTHON_EOF

echo "  ✓ Training script created"

# Step 4: Run training
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  🚀 STARTING TRAINING"
echo "════════════════════════════════════════════════════════════"

cd /workspace
export AGENT_NAME="{agent_key}"
export PYTHONUNBUFFERED=1

python /workspace/train.py 2>&1 | tee /workspace/training.log

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  🏁 TRAINING FINISHED"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Model saved to: /workspace/models/{agent_key}/"
echo "Logs: /workspace/training.log"
echo ""
echo "To download model:"
echo "  runpodctl send <local-path> /workspace/models/{agent_key}/"
'''
    return script

def deploy_agents():
    """Deploy all 3 agents to RunPod"""
    
    print("=" * 70)
    print("IRON DRAGOON - RUNPOD DEPLOYMENT")
    print("=" * 70)
    print(f"\nRepository: {GITHUB_REPO}")
    print(f"Base Model: {TRAINING_CONFIG['base_model']}")
    print(f"GPU: {TRAINING_CONFIG['gpu_type']}")
    
    if not RUNPOD_API_KEY:
        print("\n❌ RUNPOD_API_KEY not set in environment")
        print("   export RUNPOD_API_KEY='your_key_here'")
        return {}
    
    deployed_pods = {}
    
    for agent_key, agent_config in AGENTS.items():
        print(f"\n{'='*70}")
        print(f"🤖 Deploying {agent_key.upper()}: {agent_config['name']}")
        print(f"{'='*70}")
        
        start_script = create_start_script(
            agent_key, 
            agent_config['name'],
            agent_config['training_file']
        )
        
        # GraphQL mutation
        query = '''
        mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                imageName
                env
                machineId
                machine {
                    podHostId
                }
            }
        }
        '''
        
        variables = {
            'input': {
                'cloudType': 'COMMUNITY',
                'gpuCount': TRAINING_CONFIG['gpu_count'],
                'volumeInGb': TRAINING_CONFIG['volume_size_gb'],
                'containerDiskInGb': 50,
                'minVcpuCount': 8,
                'minMemoryInGb': 32,
                'gpuTypeId': TRAINING_CONFIG['gpu_type'],
                'name': f'iron-dragoon-{agent_key}',
                'imageName': TRAINING_CONFIG['container_image'],
                'dockerArgs': '',
                'ports': '8888/http,22/tcp',
                'volumeMountPath': '/workspace',
                'env': json.dumps({'AGENT_NAME': agent_key, 'PYTHONUNBUFFERED': '1'}),
                'startScript': start_script,
            }
        }
        
        try:
            print(f"\n  Creating pod...")
            response = requests.post(
                RUNPOD_API_URL,
                headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'},
                json={'query': query, 'variables': variables},
                timeout=60
            )
            
            data = response.json()
            
            if 'errors' in data:
                print(f"  ❌ GraphQL Error: {data['errors']}")
                continue
            
            pod_data = data['data']['podFindAndDeployOnDemand']
            pod_id = pod_data['id']
            pod_name = pod_data['name']
            
            deployed_pods[agent_key] = {
                'id': pod_id,
                'name': pod_name,
                'status': 'CREATED'
            }
            
            print(f"  ✅ SUCCESS!")
            print(f"     Pod ID: {pod_id}")
            print(f"     Name: {pod_name}")
            print(f"     Status: Creating...")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save deployment info
    print(f"\n{'='*70}")
    print("DEPLOYMENT SUMMARY")
    print(f"{'='*70}")
    
    deployment_info = {
        'deployed_at': datetime.now().isoformat(),
        'github_repo': GITHUB_REPO,
        'pods': deployed_pods,
        'config': TRAINING_CONFIG,
    }
    
    with open('runpod_deployment.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"\n✅ Deployment complete!")
    print(f"   Saved to: runpod_deployment.json")
    
    if deployed_pods:
        print(f"\n🚀 Pods will auto-start training when ready")
        print(f"\nTo monitor training:")
        for agent_key, pod in deployed_pods.items():
            print(f"  runpodctl logs {pod['id']}")
        
        print(f"\nTo stop training:")
        for agent_key, pod in deployed_pods.items():
            print(f"  runpodctl stop pod {pod['id']}")
    
    return deployed_pods

if __name__ == "__main__":
    from datetime import datetime
    deploy_agents()

#!/usr/bin/env python3
"""
RunPod Training Setup - 3 Agent Ensemble
Deploys but doesn't start training
"""

import os
import json
import requests

# RunPod API configuration
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_API_URL = "https://api.runpod.io/graphql"

# Training configuration
TRAINING_CONFIG = {
    'base_model': 'mistralai/Mistral-7B-Instruct-v0.3',
    'gpu_type': 'NVIDIA RTX A6000',  # 48GB VRAM
    'gpu_count': 1,
    'container_image': 'runpod/pytorch:2.2.0-py3.10-cuda12.1-devel-ubuntu22.04',
    'volume_size_gb': 100,
}

# Agent configurations
AGENTS = {
    'agent_a': {
        'name': 'Technician',
        'description': 'Trained on Chart Patterns - Hindsight Supervised',
        'training_data': 'data/training/agent_a/train.jsonl',
        'lora_rank': 16,
        'learning_rate': 1e-4,
    },
    'agent_b': {
        'name': 'Fundamentalist', 
        'description': 'Scans News/X for Catalyst Keywords',
        'training_data': 'data/training/agent_b/train.jsonl',
        'lora_rank': 16,
        'learning_rate': 1e-4,
    },
    'agent_c': {
        'name': 'RiskManager',
        'description': 'Checks BTC Dominance & Funding Rates - Veto power',
        'training_data': 'data/training/agent_c/train.jsonl',
        'lora_rank': 16,
        'learning_rate': 1e-4,
    }
}

# Training script template
TRAINING_SCRIPT = '''#!/bin/bash
set -e

echo "🚀 Starting Iron Dragoon Training - Agent: $AGENT_NAME"

# Install dependencies
pip install -q transformers datasets peft accelerate bitsandbytes scipy

# Training script
cat > /workspace/train.py << 'PYTHON_EOF'
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
import bitsandbytes as bnb

# Config
AGENT_NAME = os.getenv('AGENT_NAME', 'agent')
BASE_MODEL = os.getenv('BASE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3')
TRAINING_FILE = os.getenv('TRAINING_FILE', 'train.jsonl')
OUTPUT_DIR = f"/workspace/models/{AGENT_NAME}"

def setup_model():
    print(f"Loading {BASE_MODEL}...")
    
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
    
    return model, tokenizer

def load_data(tokenizer):
    print(f"Loading {TRAINING_FILE}...")
    dataset = load_dataset("json", data_files={"train": TRAINING_FILE}, split="train")
    print(f"Loaded {len(dataset)} examples")
    
    def tokenize(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

def train():
    model, tokenizer = setup_model()
    train_dataset = load_data(tokenizer)
    
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
    )
    
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    print(f"Training {AGENT_NAME}...")
    trainer.train()
    
    print(f"Saving to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Training complete!")

if __name__ == "__main__":
    train()
PYTHON_EOF

# Run training
python /workspace/train.py

# Upload model to cloud storage (optional)
# gsutil -m cp -r /workspace/models/* gs://haley-models/iron-dragoon/

echo "🏁 Agent $AGENT_NAME training finished"
'''

def create_pod_config(agent_key, agent_config):
    """Create RunPod pod configuration for an agent"""
    
    env = {
        'AGENT_NAME': agent_key,
        'AGENT_DESCRIPTION': agent_config['description'],
        'BASE_MODEL': TRAINING_CONFIG['base_model'],
        'TRAINING_FILE': agent_config['training_data'],
        'LORA_RANK': str(agent_config['lora_rank']),
        'LEARNING_RATE': str(agent_config['learning_rate']),
        'PYTHONUNBUFFERED': '1',
    }
    
    pod_config = {
        'cloudType': 'COMMUNITY',
        'gpuCount': TRAINING_CONFIG['gpu_count'],
        'volumeInGb': TRAINING_CONFIG['volume_size_gb'],
        'containerDiskInGb': 50,
        'minVcpuCount': 8,
        'minMemoryInGb': 32,
        'gpuTypeId': 'NVIDIA RTX A6000',
        'name': f'iron-dragoon-{agent_key}',
        'imageName': TRAINING_CONFIG['container_image'],
        'dockerArgs': '',
        'ports': '8888/http,22/tcp',
        'volumeMountPath': '/workspace',
        'env': env,
        'startScript': TRAINING_SCRIPT,
    }
    
    return pod_config

def deploy_agents():
    """Deploy all 3 agents to RunPod (but don't start training)"""
    
    print("=" * 60)
    print("DEPLOYING IRON DRAGOON AGENTS TO RUNPOD")
    print("=" * 60)
    
    deployed_pods = {}
    
    for agent_key, agent_config in AGENTS.items():
        print(f"\n🤖 Deploying {agent_key} ({agent_config['name']})...")
        
        pod_config = create_pod_config(agent_key, agent_config)
        
        # GraphQL mutation to create pod
        query = '''
        mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
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
                'cloudType': pod_config['cloudType'],
                'gpuCount': pod_config['gpuCount'],
                'volumeInGb': pod_config['volumeInGb'],
                'containerDiskInGb': pod_config['containerDiskInGb'],
                'minVcpuCount': pod_config['minVcpuCount'],
                'minMemoryInGb': pod_config['minMemoryInGb'],
                'gpuTypeId': pod_config['gpuTypeId'],
                'name': pod_config['name'],
                'imageName': pod_config['imageName'],
                'dockerArgs': pod_config['dockerArgs'],
                'ports': pod_config['ports'],
                'volumeMountPath': pod_config['volumeMountPath'],
                'env': json.dumps(pod_config['env']),
                'startScript': pod_config['startScript'],
            }
        }
        
        try:
            response = requests.post(
                RUNPOD_API_URL,
                headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'},
                json={'query': query, 'variables': variables},
                timeout=30
            )
            
            data = response.json()
            
            if 'errors' in data:
                print(f"  ❌ Error: {data['errors']}")
                continue
            
            pod_id = data['data']['podFindAndDeployOnDemand']['id']
            deployed_pods[agent_key] = pod_id
            print(f"  ✅ Deployed! Pod ID: {pod_id}")
            print(f"  📝 Status: STOPPED (training not started)")
            
        except Exception as e:
            print(f"  ❌ Failed to deploy: {e}")
    
    # Save deployment info
    deployment_info = {
        'deployed_at': datetime.now().isoformat(),
        'pods': deployed_pods,
        'config': TRAINING_CONFIG,
        'agents': AGENTS,
    }
    
    with open('runpod_deployment.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"Saved deployment info to: runpod_deployment.json")
    print(f"\nTo start training, run:")
    for agent_key, pod_id in deployed_pods.items():
        print(f"  runpodctl start pod {pod_id}")
    
    return deployed_pods

if __name__ == "__main__":
    from datetime import datetime
    
    if not RUNPOD_API_KEY:
        print("❌ RUNPOD_API_KEY not set")
        exit(1)
    
    deploy_agents()

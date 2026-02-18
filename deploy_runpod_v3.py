#!/usr/bin/env python3
"""
RunPod Training Deployment - FIXED with correct image
3 Agent Ensemble
"""

import os
import json
import requests
from datetime import datetime

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_API_URL = "https://api.runpod.io/graphql"
GITHUB_REPO = "https://github.com/anthony12125-blip/Crypto-data"

# Use official PyTorch image (guaranteed to exist)
PYTORCH_IMAGE = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"

AGENTS = {
    'agent_a': {'name': 'Technician', 'file': 'data/training/agent_a/train.jsonl'},
    'agent_b': {'name': 'Fundamentalist', 'file': 'data/training/agent_b/train.jsonl'},
    'agent_c': {'name': 'RiskManager', 'file': 'data/training/agent_c/train.jsonl'},
}

def make_docker_command(agent_key, training_file):
    """Create docker command"""
    
    lines = [
        '#!/bin/bash',
        'set -e',
        'echo "IRON DRAGOON - ' + agent_key + '"',
        'cd /workspace',
        'apt-get update -qq && apt-get install -y -qq git 2>/dev/null || true',
        'echo "Downloading training data..."',
        'git clone --depth 1 --filter=blob:none --sparse ' + GITHUB_REPO + ' /workspace/repo 2>&1 | tail -5',
        'cd /workspace/repo && git sparse-checkout set data/training',
        'cp /workspace/repo/' + training_file + ' /workspace/train.jsonl',
        'echo "Training data: $(wc -l < /workspace/train.jsonl) examples"',
        'echo "Installing dependencies..."',
        'pip install -q transformers==4.36.0 datasets peft accelerate bitsandbytes scipy 2>&1 | tail -3',
        'echo "Starting training..."',
        'export AGENT_NAME="' + agent_key + '"',
        'export PYTHONUNBUFFERED=1',
        'python3 << \'PYEOF\'',
        'import os, sys, torch',
        'from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig',
        'from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType',
        'from datasets import load_dataset',
        'AGENT = os.getenv("AGENT_NAME", "agent")',
        'BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"',
        'TRAIN_FILE = "/workspace/train.jsonl"',
        'OUT_DIR = "/workspace/models/" + AGENT',
        'print("\\n🚀 Training " + AGENT)',
        'if not torch.cuda.is_available():',
        '    print("❌ No CUDA")',
        '    sys.exit(1)',
        'print("GPU: " + torch.cuda.get_device_name(0))',
        'ds = load_dataset("json", data_files={"train": TRAIN_FILE}, split="train")',
        'print("Examples: " + str(len(ds)))',
        'tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)',
        'tok.pad_token = tok.eos_token',
        'tok.padding_side = "right"',
        'def tokenize(ex):',
        '    out = tok(ex["text"], truncation=True, max_length=2048, padding="max_length")',
        '    out["labels"] = out["input_ids"].copy()',
        '    return out',
        'tok_ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)',
        'bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)',
        'model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)',
        'model = prepare_model_for_kbit_training(model)',
        'lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)',
        'model = get_peft_model(model, lora)',
        'train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)',
        'tot_p = sum(p.numel() for p in model.parameters())',
        'print("Trainable: " + str(train_p) + " / " + str(tot_p))',
        'args = TrainingArguments(output_dir=OUT_DIR, num_train_epochs=3, per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=1e-4, warmup_steps=100, logging_steps=10, save_steps=500, bf16=True, optim="paged_adamw_8bit", report_to="none")',
        'trainer = Trainer(model=model, args=args, train_dataset=tok_ds, data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))',
        'print("\\n🎯 TRAINING STARTED")',
        'trainer.train()',
        'print("\\n💾 Saving...")',
        'trainer.save_model(OUT_DIR)',
        'tok.save_pretrained(OUT_DIR)',
        'print("\\n✅ COMPLETE: " + OUT_DIR)',
        'PYEOF',
    ]
    
    return '\\n'.join(lines)

def deploy():
    """Deploy all 3 agents"""
    print("=" * 70)
    print("IRON DRAGOON - RUNPOD DEPLOYMENT (CORRECTED IMAGE)")
    print("=" * 70)
    print(f"\nImage: {PYTORCH_IMAGE}")
    
    if not RUNPOD_API_KEY:
        print("\n❌ Set RUNPOD_API_KEY")
        return
    
    pods = {}
    
    for key, cfg in AGENTS.items():
        print(f"\n{'='*70}")
        print(f"🤖 Deploying {key}: {cfg['name']}")
        print(f"{'='*70}")
        
        docker_cmd = make_docker_command(key, cfg['file'])
        
        query = '''
        mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                machineId
            }
        }
        '''
        
        vars = {
            'input': {
                'cloudType': 'COMMUNITY',
                'gpuCount': 1,
                'volumeInGb': 100,
                'containerDiskInGb': 50,
                'minVcpuCount': 8,
                'minMemoryInGb': 32,
                'gpuTypeId': 'NVIDIA RTX A6000',
                'name': f'iron-dragoon-{key}',
                'imageName': PYTORCH_IMAGE,
                'dockerArgs': docker_cmd,
                'ports': '8888/http,22/tcp',
                'volumeMountPath': '/workspace',
                'env': [
                    {'key': 'AGENT_NAME', 'value': key},
                    {'key': 'PYTHONUNBUFFERED', 'value': '1'}
                ],
            }
        }
        
        try:
            print(f"\n  Creating pod...")
            r = requests.post(
                RUNPOD_API_URL,
                headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'},
                json={'query': query, 'variables': vars},
                timeout=60
            )
            
            data = r.json()
            
            if 'errors' in data:
                print(f"  ❌ Error: {json.dumps(data['errors'], indent=2)}")
                continue
            
            pod = data['data']['podFindAndDeployOnDemand']
            pods[key] = {'id': pod['id'], 'name': pod['name']}
            
            print(f"  ✅ Created!")
            print(f"     ID: {pod['id']}")
            print(f"     Name: {pod['name']}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Save
    with open('runpod_deployed_v2.json', 'w') as f:
        json.dump({'deployed_at': datetime.now().isoformat(), 'pods': pods, 'image': PYTORCH_IMAGE}, f, indent=2)
    
    print(f"\n{'='*70}")
    print("DEPLOYMENT COMPLETE")
    print(f"{'='*70}")
    print(f"\n✅ {len(pods)} pods deployed with {PYTORCH_IMAGE}")
    
    if pods:
        print(f"\nMonitor with:")
        for key, pod in pods.items():
            print(f"  runpodctl logs {pod['id']}")
    
    return pods

if __name__ == "__main__":
    deploy()

#!/usr/bin/env python3
"""
RunPod Training Deployment - 3 Agent Ensemble
Fixed version with proper data download and training
"""

import os
import json
import requests
from datetime import datetime

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_API_URL = "https://api.runpod.io/graphql"
GITHUB_REPO = "https://github.com/anthony12125-blip/Crypto-data"

AGENTS = {
    'agent_a': {'name': 'Technician', 'file': 'data/training/agent_a/train.jsonl'},
    'agent_b': {'name': 'Fundamentalist', 'file': 'data/training/agent_b/train.jsonl'},
    'agent_c': {'name': 'RiskManager', 'file': 'data/training/agent_c/train.jsonl'},
}

def make_start_script(agent_key, agent_name, training_file):
    """Create startup script - using string template to avoid f-string issues"""
    
    lines = [
        '#!/bin/bash',
        'set -e',
        '',
        'echo "════════════════════════════════════════════════════════════"',
        f'echo "  IRON DRAGOON - {agent_name} ({agent_key})"',
        'echo "════════════════════════════════════════════════════════════"',
        '',
        'cd /workspace',
        '',
        '# Install git',
        'echo ""',
        'echo "📥 Downloading training data..."',
        'apt-get update -qq && apt-get install -y -qq git',
        '',
        f'git clone --depth 1 --filter=blob:none --sparse {GITHUB_REPO} /workspace/repo',
        'cd /workspace/repo',
        'git sparse-checkout set data/training',
        '',
        f'cp /workspace/repo/{training_file} /workspace/train.jsonl',
        'echo "✓ Training data ready"',
        '',
        '# Install Python deps',
        'echo ""',
        'echo "📦 Installing dependencies..."',
        'pip install -q transformers==4.36.0 datasets peft accelerate bitsandbytes scipy',
        'echo "✓ Dependencies installed"',
        '',
        '# Create training script',
        'cat > /workspace/train.py << \'PYEOF\'',
        'import os, sys, torch',
        'from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig',
        'from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType',
        'from datasets import load_dataset',
        '',
        'AGENT = os.getenv("AGENT_NAME", "agent")',
        'BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"',
        'TRAIN_FILE = "/workspace/train.jsonl"',
        'OUT_DIR = "/workspace/models/" + AGENT',
        '',
        'print("\\n🚀 Starting training for " + AGENT)',
        '',
        'if not torch.cuda.is_available():',
        '    print("❌ No CUDA")',
        '    sys.exit(1)',
        '',
        'print("GPU: " + torch.cuda.get_device_name(0))',
        '',
        '# Load data',
        'print("\\n📊 Loading data...")',
        'ds = load_dataset("json", data_files={"train": TRAIN_FILE}, split="train")',
        'print("Examples: " + str(len(ds)))',
        '',
        '# Tokenizer',
        'tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)',
        'tok.pad_token = tok.eos_token',
        'tok.padding_side = "right"',
        '',
        'def tokenize(ex):',
        '    out = tok(ex["text"], truncation=True, max_length=2048, padding="max_length")',
        '    out["labels"] = out["input_ids"].copy()',
        '    return out',
        '',
        'print("\\n🔄 Tokenizing...")',
        'tok_ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)',
        '',
        '# Model',
        'print("\\n🧠 Loading model...")',
        'bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)',
        'model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)',
        'model = prepare_model_for_kbit_training(model)',
        '',
        '# LoRA',
        'lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)',
        'model = get_peft_model(model, lora_config=lora)',
        '',
        'train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)',
        'tot_p = sum(p.numel() for p in model.parameters())',
        'print("Trainable: " + str(train_p) + " / " + str(tot_p))',
        '',
        '# Training',
        'args = TrainingArguments(',
        '    output_dir=OUT_DIR,',
        '    num_train_epochs=3,',
        '    per_device_train_batch_size=4,',
        '    gradient_accumulation_steps=4,',
        '    learning_rate=1e-4,',
        '    warmup_steps=100,',
        '    logging_steps=10,',
        '    save_steps=500,',
        '    bf16=True,',
        '    optim="paged_adamw_8bit",',
        '    report_to="none",',
        ')',
        '',
        'trainer = Trainer(model=model, args=args, train_dataset=tok_ds, data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))',
        '',
        'print("\\n" + "="*60)',
        'print("🎯 TRAINING STARTED")',
        'print("="*60)',
        '',
        'trainer.train()',
        '',
        'print("\\n💾 Saving...")',
        'trainer.save_model(OUT_DIR)',
        'tok.save_pretrained(OUT_DIR)',
        '',
        'print("\\n✅ TRAINING COMPLETE")',
        'print("Model: " + OUT_DIR)',
        'PYEOF',
        '',
        '# Run training',
        'echo ""',
        'echo "════════════════════════════════════════════════════════════"',
        'echo "🚀 STARTING TRAINING"',
        'echo "════════════════════════════════════════════════════════════"',
        '',
        f'export AGENT_NAME="{agent_key}"',
        'export PYTHONUNBUFFERED=1',
        '',
        'cd /workspace',
        'python /workspace/train.py 2>&1 | tee /workspace/training.log',
        '',
        'echo ""',
        'echo "════════════════════════════════════════════════════════════"',
        'echo "🏁 TRAINING FINISHED"',
        'echo "════════════════════════════════════════════════════════════"',
    ]
    
    return '\n'.join(lines)

def deploy():
    """Deploy all agents"""
    print("=" * 60)
    print("IRON DRAGOON - RUNPOD DEPLOYMENT")
    print("=" * 60)
    
    if not RUNPOD_API_KEY:
        print("\n❌ Set RUNPOD_API_KEY environment variable")
        return
    
    pods = {}
    
    for key, cfg in AGENTS.items():
        print(f"\n{'='*60}")
        print(f"🤖 Deploying {key}: {cfg['name']}")
        print(f"{'='*60}")
        
        script = make_start_script(key, cfg['name'], cfg['file'])
        
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
                'imageName': 'runpod/pytorch:2.2.0-py3.10-cuda12.1-devel-ubuntu22.04',
                'dockerArgs': '',
                'ports': '8888/http,22/tcp',
                'volumeMountPath': '/workspace',
                'env': json.dumps({'AGENT_NAME': key}),
                'startScript': script,
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
                print(f"  ❌ Error: {data['errors']}")
                continue
            
            pod = data['data']['podFindAndDeployOnDemand']
            pods[key] = {'id': pod['id'], 'name': pod['name']}
            
            print(f"  ✅ Created!")
            print(f"     ID: {pod['id']}")
            print(f"     Name: {pod['name']}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Save
    with open('runpod_deployment.json', 'w') as f:
        json.dump({'deployed_at': datetime.now().isoformat(), 'pods': pods}, f, indent=2)
    
    print(f"\n{'='*60}")
    print("DEPLOYMENT COMPLETE")
    print(f"{'='*60}")
    print(f"\n✅ {len(pods)} pods deployed")
    print(f"   Saved to: runpod_deployment.json")
    
    if pods:
        print(f"\n🚀 Pods will auto-start training")
        print(f"\nTo check logs:")
        for key, pod in pods.items():
            print(f"  runpodctl logs {pod['id']}")
    
    return pods

if __name__ == "__main__":
    deploy()

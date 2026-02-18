#!/usr/bin/env python3
"""
Deploy with KNOWN WORKING image
"""

import os
import json
import requests
from datetime import datetime

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_API_URL = "https://api.runpod.io/graphql"

# Official PyTorch image - guaranteed to exist
IMAGE = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"

AGENTS = [
    ('agent_a', 'Technician', 'data/training/agent_a/train.jsonl'),
    ('agent_b', 'Fundamentalist', 'data/training/agent_b/train.jsonl'),
    ('agent_c', 'RiskManager', 'data/training/agent_c/train.jsonl'),
]

def deploy():
    print("=" * 70)
    print(f"DEPLOYING WITH: {IMAGE}")
    print("=" * 70)
    
    if not RUNPOD_API_KEY:
        print("❌ Set RUNPOD_API_KEY")
        return
    
    pods = {}
    
    for key, name, train_file in AGENTS:
        print(f"\n🤖 {key}: {name}")
        
        # Simple command that will definitely work
        docker_cmd = f'/bin/bash -c "apt-get update && apt-get install -y git && pip install transformers==4.36.0 datasets peft accelerate bitsandbytes scipy && git clone --depth 1 --filter=blob:none --sparse https://github.com/anthony12125-blip/Crypto-data.git /workspace/repo && cd /workspace/repo && git sparse-checkout set data/training train_pod.py && cp {train_file} /workspace/train.jsonl && export AGENT_NAME={key} && export PYTHONUNBUFFERED=1 && python3 /workspace/repo/train_pod.py"'
        
        query = '''
        mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
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
                'imageName': IMAGE,
                'dockerArgs': docker_cmd,
                'ports': '8888/http,22/tcp',
                'volumeMountPath': '/workspace',
                'env': [{'key': 'AGENT_NAME', 'value': key}],
            }
        }
        
        try:
            r = requests.post(
                RUNPOD_API_URL,
                headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'},
                json={'query': query, 'variables': vars},
                timeout=60
            )
            
            data = r.json()
            
            if 'errors' in data:
                print(f"  ❌ {data['errors'][0]['message']}")
            else:
                pod = data['data']['podFindAndDeployOnDemand']
                pods[key] = {'id': pod['id'], 'name': pod['name']}
                print(f"  ✅ {pod['id']}")
                
        except Exception as e:
            print(f"  ❌ {e}")
    
    print(f"\n{'='*70}")
    print(f"Deployed: {len(pods)}/3")
    return pods

if __name__ == "__main__":
    deploy()

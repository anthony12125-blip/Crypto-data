#!/usr/bin/env python3
"""
Check ACTUAL status of deployed pods
"""

import os
import requests

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_API_URL = "https://api.runpod.io/graphql"

# Pod IDs from successful deployment
PODS = {
    'agent_a': 'dbj6fou7fjjix2',
    'agent_b': 'u9w634wsvtxbmd', 
    'agent_c': 'oxt1ekngidltqr'
}

def get_pod_status(pod_id):
    query = '''
    query Pod($id: String!) {
        pod(id: $id) {
            id
            name
            desiredStatus
            runtime {
                uptimeInSeconds
                gpus {
                    gpuUtilPercent
                    memoryUtilPercent
                }
            }
        }
    }
    '''
    
    try:
        r = requests.post(
            RUNPOD_API_URL,
            headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'},
            json={'query': query, 'variables': {'id': pod_id}},
            timeout=30
        )
        return r.json()
    except Exception as e:
        return {'error': str(e)}

print("=" * 60)
print("ACTUAL POD STATUS")
print("=" * 60)

for agent, pod_id in PODS.items():
    print(f"\n🤖 {agent.upper()}")
    print(f"   ID: {pod_id}")
    
    result = get_pod_status(pod_id)
    
    if 'errors' in result:
        print(f"   Error: {result['errors']}")
        continue
    
    pod = result.get('data', {}).get('pod')
    if not pod:
        print(f"   No data returned")
        continue
    
    print(f"   Name: {pod.get('name')}")
    print(f"   Status: {pod.get('desiredStatus')}")
    
    runtime = pod.get('runtime', {})
    if runtime:
        uptime = runtime.get('uptimeInSeconds', 0)
        print(f"   Uptime: {uptime//60} minutes")
        
        gpus = runtime.get('gpus', [])
        if gpus:
            gpu = gpus[0]
            print(f"   GPU Util: {gpu.get('gpuUtilPercent', 0)}%")
            print(f"   Memory: {gpu.get('memoryUtilPercent', 0)}%")
            
            if gpu.get('gpuUtilPercent', 0) > 50:
                print(f"   ✅ TRAINING ACTIVE")
            elif uptime > 600 and gpu.get('gpuUtilPercent', 0) < 10:
                print(f"   ⚠️  IDLE - Check logs")
            else:
                print(f"   ⏳ Starting up...")

print("\n" + "=" * 60)

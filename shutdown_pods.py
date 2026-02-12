#!/usr/bin/env python3
"""
Shutdown all RunPod training pods
Run this when training is complete to save credits
"""

import os
import requests

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_API_URL = "https://api.runpod.io/graphql"

PODS = {
    'agent_a': 'dbj6fou7fjjix2',
    'agent_b': 'u9w634wsvtxbmd',
    'agent_c': 'oxt1ekngidltqr'
}

def stop_pod(pod_id):
    """Stop a pod via API"""
    query = '''
    mutation PodStop($id: String!) {
        podStop(id: $id) {
            id
            desiredStatus
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
        data = r.json()
        if 'errors' in data:
            return f"Error: {data['errors']}"
        return f"Stopped - Status: {data['data']['podStop']['desiredStatus']}"
    except Exception as e:
        return f"Error: {e}"

print("=" * 60)
print("IRON DRAGOON - SHUTTING DOWN PODS")
print("=" * 60)

for agent, pod_id in PODS.items():
    print(f"\n🤖 {agent.upper()}")
    print(f"   Pod ID: {pod_id}")
    result = stop_pod(pod_id)
    print(f"   Result: {result}")

print("\n" + "=" * 60)
print("All pods stopped. Credits saved.")
print("=" * 60)

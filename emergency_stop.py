#!/usr/bin/env python3
"""
Emergency: Shutdown failing pods and re-deploy with correct image
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
    query = '''mutation PodStop($id: String!) { podStop(id: $id) { id desiredStatus } }'''
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

print("EMERGENCY: Stopping failing pods to save credits...")
for agent, pod_id in PODS.items():
    print(f"  Stopping {agent}...")
    result = stop_pod(pod_id)
    print(f"    Result: {result}")
print("\nPods stopped. Credits saved.")

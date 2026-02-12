#!/usr/bin/env python3
"""
RunPod Training Monitor
Checks pod status and logs to verify training is happening
"""

import os
import json
import requests
from datetime import datetime

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_API_URL = "https://api.runpod.io/graphql"

PODS = {
    'agent_a': 'dbj6fou7fjjix2',
    'agent_b': 'u9w634wsvtxbmd',
    'agent_c': 'oxt1ekngidltqr'
}

def get_pod_status(pod_id):
    """Get pod status via API"""
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
        data = r.json()
        return data.get('data', {}).get('pod')
    except Exception as e:
        return {'error': str(e)}

def get_pod_logs(pod_id):
    """Get recent logs"""
    query = '''
    query PodLogs($podId: String!, $tail: Int) {
        podLogs(podId: $podId, tail: $tail)
    }
    '''
    
    try:
        r = requests.post(
            RUNPOD_API_URL,
            headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'},
            json={'query': query, 'variables': {'podId': pod_id, 'tail': 30}},
            timeout=30
        )
        data = r.json()
        return data.get('data', {}).get('podLogs', '')
    except Exception as e:
        return f'Error: {e}'

def check_training_status():
    """Check all pods and verify training is happening"""
    print("=" * 70)
    print(f"IRON DRAGOON - TRAINING MONITOR - {datetime.now().isoformat()}")
    print("=" * 70)
    
    results = {}
    
    for agent, pod_id in PODS.items():
        print(f"\n🤖 {agent.upper()}")
        print(f"   Pod ID: {pod_id}")
        
        # Get status
        status = get_pod_status(pod_id)
        if status is None or (isinstance(status, dict) and 'error' in status):
            error_msg = status.get('error', 'Unknown error') if isinstance(status, dict) else 'No response'
            print(f"   ❌ Error: {error_msg}")
            continue
        
        desired = status.get('desiredStatus', 'unknown')
        runtime = status.get('runtime', {})
        uptime = runtime.get('uptimeInSeconds', 0)
        gpus = runtime.get('gpus', [])
        
        print(f"   Status: {desired}")
        print(f"   Uptime: {uptime//60} minutes")
        
        if gpus:
            gpu_util = gpus[0].get('gpuUtilPercent', 0)
            mem_util = gpus[0].get('memoryUtilPercent', 0)
            print(f"   GPU Util: {gpu_util}% | Memory: {mem_util}%")
            
            # Check if training is happening
            if gpu_util > 50:
                print(f"   ✅ TRAINING ACTIVE (GPU busy)")
                results[agent] = 'training'
            elif uptime > 300 and gpu_util < 10:
                print(f"   ⚠️  IDLE (may be downloading data)")
                results[agent] = 'idle'
            else:
                print(f"   ⏳ STARTING UP")
                results[agent] = 'starting'
        
        # Get logs
        logs = get_pod_logs(pod_id)
        if logs:
            # Check for key indicators
            if 'TRAINING STARTED' in logs:
                print(f"   ✅ Training script running")
            elif 'Downloading training data' in logs:
                print(f"   📥 Downloading data...")
            elif 'Error' in logs or 'error' in logs.lower():
                print(f"   ❌ ERRORS in logs")
            
            # Show last few lines
            last_lines = logs.strip().split('\n')[-5:]
            print(f"   Recent logs:")
            for line in last_lines:
                if line.strip():
                    print(f"      {line.strip()[:80]}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    training_count = sum(1 for v in results.values() if v == 'training')
    idle_count = sum(1 for v in results.values() if v == 'idle')
    starting_count = sum(1 for v in results.values() if v == 'starting')
    
    print(f"Training: {training_count} | Idle: {idle_count} | Starting: {starting_count}")
    
    # Check if all done
    if idle_count == 3 and all(results.get(a) == 'idle' for a in PODS.keys()):
        print(f"\n🎉 ALL AGENTS COMPLETE - Ready to shut down")
        return 'complete'
    
    return 'running'

def shutdown_pods():
    """Shutdown all pods"""
    print(f"\n{'='*70}")
    print("SHUTTING DOWN PODS")
    print(f"{'='*70}")
    
    query = '''
    mutation PodStop($id: String!) {
        podStop(id: $id) {
            id
            desiredStatus
        }
    }
    '''
    
    for agent, pod_id in PODS.items():
        try:
            r = requests.post(
                RUNPOD_API_URL,
                headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'},
                json={'query': query, 'variables': {'id': pod_id}},
                timeout=30
            )
            print(f"  {agent}: Stopped")
        except Exception as e:
            print(f"  {agent}: Error - {e}")

def save_status(status):
    """Save status to file"""
    with open('training_monitor_status.json', 'w') as f:
        json.dump({
            'checked_at': datetime.now().isoformat(),
            'status': status
        }, f)

if __name__ == "__main__":
    status = check_training_status()
    save_status(status)
    
    if status == 'complete':
        print("\n⚠️  Training appears complete. Run shutdown?")
        print("   python monitor_pods.py --shutdown")
    
    print(f"\nNext check: Run this script again in 30 minutes")
    print(f"Or set up: python monitor_pods.py --watch")

import runpod
import os
import subprocess

# RunPod handler
print("🚀 RunPod Serverless Handler Starting...")

# Run training
result = subprocess.run(
    ["python3", "train_all_agents.py"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("✅ Training job complete")

# Return results
job_done = {
    "status": "complete",
    "models": "/workspace/models/"
}

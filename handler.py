import runpod
import os
import subprocess
import sys

# RunPod serverless handler
def handler(job):
    """
    RunPod serverless handler for crypto agent training
    """
    job_input = job.get('input', {})
    
    print("🚀 RunPod Serverless Handler Starting...")
    print(f"Job ID: {job.get('id', 'unknown')}")
    
    # Step 1: Run data preparation
    print("\n📊 Step 1: Running data preparation...")
    result_prep = subprocess.run(
        ["python3", "data_prep.py"],
        capture_output=True,
        text=True
    )
    
    print(result_prep.stdout)
    if result_prep.stderr:
        print("Data prep STDERR:", result_prep.stderr)
    
    if result_prep.returncode != 0:
        return {
            "status": "error",
            "stage": "data_preparation",
            "message": "Data preparation failed",
            "stderr": result_prep.stderr
        }
    
    # Step 2: Run training
    print("\n🎓 Step 2: Running model training...")
    result_train = subprocess.run(
        ["python3", "train_all_agents.py"],
        capture_output=True,
        text=True
    )
    
    print(result_train.stdout)
    if result_train.stderr:
        print("Training STDERR:", result_train.stderr)
    
    # Check results
    models_dir = "/workspace/models"
    agent_models = []
    
    if os.path.exists(models_dir):
        for agent in ["a", "b", "c"]:
            agent_path = f"{models_dir}/crypto_agent_{agent}"
            if os.path.exists(agent_path):
                files = os.listdir(agent_path)
                safetensors = [f for f in files if f.endswith('.safetensors')]
                agent_models.append({
                    "agent": agent,
                    "path": agent_path,
                    "files": len(files),
                    "safetensors": len(safetensors)
                })
    
    return {
        "status": "complete" if result_train.returncode == 0 else "error",
        "training_exit_code": result_train.returncode,
        "models_directory": models_dir,
        "trained_models": agent_models,
        "stdout_preview": result_train.stdout[-2000:] if len(result_train.stdout) > 2000 else result_train.stdout
    }

# Start the serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

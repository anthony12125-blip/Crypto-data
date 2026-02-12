#!/bin/bash
# Manual RunPod deployment commands
# Copy and run these one by one, or use runpodctl

echo "IRON DRAGOON - Manual RunPod Deployment"
echo "=========================================="
echo ""
echo "Since the API has schema issues, here are the runpodctl commands:"
echo ""

for agent in agent_a agent_b agent_c; do
    name=$(echo $agent | tr '_' '-')
    echo "# Deploy $agent"
    echo "runpodctl create pod \\"
    echo "  --name iron-dragoon-$name \\"
    echo "  --gpuType 'NVIDIA RTX A6000' \\"
    echo "  --image runpod/pytorch:2.2.0-py3.10-cuda12.1-devel-ubuntu22.04 \\"
    echo "  --volumeSize 100 \\"
    echo "  --ports 8888:8888,22:22 \\"
    echo "  --env AGENT_NAME=$agent"
    echo ""
done

echo ""
echo "After pods are created, SSH into each and run:"
echo ""
echo "  git clone --depth 1 https://github.com/anthony12125-blip/Crypto-data /workspace/repo"
echo "  cp /workspace/repo/data/training/agent_a/train.jsonl /workspace/train.jsonl"
echo "  pip install transformers datasets peft accelerate bitsandbytes"
echo "  # Then run training script"
echo ""

# Crypto LLM Training - RunPod Serverless

This repo contains training data and scripts for fine-tuning 3 crypto analysis agents.

## Quick Deploy on RunPod

1. Go to https://console.runpod.io/serverless
2. Click "Deploy Serverless Endpoint"
3. Select this repo: `anthony12125-blip/Crypto-data`
4. Choose GPU: **RTX A4000** or **A40**
5. Click Deploy

Training starts automatically. Takes 6-12 hours for all 3 agents.

## Training Data

- **Agent A**: Category-chronological ordering (regulation → adoption → price action, etc.)
- **Agent B**: Reverse chronological (newest first)
- **Agent C**: Random shuffle

Each agent sees 590,940 training examples from 248K+ crypto articles (2021-2026).

## Output

Models saved to `/workspace/models/`:
- `crypto_agent_a/` - Category-focused analyst
- `crypto_agent_b/` - Recent-biased analyst  
- `crypto_agent_c/` - Pattern-focused analyst

## Manual Training

```bash
python3 train_all_agents.py
```

## Data Format

Training examples use Llama-style instruction format:
```json
{"text": "<s>[INST] Analyze crypto news from 2022-06-09: 'Supply chain solution Trackgood expands...' What is sentiment? [/INST] Sentiment: neutral. This is business adoption news... </s>"}
```

Categories: adoption, defi, exchange, macro, nft_metaverse, price_action, regulation, security, stablecoin, technology

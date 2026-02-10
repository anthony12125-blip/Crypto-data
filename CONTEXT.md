# PROJECT CONTEXT — Read This First

## Who This Is For
This project belongs to a trader who identified a key market pattern in 2017 and has been waiting for the right tools to exploit it at machine speed.

## The Core Thesis
**Bitcoin is the signal. Altcoins are the trade.**

During bull markets, when Bitcoin rallies (say 10% in a day), altcoins don't just follow — they AMPLIFY the move by 5x, 10x, even 100x on the SAME DAY. The media focuses on BTC, but the real money is in the amplification. No human can watch hundreds of coins simultaneously. This machine can.

## The Strategy
1. DO NOT trade Bitcoin. Bitcoin is only used as a momentum signal.
2. When BTC momentum is high, scan all altcoins for which ones are amplifying BTC's move the most (highest beta).
3. Enter positions in the top-amplifying alts.
4. Exit when BTC momentum fades — alts crash HARDER than BTC drops.

## System Architecture (Two Layers)

### Layer 1: Price Bot (always running, 24/7)
Pure math. Watches charts, calculates momentum, beta, volume. Makes fast trading decisions every 5 minutes.
- `btc_momentum.py` — Scores BTC momentum 0-100
- `alt_scanner.py` — Scans 200+ alts, ranks by beta amplification
- `regime_detector.py` — Bear/transition/bull/overheated classifier
- `live_trader.py` — Original trading orchestrator
- `data_collector.py` — 24/7 data pipeline into SQLite
- `backtest.py` — Walk-forward backtester
- `train_models.py` — Trains LightGBM on price data

### Layer 2: LLM Intelligence (spins up every 4 hours)
The real-world awareness layer. Three AI agents analyze news, sentiment, on-chain data and VOTE on what the bot should do.

**3-Agent Voting System:**
- Main Agent: Always monitoring, triggers votes when needed
- Agent A: 7B LLM fine-tuned on crypto data (conservative bias)
- Agent B: 7B LLM fine-tuned on crypto data (aggressive bias)
- All three vote: trade/hold/exit + aggressiveness level
- Majority wins. Decision valid for 4 hours.
- Emergency re-vote if BTC momentum crashes

**Files:**
- `agent_voting.py` — The 3-agent voting system and orchestrator
- `agent_trader.py` — Integrates agent decisions with the price bot
- `scrape_historical.py` — Collects years of crypto data for training
- `finetune_llm.py` — Fine-tunes Mistral 7B on crypto data using MLX LoRA

### How They Work Together
1. Price bot runs 24/7 watching charts
2. Every 4 hours, agents spin up and analyze news + sentiment
3. Agents vote: "trade aggressively" or "hold" or "exit everything"
4. Price bot adjusts its behavior based on the vote
5. If BTC momentum crashes, emergency re-vote triggers immediately

## Hardware
- Mac Mini M4 (16GB RAM)
- Mistral 7B (4-bit quantized) fits in ~6GB, leaving room for everything else
- MLX framework for native Apple Silicon inference
- Fine-tuning: either locally (8-24 hours) or rented GPU ($5-30, 2-5 hours)

## Current Status
- Bear market — building phase
- Data collector should be running 24/7
- Historical data needs to be scraped (scrape_historical.py)
- LLM needs to be fine-tuned on crypto data
- Paper trade on testnet before going live

## Getting Started
1. `chmod +x setup_m4.sh && ./setup_m4.sh`
2. `python3 data_collector.py` — START THIS NOW
3. `python3 scrape_historical.py` — Collect years of training data
4. `python3 finetune_llm.py --mode local` — Fine-tune the LLM
5. `python3 agent_trader.py` — Run the full system (testnet first!)

## Three Risk Profiles
- **Conservative**: Small positions, tight stops, high-confidence only
- **Moderate**: Balanced (default)
- **Aggressive**: Larger positions, wider stops, catches bigger moves

## Important Warnings
- Always start with testnet: true in config.yaml
- Risk limits exist (max daily loss, max drawdown) — don't disable them
- Alts crash HARDER than BTC — trailing stops are critical
- The 3-agent vote can be overridden by emergency BTC exit signals

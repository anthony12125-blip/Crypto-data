# PROJECT IRON DRAGOON

**Human-in-the-Loop AI-Assisted Crypto Trading System**

> *"Build it less like a Trader and more like a Hunter-Killer Drone."*

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HUMAN (You)                                       │
│                     Discord Command Interface                                │
│                      "Permission to Fire" System                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DISCORD BOT                                       │
│                    Human-in-the-Loop Controller                              │
│         • Receives alerts  • Shows status  • Gets approvals                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
┌───────────────────────────────┐         ┌───────────────────────────────┐
│      THE SENTRY (Mac Mini)    │         │    FREQTRADE (Docker)         │
│      24/7 Market Watchman     │         │    Execution Engine           │
│                               │         │                               │
│  • Scans top 50 volatile      │         │  • Handles exchange           │
│    coins every 5 minutes      │         │    connectivity               │
│  • Detects volume spikes      │◄───────►│  • Manages wallet             │
│    (>300% avg)                │         │  • Executes orders            │
│  • RSI extremes               │         │  • Tracks P&L                 │
│  • Volatility anomalies       │         │  • Limit orders only          │
│                               │         │    (no MEV sandwich)          │
│  Hard-coded Python math       │         │                               │
│  NO LLM hallucinations        │         │  Trailing Entry System        │
│                               │         │  • Buy Stop 1% above price    │
└───────────────────────────────┘         └───────────────────────────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      THE BRAIN (RunPod Cloud)                               │
│                     3-Agent Ensemble (Serverless)                            │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   AGENT A        │  │   AGENT B        │  │   AGENT C        │          │
│  │   Technician     │  │   Fundamentalist │  │   Risk Manager   │          │
│  │   (Mistral 7B)   │  │   (Mistral 7B)   │  │   (Mistral 7B)   │          │
│  │                  │  │                  │  │                  │          │
│  │ Chart patterns   │  │ News/X catalyst  │  │ BTC Dominance    │          │
│  │ Hindsight        │  │ keywords         │  │ Funding rates    │          │
│  │ supervised       │  │ Partnerships     │  │ Veto power       │          │
│  │                  │  │ Mainnet          │  │                  │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                     │                     │
│           └─────────────────────┼─────────────────────┘                     │
│                                 ▼                                           │
│                    ┌────────────────────────┐                               │
│                    │   ENSEMBLE CONSENSUS   │                               │
│                    │   2 of 3 to approve    │                               │
│                    └────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

1. **Human-in-the-Loop**: You are the kill switch. Discord approval required for all entries.
2. **Hindsight Training**: Models learn from "Perfect Past Trades" not price prediction
3. **Python Does Math**: All technical indicators hard-coded (RSI, MACD, Bollinger). AI gets summaries only.
4. **Trailing Entry**: Buy Stop orders 1% above current price - confirms momentum before risking capital
5. **Ensemble Intelligence**: 3 specialized agents vote. 2 of 3 required to proceed.

## System Requirements

- **Mac Mini** (M1/M2 or Intel) with Docker Desktop
- **Binance US** account (starting capital: $150)
- **Discord** server with bot token
- **RunPod** account (for training - can use free credits)

## Quick Start

### 1. Clone and Setup

```bash
cd Crypto-data
python setup.py
```

### 2. Configure Environment

Edit `.env` file:

```bash
# Discord (REQUIRED)
DISCORD_TOKEN=your_discord_bot_token
DISCORD_CHANNEL_ID=your_channel_id

# Binance US (for live trading)
BINANCE_US_API_KEY=your_api_key
BINANCE_US_SECRET=your_secret

# RunPod (for training)
RUNPOD_API_KEY=your_runpod_key
```

### 3. Harvest Historical Data

```bash
python iron_dragoon.py harvest
```

Downloads 12 years of OHLCV data for all target coins.

### 4. Label Training Data

```bash
python iron_dragoon.py label
```

Creates "hindsight labels" - marks trades that WOULD HAVE won.

### 5. Deploy Agents to RunPod

```bash
python deploy_runpod_agents.py
```

Deploys 3 training pods but **does not start training**.

### 6. Build and Start System

```bash
docker-compose build
docker-compose up -d
```

### 7. Start Training (when ready)

```bash
runpodctl start pod <pod_id>
```

## Discord Commands

| Command | Description |
|---------|-------------|
| `!status` | Show portfolio status, P&L, active positions |
| `!halt` | Emergency stop - pause all trading |
| `!resume` | Resume trading after halt |

## Signal Flow

```
1. Sentry detects Volume Spike / RSI Oversold / Volatility Anomaly
   ↓
2. Alert sent to Discord with: Price, RSI, Volume, Technical Summary
   ↓
3. You react: ✅ Approve  or  ❌ Reject
   ↓
4. If approved → Query 3-agent ensemble for consensus
   ↓
5. If 2 of 3 agents agree → Place Trailing Entry (Buy Stop)
   ↓
6. If price hits Buy Stop → Position opened
   ↓
7. Stop Loss and Take Profit automatically managed
```

## Trading Parameters

| Parameter | Value |
|-----------|-------|
| Starting Capital | $150 |
| Max Positions | 3 |
| Target Profit | 5% |
| Stop Loss | 2% |
| Trailing Entry | 1% above signal price |
| Max Drawdown | 10% (auto-halt) |
| Timeframe | 1h candles |

## Target Coins (Binance US)

- BTC/USD, ETH/USD, SOL/USD, ADA/USD
- DOT/USD, LINK/USD, MATIC/USD, AVAX/USD
- DOGE/USD, XRP/USD, LTC/USD, UNI/USD

## File Structure

```
Crypto-data/
├── iron_dragoon.py           # Main application
├── deploy_runpod_agents.py   # RunPod deployment
├── setup.py                  # System setup
├── docker-compose.yml        # Docker orchestration
├── Dockerfile.sentry         # Sentry container
├── Dockerfile.discord        # Discord bot container
├── freqtrade/
│   ├── config.json           # Freqtrade config
│   └── strategies/
│       └── IronDragoonStrategy.py
├── data/
│   └── irondragoons/
│       ├── raw/              # Historical price data
│       ├── labeled/          # Training data with labels
│       ├── training/         # Agent training splits
│       ├── sentry.db         # Sentry scan database
│       ├── signals.db        # Trade signals database
│       └── trades.db         # Performance tracking
├── models/                   # Trained model storage
└── logs/                     # Application logs
```

## Safety Features

- **Dry Run Mode**: Start with simulated trading (`dry_run: true` in config)
- **Emergency Halt**: `!halt` command stops all activity immediately
- **Drawdown Protection**: Auto-halt at 10% portfolio loss
- **Limit Orders Only**: No market orders (prevents MEV sandwich attacks)
- **Trailing Entry**: Confirms momentum before entry

## Monitoring

View logs:
```bash
docker-compose logs -f freqtrade
docker-compose logs -f sentry
docker-compose logs -f discord-bot
```

Access Freqtrade UI:
```
http://localhost:8080
Username: irondragoon
Password: (from .env)
```

## Training the Agents

Each agent trains independently on RunPod:

| Agent | Training Data | Focus |
|-------|---------------|-------|
| Technician | Category-chronological | Chart patterns, technical setups |
| Fundamentalist | Reverse-chronological | News catalysts, social sentiment |
| Risk Manager | Random shuffle | BTC dominance, funding rates, macro |

Training takes ~2-4 hours per agent on A6000 GPU.

## Performance Tracking

All trades logged to `data/irondragoons/trades.db`:

```sql
SELECT 
    pair,
    COUNT(*) as total_trades,
    AVG(pnl_pct) as avg_return,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses
FROM trades
GROUP BY pair;
```

## Troubleshooting

**Discord bot not responding:**
- Check `DISCORD_TOKEN` is set correctly
- Ensure bot has permissions in your server
- Check logs: `docker-compose logs discord-bot`

**No signals appearing:**
- Verify sentry is running: `docker-compose ps`
- Check Binance US API connectivity
- Review sentry logs: `docker-compose logs sentry`

**Freqtrade won't start:**
- Check config.json syntax
- Ensure strategy file exists
- Review logs: `docker-compose logs freqtrade`

## License

MIT License - Use at your own risk. Trading cryptocurrencies carries significant risk of loss.

## Disclaimer

This is an experimental trading system. Past performance does not guarantee future results. Always start with dry-run mode and never trade more than you can afford to lose. The authors are not responsible for any financial losses incurred through use of this system.

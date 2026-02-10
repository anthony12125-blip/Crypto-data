# 🚀 Crypto Alpha Bot — Alt Beta Amplification System

## The Thesis
Bitcoin is the signal. Altcoins are the trade.
When BTC rallies, alts amplify those moves 5-50x. This bot watches Bitcoin's momentum
and automatically finds & trades the altcoins that are amplifying the most — 24/7, across
hundreds of coins simultaneously. What a human can't do, the machine can.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  REGIME DETECTOR                     │
│  Bear Market → Accumulate & Collect Data             │
│  Bull Market → Activate Alt Trading Bots             │
│  Overheated  → Systematic Exit                       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              BTC MOMENTUM SCANNER                    │
│  Monitors BTC price action, volume, momentum         │
│  Outputs: momentum_score (0-100)                     │
│  "Is Bitcoin hot right now?"                         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│          ALT BETA AMPLIFICATION DETECTOR             │
│  Scans 200+ alts in real-time                        │
│  Calculates beta vs BTC on multiple timeframes       │
│  Ranks by: amplification ratio, volume surge,        │
│            relative strength vs BTC                  │
│  Outputs: ranked list of highest-beta alts           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              TRADING ENGINE                          │
│  Conservative / Moderate / Aggressive profiles       │
│  Entry: alt showing high beta + volume surge         │
│  Exit: trailing stop OR BTC momentum fading          │
│  Risk: position sizing based on profile              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              DASHBOARD (Streamlit)                    │
│  Live BTC momentum gauge                             │
│  Alt amplification heatmap                           │
│  Open positions & P/L                                │
│  Regime status indicator                             │
└─────────────────────────────────────────────────────┘
```

## Setup on Mac Mini M4

### Prerequisites
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.12+
brew install python@3.12

# Install project dependencies
pip3 install -r requirements.txt
```

### Configuration
1. Copy `config_example.yaml` to `config.yaml`
2. Add your exchange API keys (Binance recommended for most alt pairs)
3. Set your risk profile (conservative/moderate/aggressive)
4. Set your portfolio size

### Running
```bash
# Phase 1: Start data collection (run this NOW in bear market)
python3 data_collector.py

# Phase 2: Train models (once you have 30+ days of data)
python3 train_models.py

# Phase 3: Backtest strategies
python3 backtest.py

# Phase 4: Live trading (when bull market confirmed)
python3 live_trader.py

# Dashboard (anytime)
streamlit run dashboard.py
```

## Project Structure
```
crypto-alpha-bot/
├── config.yaml              # API keys, risk params, exchange settings
├── requirements.txt         # Python dependencies
├── data_collector.py        # Continuous data collection pipeline
├── btc_momentum.py          # Bitcoin momentum scoring engine
├── alt_scanner.py           # Alt beta amplification detector
├── regime_detector.py       # Bear/Bull/Overheated classifier
├── trading_engine.py        # Order execution & position management
├── backtest.py              # Walk-forward backtesting framework
├── train_models.py          # Model training pipeline
├── live_trader.py           # Main live trading orchestrator
├── dashboard.py             # Streamlit monitoring UI
├── data/                    # Collected market data
│   ├── ohlcv/               # Price candles
│   ├── orderbook/           # Order book snapshots
│   ├── sentiment/           # Fear & greed, social signals
│   └── onchain/             # On-chain metrics
├── models/                  # Trained model artifacts
└── logs/                    # Trading logs & performance
```

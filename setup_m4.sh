#!/bin/bash
# ============================================================
# CRYPTO ALPHA BOT — Mac Mini M4 Setup Script
# ============================================================
# Run this once to set up everything:
#   chmod +x setup_m4.sh && ./setup_m4.sh
# ============================================================

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   🚀 CRYPTO ALPHA BOT — M4 Mac Mini Setup                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "⚠️  This script is designed for macOS. Adjust for your OS."
fi

# Check for Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version)
    echo "  ✅ $PY_VERSION"
else
    echo "  ❌ Python3 not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install python@3.12
    else
        echo "  Install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "  ✅ Virtual environment created and activated"

# Install dependencies
echo ""
echo "Installing Python packages..."
pip install --upgrade pip

# Core packages
pip install ccxt pandas numpy requests aiohttp websockets pyyaml python-dotenv loguru rich

# ML packages (Apple Silicon optimized)
pip install scikit-learn lightgbm xgboost

# PyTorch with MPS (Metal Performance Shaders) for M4 GPU
pip install torch torchvision torchaudio

# Try MLX (Apple's native ML framework — blazing fast on M4)
pip install mlx 2>/dev/null || echo "  ⚠️  MLX install failed — optional, continuing..."

# Technical analysis
pip install ta pandas-ta

# Dashboard
pip install streamlit plotly matplotlib

# Scheduling
pip install apscheduler schedule

echo "  ✅ All packages installed"

# Create directories
echo ""
echo "Creating project directories..."
mkdir -p data/ohlcv data/sentiment data/onchain
mkdir -p models
mkdir -p logs
echo "  ✅ Directories created"

# Copy config
if [ ! -f config.yaml ]; then
    cp config_example.yaml config.yaml
    echo "  ✅ Config file created (config.yaml)"
    echo "  ⚠️  IMPORTANT: Edit config.yaml and add your exchange API keys!"
else
    echo "  ℹ️  config.yaml already exists — not overwriting"
fi

# Add to .gitignore
cat > .gitignore << 'EOF'
config.yaml
.venv/
data/
models/
logs/
__pycache__/
*.pyc
.DS_Store
EOF
echo "  ✅ .gitignore created"

# Verify GPU/MPS availability
echo ""
echo "Checking Apple Silicon GPU (MPS)..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('  ✅ MPS (Metal) GPU available — PyTorch will use M4 GPU!')
    device = torch.device('mps')
    x = torch.randn(1000, 1000, device=device)
    y = torch.matmul(x, x)
    print(f'  ✅ GPU test passed — matrix multiply on M4 GPU successful')
else:
    print('  ⚠️  MPS not available — will use CPU (still fast on M4)')
" 2>/dev/null || echo "  ⚠️  Could not test GPU — continuing..."

# Quick validation
echo ""
echo "Running quick validation..."
python3 -c "
import ccxt, pandas, numpy, yaml, requests
print('  ✅ All core imports successful')
exchange = ccxt.binance({'enableRateLimit': True})
ticker = exchange.fetch_ticker('BTC/USDT')
print(f'  ✅ Exchange connection OK — BTC: \${ticker[\"last\"]:,.2f}')
"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   ✅ SETUP COMPLETE!                                            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                  ║"
echo "║   1. Edit config.yaml — add your Binance API keys               ║"
echo "║                                                                  ║"
echo "║   2. Start data collection (DO THIS NOW):                        ║"
echo "║      source .venv/bin/activate                                   ║"
echo "║      python3 data_collector.py                                   ║"
echo "║                                                                  ║"
echo "║   3. Quick test (see live market scan):                          ║"
echo "║      python3 quick_start.py                                      ║"
echo "║                                                                  ║"
echo "║   4. Dashboard:                                                  ║"
echo "║      streamlit run dashboard.py                                  ║"
echo "║                                                                  ║"
echo "║   5. Backtest (after 7+ days of data):                           ║"
echo "║      python3 backtest.py                                         ║"
echo "║                                                                  ║"
echo "║   6. Train models (after 14+ days of data):                      ║"
echo "║      python3 train_models.py                                     ║"
echo "║                                                                  ║"
echo "║   7. Go live (when bull market confirmed):                       ║"
echo "║      python3 live_trader.py                                      ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# Create a helper to run data collector as background service
cat > start_collector.sh << 'EOF'
#!/bin/bash
# Start data collector as background process
source .venv/bin/activate
nohup python3 data_collector.py --interval 60 > logs/collector_stdout.log 2>&1 &
echo $! > logs/collector.pid
echo "Data collector started (PID: $(cat logs/collector.pid))"
echo "Logs: tail -f logs/collector_stdout.log"
EOF
chmod +x start_collector.sh

cat > stop_collector.sh << 'EOF'
#!/bin/bash
# Stop data collector
if [ -f logs/collector.pid ]; then
    kill $(cat logs/collector.pid) 2>/dev/null
    rm logs/collector.pid
    echo "Data collector stopped"
else
    echo "No collector PID found"
fi
EOF
chmod +x stop_collector.sh

echo ""
echo "  Helper scripts created:"
echo "    ./start_collector.sh  — Start data collector in background"
echo "    ./stop_collector.sh   — Stop data collector"
echo ""

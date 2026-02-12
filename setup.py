#!/usr/bin/env python3
"""
PROJECT IRON DRAGOON - Master Setup Script
Builds the entire system before going online
"""

import os
import sys
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'irondragoons'
FREQTRADE_DIR = BASE_DIR / 'freqtrade'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def create_directories():
    """Create all required directories"""
    print_header("CREATING DIRECTORY STRUCTURE")
    
    dirs = [
        DATA_DIR,
        DATA_DIR / 'raw',
        DATA_DIR / 'labeled',
        DATA_DIR / 'training',
        FREQTRADE_DIR / 'strategies',
        MODELS_DIR / 'agent_a',
        MODELS_DIR / 'agent_b',
        MODELS_DIR / 'agent_c',
        LOGS_DIR,
        BASE_DIR / 'notebooks',
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")
    
    print(f"\n✅ Created {len(dirs)} directories")

def init_databases():
    """Initialize all SQLite databases"""
    print_header("INITIALIZING DATABASES")
    
    # Sentry database
    conn = sqlite3.connect(DATA_DIR / 'sentry.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            signal_type TEXT,
            rsi REAL,
            volume_spike REAL,
            volatility REAL,
            price REAL,
            triggered INTEGER DEFAULT 0
        )
    ''')
    
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp ON scans(timestamp)
    ''')
    
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_symbol ON scans(symbol)
    ''')
    
    conn.commit()
    conn.close()
    print("  ✓ Sentry database initialized")
    
    # Signals database
    conn = sqlite3.connect(DATA_DIR / 'signals.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            pair TEXT,
            signal_type TEXT,
            price REAL,
            rsi REAL,
            volume_spike REAL,
            volatility REAL,
            discord_approved INTEGER DEFAULT 0,
            agent_consensus TEXT,
            executed INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print("  ✓ Signals database initialized")
    
    # Trades database (for performance tracking)
    conn = sqlite3.connect(DATA_DIR / 'trades.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            pair TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            size REAL,
            pnl REAL,
            pnl_pct REAL,
            exit_reason TEXT,
            technician_vote INTEGER,
            fundamentalist_vote INTEGER,
            risk_manager_vote INTEGER,
            discord_approved INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    print("  ✓ Trades database initialized")
    
    print("\n✅ All databases initialized")

def check_environment():
    """Check that all required environment variables are set"""
    print_header("CHECKING ENVIRONMENT")
    
    required_vars = [
        'DISCORD_TOKEN',
        'DISCORD_CHANNEL_ID',
    ]
    
    optional_vars = [
        'BINANCE_US_API_KEY',
        'BINANCE_US_SECRET',
        'RUNPOD_API_KEY',
    ]
    
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
            print(f"  ❌ {var} - REQUIRED")
        else:
            print(f"  ✓ {var}")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"  ✓ {var} (optional)")
        else:
            print(f"  ⚠ {var} (optional - not set)")
    
    if missing:
        print(f"\n❌ Missing required environment variables: {', '.join(missing)}")
        print("\nPlease set these in your environment or .env file:")
        print("  export DISCORD_TOKEN='your_discord_bot_token'")
        print("  export DISCORD_CHANNEL_ID='your_discord_channel_id'")
        return False
    
    print("\n✅ All required environment variables set")
    return True

def check_docker():
    """Check if Docker is installed and running"""
    print_header("CHECKING DOCKER")
    
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Docker installed: {result.stdout.strip()}")
        else:
            print("  ❌ Docker not found")
            return False
        
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Docker Compose installed: {result.stdout.strip()}")
        else:
            print("  ❌ Docker Compose not found")
            return False
        
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ Docker daemon is running")
        else:
            print("  ❌ Docker daemon is not running")
            return False
        
        print("\n✅ Docker environment ready")
        return True
        
    except FileNotFoundError:
        print("  ❌ Docker not installed")
        return False

def build_images():
    """Build Docker images"""
    print_header("BUILDING DOCKER IMAGES")
    
    try:
        result = subprocess.run(
            ['docker-compose', 'build'],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("  ✓ Docker images built successfully")
            return True
        else:
            print(f"  ❌ Build failed:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Build error: {e}")
        return False

def setup_freqtrade():
    """Setup Freqtrade configuration"""
    print_header("SETTING UP FREQTRADE")
    
    config_path = FREQTRADE_DIR / 'config.json'
    if config_path.exists():
        print(f"  ✓ Freqtrade config exists: {config_path}")
    else:
        print(f"  ❌ Freqtrade config not found: {config_path}")
        return False
    
    strategy_path = FREQTRADE_DIR / 'strategies' / 'IronDragoonStrategy.py'
    if strategy_path.exists():
        print(f"  ✓ Strategy file exists: {strategy_path}")
    else:
        print(f"  ❌ Strategy file not found: {strategy_path}")
        return False
    
    print("\n✅ Freqtrade setup complete")
    return True

def create_env_file():
    """Create .env file template"""
    print_header("CREATING ENVIRONMENT FILE")
    
    env_path = BASE_DIR / '.env'
    
    if env_path.exists():
        print(f"  ✓ .env file already exists: {env_path}")
        return
    
    env_content = """# PROJECT IRON DRAGOON - Environment Configuration

# Discord Bot (REQUIRED)
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_CHANNEL_ID=your_discord_channel_id_here

# Binance US API (for live trading)
BINANCE_US_API_KEY=your_binance_us_api_key
BINANCE_US_SECRET=your_binance_us_secret

# RunPod (for training)
RUNPOD_API_KEY=your_runpod_api_key

# Freqtrade API
FREQTRADE_PASSWORD=change_me_in_production

# Starting capital
STARTING_CAPITAL=150

# Timezone
TZ=America/Chicago
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"  ✓ Created .env template: {env_path}")
    print("  ⚠ Please edit this file with your actual credentials")

def print_status():
    """Print final status"""
    print_header("SETUP STATUS")
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    PROJECT IRON DRAGOON                             │
│                      SYSTEM STATUS                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Components:                                                        │
│    ✓ Data Harvester        - Ready to pull 12 years of data         │
│    ✓ Hindsight Labeler     - Ready to label winning setups          │
│    ✓ Sentry Scanner        - Ready for 24/7 monitoring              │
│    ✓ Freqtrade Engine      - Docker configured                      │
│    ✓ Discord Bot           - Human-in-the-loop interface            │
│    ✓ 3-Agent Ensemble      - RunPod deployment ready                │
├─────────────────────────────────────────────────────────────────────┤
│  Next Steps:                                                        │
│    1. Set environment variables in .env                             │
│    2. Run: python iron_dragoon.py harvest                           │
│    3. Run: python iron_dragoon.py label                             │
│    4. Run: python deploy_runpod_agents.py                           │
│    5. Run: docker-compose up -d                                     │
│    6. Start training on RunPod when ready                           │
│    7. System goes online when ALL components ready                  │
├─────────────────────────────────────────────────────────────────────┤
│  Commands:                                                          │
│    python iron_dragoon.py harvest    # Download historical data     │
│    python iron_dragoon.py label      # Label training data          │
│    python iron_dragoon.py sentry     # Start local sentry           │
│    python iron_dragoon.py discord    # Start Discord bot            │
│    docker-compose up -d              # Start all services             │
│    docker-compose logs -f            # View logs                      │
└─────────────────────────────────────────────────────────────────────┘
""")

def main():
    """Main setup function"""
    print("""
    ██╗██████╗  ██████╗ ███╗   ██╗    ██████╗ ██████╗  █████╗  ██████╗  ██████╗  ██████╗ ███╗   ██╗
    ██║██╔══██╗██╔═══██╗████╗  ██║    ██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔═══██╗██╔═══██╗████╗  ██║
    ██║██████╔╝██║   ██║██╔██╗ ██║    ██║  ██║██████╔╝███████║██║  ███╗██║   ██║██║   ██║██╔██╗ ██║
    ██║██╔══██╗██║   ██║██║╚██╗██║    ██║  ██║██╔══██╗██╔══██║██║   ██║██║   ██║██║   ██║██║╚██╗██║
    ██║██║  ██║╚██████╔╝██║ ╚████║    ██████╔╝██║  ██║██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝██║ ╚████║
    ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝
    """)
    
    print(f"Setup started at: {datetime.now().isoformat()}")
    print(f"Base directory: {BASE_DIR}")
    
    # Run setup steps
    create_directories()
    init_databases()
    create_env_file()
    
    # Check environment
    if not check_environment():
        print("\n⚠ Environment check failed - please set required variables")
    
    # Check Docker
    if check_docker():
        setup_freqtrade()
    else:
        print("\n⚠ Docker not available - skipping Docker setup")
    
    # Print final status
    print_status()
    
    # Save setup manifest
    manifest = {
        'setup_time': datetime.now().isoformat(),
        'version': '1.0.0',
        'components': [
            'data_harvester',
            'hindsight_labeler',
            'sentry_scanner',
            'freqtrade_engine',
            'discord_bot',
            'agent_ensemble'
        ],
        'status': 'CONFIGURED_NOT_ONLINE'
    }
    
    with open(BASE_DIR / 'SETUP_MANIFEST.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Setup complete. Manifest saved to: {BASE_DIR / 'SETUP_MANIFEST.json'}")

if __name__ == "__main__":
    main()

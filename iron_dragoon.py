#!/usr/bin/env python3
"""
PROJECT IRON DRAGOON - Full System Implementation
Binance US | $150 Starting Capital | Discord Command Interface
"""

import os
import json
import sqlite3
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import discord
from discord.ext import commands, tasks

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'exchange': 'binanceus',
    'starting_capital': 150.0,
    'target_profit': 0.05,      # 5% profit target
    'stop_loss': 0.02,          # 2% stop loss
    'time_horizon': 24,         # 24 hour horizon
    'max_positions': 3,         # Max concurrent positions
    'drawdown_limit': 0.10,     # 10% max drawdown
    
    # Target coins for Binance US
    'target_coins': [
        'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD',
        'DOT/USD', 'LINK/USD', 'MATIC/USD', 'AVAX/USD',
        'DOGE/USD', 'XRP/USD', 'LTC/USD', 'UNI/USD'
    ],
    
    # Sentry thresholds
    'volume_spike_threshold': 3.0,      # 300% volume spike
    'volatility_threshold': 0.05,       # 5% volatility
    'rsi_overbought': 75,
    'rsi_oversold': 25,
    
    # Discord
    'discord_channel_id': None,  # Will be set from env
    
    # Paths
    'data_dir': 'data/irondragoons',
    'models_dir': 'models',
    'logs_dir': 'logs',
}

# =============================================================================
# DATA HARVESTER - Script 1 from Blueprint
# =============================================================================

class DataHarvester:
    """Pulls raw training data from Binance US"""
    
    def __init__(self):
        self.exchange = ccxt.binanceus({
            'enableRateLimit': True,
        })
        self.data_dir = Path(CONFIG['data_dir']) / 'raw'
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_ohlcv(self, symbol, timeframe='1h', days_back=365):
        """Fetch OHLCV data for a symbol"""
        print(f"⚔ HARVESTING DATA FOR: {symbol}")
        
        since = self.exchange.milliseconds() - (days_back * 24 * 60 * 60 * 1000)
        all_ohlcv = []
        
        while since < self.exchange.milliseconds():
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=1000, since=since)
                if len(ohlcv) == 0:
                    break
                since = ohlcv[-1][0] + 1
                all_ohlcv += ohlcv
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Save
        filename = f"{symbol.replace('/', '_')}_raw.csv"
        df.to_csv(self.data_dir / filename, index=False)
        print(f"✅ Secured {len(df)} candles for {symbol}")
        
        return df
    
    def harvest_all(self):
        """Harvest data for all target coins"""
        for coin in CONFIG['target_coins']:
            try:
                self.fetch_ohlcv(coin)
            except Exception as e:
                print(f"❌ Failed to harvest {coin}: {e}")

# =============================================================================
# HINDSIGHT LABELER - Script 2 from Blueprint
# =============================================================================

class HindsightLabeler:
    """Creates the 'Answer Key' - marks trades that WOULD HAVE won"""
    
    def __init__(self):
        self.raw_dir = Path(CONFIG['data_dir']) / 'raw'
        self.labeled_dir = Path(CONFIG['data_dir']) / 'labeled'
        self.labeled_dir.mkdir(parents=True, exist_ok=True)
    
    def label_data(self, filename):
        """Label a single raw data file"""
        df = pd.read_csv(self.raw_dir / filename)
        
        # "Look Ahead" Window
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=CONFIG['time_horizon'])
        df['future_high'] = df['high'].rolling(window=indexer).max()
        df['future_low'] = df['low'].rolling(window=indexer).min()
        
        # THE GOLDEN RULE: Did it hit Profit BEFORE it hit Stop Loss?
        conditions = [
            (df['future_high'] > df['close'] * (1 + CONFIG['target_profit'])) &
            (df['future_low'] > df['close'] * (1 - CONFIG['stop_loss']))
        ]
        
        # Label: 1 = BUY, 0 = WAIT
        df['target_action'] = np.select(conditions, [1], default=0)
        
        # Add technical indicators
        df = self.add_indicators(df)
        
        # Save
        output_name = filename.replace('_raw.csv', '_labeled.csv')
        df.to_csv(self.labeled_dir / output_name, index=False)
        
        win_count = df['target_action'].sum()
        print(f"🎯 Labeled {filename}: Found {win_count} winning setups ({win_count/len(df)*100:.1f}%)")
        
        return df
    
    def add_indicators(self, df):
        """Add technical indicators (Python does the math, not LLM)"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Moving averages
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        df['ma_90'] = df['close'].rolling(window=90).mean()
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=30).std()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=30).mean()
        df['volume_spike'] = df['volume'] / df['volume_ma']
        
        return df
    
    def label_all(self):
        """Label all raw files"""
        for f in self.raw_dir.glob('*_raw.csv'):
            try:
                self.label_data(f.name)
            except Exception as e:
                print(f"❌ Failed to label {f.name}: {e}")

# =============================================================================
# THE SENTRY - Local Mac Mini Watchman
# =============================================================================

class Sentry:
    """24/7 Watchman - Scans for Volume Spikes and Volatility Anomalies"""
    
    def __init__(self):
        self.exchange = ccxt.binanceus({'enableRateLimit': True})
        self.alerts = []
        self.db_path = Path(CONFIG['data_dir']) / 'sentry.db'
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database for sentry data"""
        conn = sqlite3.connect(self.db_path)
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
        conn.commit()
        conn.close()
    
    def scan_coin(self, symbol):
        """Scan a single coin for signals"""
        try:
            # Fetch recent candles
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate indicators
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=24).std()
            df['volume_ma'] = df['volume'].rolling(window=24).mean()
            df['volume_spike'] = df['volume'] / df['volume_ma']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            latest = df.iloc[-1]
            
            signals = []
            
            # Volume spike detection
            if latest['volume_spike'] > CONFIG['volume_spike_threshold']:
                signals.append({
                    'type': 'VOLUME_SPIKE',
                    'symbol': symbol,
                    'value': latest['volume_spike'],
                    'price': latest['close'],
                    'rsi': latest['rsi']
                })
            
            # Volatility anomaly
            if latest['volatility'] > CONFIG['volatility_threshold']:
                signals.append({
                    'type': 'VOLATILITY_ANOMALY',
                    'symbol': symbol,
                    'value': latest['volatility'],
                    'price': latest['close'],
                    'rsi': latest['rsi']
                })
            
            # RSI extremes
            if latest['rsi'] < CONFIG['rsi_oversold']:
                signals.append({
                    'type': 'RSI_OVERSOLD',
                    'symbol': symbol,
                    'value': latest['rsi'],
                    'price': latest['close'],
                    'rsi': latest['rsi']
                })
            elif latest['rsi'] > CONFIG['rsi_overbought']:
                signals.append({
                    'type': 'RSI_OVERBOUGHT',
                    'symbol': symbol,
                    'value': latest['rsi'],
                    'price': latest['close'],
                    'rsi': latest['rsi']
                })
            
            # Log to database
            if signals:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                for signal in signals:
                    c.execute('''
                        INSERT INTO scans (timestamp, symbol, signal_type, rsi, volume_spike, volatility, price)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        datetime.now().isoformat(),
                        signal['symbol'],
                        signal['type'],
                        signal['rsi'],
                        signal.get('volume_spike', 0),
                        signal.get('volatility', 0),
                        signal['price']
                    ))
                conn.commit()
                conn.close()
            
            return signals
            
        except Exception as e:
            print(f"❌ Sentry error for {symbol}: {e}")
            return []
    
    def scan_all(self):
        """Scan all target coins"""
        all_signals = []
        for coin in CONFIG['target_coins']:
            signals = self.scan_coin(coin)
            all_signals.extend(signals)
        return all_signals

# =============================================================================
# DISCORD BOT - Human-in-the-Loop Interface
# =============================================================================

class IronDragoonBot(commands.Bot):
    """Discord bot for human-in-the-loop control"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.sentry = Sentry()
        self.pending_trades = []
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.peak_value = CONFIG['starting_capital']
        self.current_value = CONFIG['starting_capital']
        self.drawdown_triggered = False
    
    async def setup_hook(self):
        self.sentry_scan.start()
    
    @tasks.loop(minutes=5)
    async def sentry_scan(self):
        """Run sentry scan every 5 minutes"""
        if self.drawdown_triggered:
            return
        
        signals = self.sentry.scan_all()
        
        for signal in signals:
            # Check if we should alert
            if signal['type'] in ['VOLUME_SPIKE', 'RSI_OVERSOLD']:
                if len(self.active_positions) < CONFIG['max_positions']:
                    await self.alert_signal(signal)
    
    async def alert_signal(self, signal):
        """Send alert to Discord for human approval"""
        channel = self.get_channel(int(os.getenv('DISCORD_CHANNEL_ID', 0)))
        if not channel:
            return
        
        embed = discord.Embed(
            title=f"🎯 TARGET ACQUIRED: {signal['symbol']}",
            description=f"Signal: {signal['type']}",
            color=discord.Color.gold()
        )
        embed.add_field(name="Price", value=f"${signal['price']:.2f}", inline=True)
        embed.add_field(name="RSI", value=f"{signal['rsi']:.1f}", inline=True)
        embed.add_field(name="Volume Spike", value=f"{signal.get('volume_spike', 0):.2f}x", inline=True)
        embed.add_field(name="Action", value="React with ✅ to approve trade
React with ❌ to reject", inline=False)
        
        message = await channel.send(embed=embed)
        await message.add_reaction('✅')
        await message.add_reaction('❌')
        
        self.pending_trades.append({
            'message_id': message.id,
            'signal': signal,
            'timestamp': datetime.now()
        })
    
    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        """Handle trade approval/rejection"""
        if user.bot:
            return
        
        for trade in self.pending_trades:
            if trade['message_id'] == reaction.message.id:
                if reaction.emoji == '✅':
                    await self.execute_trade(trade['signal'])
                elif reaction.emoji == '❌':
                    await reaction.message.channel.send(f"❌ Trade rejected for {trade['signal']['symbol']}")
                
                self.pending_trades.remove(trade)
                break
    
    async def execute_trade(self, signal):
        """Execute approved trade with trailing entry"""
        symbol = signal['symbol']
        current_price = signal['price']
        
        # Trailing entry: Place buy stop 1% above current price
        entry_price = current_price * 1.01
        
        # Calculate position size (max 1/3 of capital per position)
        position_value = self.current_value / CONFIG['max_positions']
        
        # Set stop loss and take profit
        stop_loss = entry_price * (1 - CONFIG['stop_loss'])
        take_profit = entry_price * (1 + CONFIG['target_profit'])
        
        self.active_positions[symbol] = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_value': position_value,
            'entry_time': datetime.now()
        }
        
        channel = self.get_channel(int(os.getenv('DISCORD_CHANNEL_ID', 0)))
        if channel:
            embed = discord.Embed(
                title=f"🔫 TRADE EXECUTED: {symbol}",
                color=discord.Color.green()
            )
            embed.add_field(name="Entry", value=f"${entry_price:.2f}", inline=True)
            embed.add_field(name="Stop Loss", value=f"${stop_loss:.2f}", inline=True)
            embed.add_field(name="Take Profit", value=f"${take_profit:.2f}", inline=True)
            embed.add_field(name="Position Size", value=f"${position_value:.2f}", inline=True)
            await channel.send(embed=embed)
    
    @commands.command()
    async def status(self, ctx):
        """Show current portfolio status"""
        embed = discord.Embed(
            title="📊 PORTFOLIO STATUS",
            color=discord.Color.blue()
        )
        embed.add_field(name="Current Value", value=f"${self.current_value:.2f}", inline=True)
        embed.add_field(name="Starting Value", value=f"${CONFIG['starting_capital']:.2f}", inline=True)
        embed.add_field(name="Peak Value", value=f"${self.peak_value:.2f}", inline=True)
        embed.add_field(name="Active Positions", value=f"{len(self.active_positions)}/{CONFIG['max_positions']}", inline=True)
        embed.add_field(name="Daily P&L", value=f"${self.daily_pnl:.2f}", inline=True)
        embed.add_field(name="Status", value="🟢 ACTIVE" if not self.drawdown_triggered else "🔴 HALTED", inline=True)
        
        if self.active_positions:
            positions_text = "\n".join([f"{s}: ${p['entry_price']:.2f}" for s, p in self.active_positions.items()])
            embed.add_field(name="Open Positions", value=positions_text, inline=False)
        
        await ctx.send(embed=embed)
    
    @commands.command()
    async def halt(self, ctx):
        """Emergency halt all trading"""
        self.drawdown_triggered = True
        await ctx.send("🛑 EMERGENCY HALT ACTIVATED. All trading paused.")
    
    @commands.command()
    async def resume(self, ctx):
        """Resume trading"""
        self.drawdown_triggered = False
        await ctx.send("🟢 TRADING RESUMED.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python iron_dragoon.py [harvest|label|sentry|discord]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "harvest":
        harvester = DataHarvester()
        harvester.harvest_all()
    
    elif command == "label":
        labeler = HindsightLabeler()
        labeler.label_all()
    
    elif command == "sentry":
        sentry = Sentry()
        while True:
            signals = sentry.scan_all()
            if signals:
                print(f"🚨 Found {len(signals)} signals")
                for s in signals:
                    print(f"   {s['symbol']}: {s['type']}")
            time.sleep(300)  # 5 minutes
    
    elif command == "discord":
        bot = IronDragoonBot()
        bot.run(os.getenv('DISCORD_TOKEN'))
    
    else:
        print(f"Unknown command: {command}")

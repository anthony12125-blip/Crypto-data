# =============================================================================
# PROJECT IRON DRAGOON - Strategy
# Human-in-the-Loop AI-assisted trading strategy
# =============================================================================

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union, List
import logging
import sqlite3
import json
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class IronDragoonStrategy(IStrategy):
    """
    Iron Dragoon Strategy
    - Waits for Discord approval before entering trades
    - Uses trailing entry (Buy Stop) instead of market orders
    - Hard-coded technical analysis (no LLM math)
    - Integrates with 3-agent ensemble for final confirmation
    """
    
    INTERFACE_VERSION = 3
    
    # Minimal ROI - we use custom exit logic
    minimal_roi = {
        "0": 0.10  # 10% default
    }
    
    # Stoploss - we manage this manually
    stoploss = -0.02
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    # Timeframe
    timeframe = '1h'
    
    # Startup candle count
    startup_candle_count = 100
    
    # Can short? No - spot trading only
    can_short = False
    
    # Process only new candles
    process_only_new_candles = True
    
    # Use sell signal?
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Strategy parameters
    target_profit = 0.05
    stop_loss_pct = 0.02
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Database connection
        self.db_path = Path('data/irondragoons/sentry.db')
        self.signals_db = Path('data/irondragoons/signals.db')
        self.init_signals_db()
        
        # Pending signals awaiting Discord approval
        self.pending_signals = {}
        
        # Agent ensemble endpoint (RunPod)
        self.agent_endpoint = config.get('agent_endpoint', None)
        
        logger.info("Iron Dragoon Strategy initialized")
    
    def init_signals_db(self):
        """Initialize database for trade signals"""
        conn = sqlite3.connect(self.signals_db)
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
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add technical indicators using hard-coded Python math
        (NOT LLM calculations)
        """
        # RSI
        delta = dataframe['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        dataframe['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = dataframe['close'].ewm(span=12).mean()
        exp2 = dataframe['close'].ewm(span=26).mean()
        dataframe['macd'] = exp1 - exp2
        dataframe['macd_signal'] = dataframe['macd'].ewm(span=9).mean()
        dataframe['macd_hist'] = dataframe['macd'] - dataframe['macd_signal']
        
        # Bollinger Bands
        dataframe['bb_middle'] = dataframe['close'].rolling(window=20).mean()
        bb_std = dataframe['close'].rolling(window=20).std()
        dataframe['bb_upper'] = dataframe['bb_middle'] + (bb_std * 2)
        dataframe['bb_lower'] = dataframe['bb_middle'] - (bb_std * 2)
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # Moving averages
        dataframe['ema_7'] = dataframe['close'].ewm(span=7).mean()
        dataframe['ema_30'] = dataframe['close'].ewm(span=30).mean()
        dataframe['ema_90'] = dataframe['close'].ewm(span=90).mean()
        
        # Volume analysis
        dataframe['volume_ma'] = dataframe['volume'].rolling(window=24).mean()
        dataframe['volume_spike'] = dataframe['volume'] / dataframe['volume_ma']
        
        # Volatility
        dataframe['returns'] = dataframe['close'].pct_change()
        dataframe['volatility'] = dataframe['returns'].rolling(window=24).std()
        
        # Price position within BB
        dataframe['bb_position'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        
        # Trend direction
        dataframe['trend'] = np.where(
            dataframe['ema_7'] > dataframe['ema_30'],
            'uptrend',
            np.where(dataframe['ema_7'] < dataframe['ema_30'], 'downtrend', 'sideways')
        )
        
        # Signal summary (for AI input)
        dataframe['signal_summary'] = self.create_signal_summary(dataframe)
        
        return dataframe
    
    def create_signal_summary(self, dataframe: DataFrame) -> pd.Series:
        """Create natural language signal summary for AI input"""
        summaries = []
        
        for idx, row in dataframe.iterrows():
            rsi_status = "Overbought" if row['rsi'] > 75 else "Oversold" if row['rsi'] < 25 else "Neutral"
            volume_status = "Spike" if row['volume_spike'] > 3 else "Normal"
            trend = row['trend']
            bb_position = row['bb_position']
            
            summary = (
                f"RSI is {rsi_status} ({row['rsi']:.1f}). "
                f"Trend is {trend}. "
                f"Volume shows {volume_status} ({row['volume_spike']:.1f}x avg). "
                f"Price is at {bb_position*100:.1f}% of Bollinger Band range. "
                f"Volatility is {row['volatility']*100:.1f}%."
            )
            summaries.append(summary)
        
        return pd.Series(summaries, index=dataframe.index)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions
        Note: Actual entry requires Discord approval
        """
        dataframe.loc[
            (
                # RSI oversold bounce setup
                (dataframe['rsi'] < 35) &
                (dataframe['rsi'].shift(1) < dataframe['rsi']) &  # RSI turning up
                (dataframe['volume_spike'] > 2.0) &
                (dataframe['close'] > dataframe['ema_7'])
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                # Volume breakout setup
                (dataframe['volume_spike'] > 3.0) &
                (dataframe['close'] > dataframe['bb_upper']) &
                (dataframe['macd'] > dataframe['macd_signal']) &
                (dataframe['trend'] == 'uptrend')
            ),
            'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions"""
        dataframe.loc[
            (
                # RSI overbought
                (dataframe['rsi'] > 75) |
                # Price below lower BB
                (dataframe['close'] < dataframe['bb_lower']) |
                # MACD bearish crossover
                ((dataframe['macd'] < dataframe['macd_signal']) & 
                 (dataframe['macd'].shift(1) > dataframe['macd_signal'].shift(1)))
            ),
            'exit_long'] = 1
        
        return dataframe
    
    def custom_entry_price(self, pair: str, trade_size: float, entry_tag: str, 
                           side: str, **kwargs) -> float:
        """
        Trailing Entry: Place Buy Stop 1% above current price
        This confirms momentum before risking capital
        """
        current_rate = kwargs.get('current_rate', 0)
        if current_rate == 0:
            return None
        
        # Buy stop 1% above current price
        buy_stop_price = current_rate * 1.01
        
        logger.info(f"Trailing entry for {pair}: Buy stop at ${buy_stop_price:.2f} (current: ${current_rate:.2f})")
        
        return buy_stop_price
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        """
        Dynamic stoploss based on volatility
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Adjust stop based on volatility
        if last_candle['volatility'] > 0.05:  # High volatility
            return -0.03  # Wider stop
        elif last_candle['volatility'] < 0.02:  # Low volatility
            return -0.015  # Tighter stop
        
        return self.stoploss
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        """
        Final confirmation before trade entry
        Check Discord approval and agent consensus
        """
        # Check if we have Discord approval
        conn = sqlite3.connect(self.signals_db)
        c = conn.cursor()
        c.execute('''
            SELECT discord_approved, agent_consensus FROM signals 
            WHERE pair = ? AND executed = 0 
            ORDER BY timestamp DESC LIMIT 1
        ''', (pair,))
        result = c.fetchone()
        conn.close()
        
        if not result:
            logger.warning(f"No signal found for {pair}, rejecting entry")
            return False
        
        discord_approved, agent_consensus = result
        
        if not discord_approved:
            logger.info(f"Discord approval pending for {pair}, delaying entry")
            return False
        
        # Check agent consensus if available
        if agent_consensus:
            consensus = json.loads(agent_consensus)
            technician_vote = consensus.get('technician', 0)
            fundamentalist_vote = consensus.get('fundamentalist', 0)
            risk_manager_vote = consensus.get('risk_manager', 0)
            
            # Require at least 2 of 3 agents to agree
            if (technician_vote + fundamentalist_vote + risk_manager_vote) < 2:
                logger.info(f"Agent consensus not reached for {pair}, rejecting entry")
                return False
        
        # Mark signal as executed
        conn = sqlite3.connect(self.signals_db)
        c = conn.cursor()
        c.execute('UPDATE signals SET executed = 1 WHERE pair = ? AND executed = 0', (pair,))
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Trade confirmed for {pair}")
        return True
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        """No leverage - spot trading only"""
        return 1.0

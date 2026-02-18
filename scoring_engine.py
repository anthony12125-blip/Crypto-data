#!/usr/bin/env python3
"""
PROJECT IRON DRAGOON - ENHANCED SCORING ENGINE
Adapted from Options Bot Research (Piotroski F-Score, Vol-Edge concepts)
Adds quantitative scoring to crypto trading signals
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

class CryptoScoringEngine:
    """
    4-Category Scoring System (0-100 each)
    Inspired by options bot's scoring engine
    """
    
    def __init__(self, db_path='data/irondragoons/sentry.db'):
        self.db_path = Path(db_path)
        self.scores_cache = {}
    
    def score_coin(self, symbol, ohlcv_data, market_context=None):
        """
        Score a single coin across 4 dimensions
        Returns dict with individual scores and composite
        """
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        scores = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'vol_edge': self._calc_vol_edge(df),
            'quality': self._calc_quality(df),
            'setup': self._calc_setup(df),
            'regime': self._calc_regime(df, market_context),
        }
        
        # Composite score (weighted average)
        weights = {'vol_edge': 0.30, 'quality': 0.25, 'setup': 0.30, 'regime': 0.15}
        scores['composite'] = sum(scores[k] * weights[k] for k in weights.keys())
        
        # Signal strength classification
        if scores['composite'] >= 80:
            scores['signal_strength'] = 'STRONG_BUY'
        elif scores['composite'] >= 65:
            scores['signal_strength'] = 'BUY'
        elif scores['composite'] >= 50:
            scores['signal_strength'] = 'NEUTRAL'
        elif scores['composite'] >= 35:
            scores['signal_strength'] = 'AVOID'
        else:
            scores['signal_strength'] = 'STRONG_AVOID'
        
        return scores
    
    def _calc_vol_edge(self, df):
        """
        Volatility Edge Score (0-100)
        Crypto adaptation of options "implied vs historical" concept:
        - Measures if current volatility is high/low relative to recent history
        - Identifies mean reversion opportunities
        """
        if len(df) < 30:
            return 50
        
        # Calculate realized volatility (annualized)
        returns = df['close'].pct_change().dropna()
        
        # Current volatility (7-day)
        current_vol = returns.tail(7).std() * np.sqrt(365) * 100
        
        # Historical volatility regimes (30-day and 90-day)
        vol_30d = returns.tail(30).std() * np.sqrt(365) * 100
        vol_90d = returns.tail(90).std() * np.sqrt(365) * 100 if len(returns) >= 90 else vol_30d
        
        # Volatility percentile (where does current vol rank in last 90 days)
        vol_percentile = (returns.tail(90).rolling(7).std().dropna() < current_vol / np.sqrt(365)).mean() * 100
        
        # Vol Edge Calculation
        # High score when: volatility is high but compressing (mean reversion down)
        # OR volatility is low but expanding (trend forming)
        
        vol_trend = (current_vol - vol_30d) / vol_30d if vol_30d > 0 else 0
        
        # Scoring logic:
        # - Volatility in 40th-70th percentile with positive momentum = high score (trend forming)
        # - Extreme volatility (>90th percentile) with negative momentum = high score (mean reversion)
        # - Low volatility (<20th percentile) = low score (no edge)
        
        if vol_percentile > 90 and vol_trend < -0.1:
            # Extreme vol cooling off - mean reversion opportunity
            score = 85 + min(15, abs(vol_trend) * 100)
        elif 40 <= vol_percentile <= 70 and vol_trend > 0:
            # Sweet spot - moderate vol expanding into trend
            score = 80 + min(15, vol_trend * 200)
        elif vol_percentile < 20:
            # Too quiet - no edge
            score = 30
        elif vol_percentile > 80:
            # Too chaotic - avoid
            score = 40
        else:
            # Middle range
            score = 50 + (vol_percentile - 50) * 0.5
        
        return min(100, max(0, score))
    
    def _calc_quality(self, df):
        """
        Quality Score (0-100)
        Crypto adaptation of Piotroski F-Score concept:
        - Liquidity health (volume consistency)
        - Trend strength (directional conviction)
        - Price stability (avoiding flash crash coins)
        """
        if len(df) < 30:
            return 50
        
        scores = []
        
        # 1. Liquidity Health (30 points)
        volume_ma = df['volume'].rolling(30).mean()
        volume_consistency = 1 - (df['volume'].tail(30).std() / df['volume'].tail(30).mean())
        volume_consistency = max(0, min(1, volume_consistency))
        scores.append(volume_consistency * 30)
        
        # 2. Trend Quality (30 points)
        # ADX-like measure using directional movement
        plus_dm = ((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low'])) & \
                  ((df['high'] - df['high'].shift(1)) > 0)
        minus_dm = ((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1))) & \
                   ((df['low'].shift(1) - df['low']) > 0)
        
        plus_di = plus_dm.rolling(14).mean()
        minus_di = minus_dm.rolling(14).mean()
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
        adx_approx = dx.rolling(14).mean()
        
        trend_score = min(30, adx_approx.iloc[-1] / 3)  # ADX 30 = perfect
        scores.append(trend_score)
        
        # 3. Price Stability (20 points)
        # Avoid coins with massive wicks/gaps
        body_size = abs(df['close'] - df['open']) / df['open']
        wick_size = (df['high'] - df['low']) / df['low']
        wick_ratio = (wick_size / body_size.replace(0, 1e-10)).tail(30).mean()
        
        stability_score = max(0, 20 - min(20, wick_ratio))
        scores.append(stability_score)
        
        # 4. Volume Trend (20 points)
        vol_trend = (df['volume'].tail(7).mean() / df['volume'].tail(30).mean() - 1) * 100
        vol_trend_score = min(20, max(0, 10 + vol_trend / 2))
        scores.append(vol_trend_score)
        
        return sum(scores)
    
    def _calc_setup(self, df):
        """
        Setup Score (0-100)
        Technical pattern recognition:
        - RSI extremes (oversold bounces, overbought fades)
        - Moving average positioning
        - Bollinger Band squeezes/breaks
        - Volume-price divergence
        """
        if len(df) < 50:
            return 50
        
        scores = []
        close = df['close']
        
        # 1. RSI Setup (25 points)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Best setups: RSI 25-40 (oversold bounce) or RSI 60-75 (momentum)
        if 25 <= current_rsi <= 40:
            rsi_score = 25 - (current_rsi - 25) * 0.5
        elif 60 <= current_rsi <= 75:
            rsi_score = 15 + (current_rsi - 60) * 0.67
        else:
            rsi_score = max(0, 25 - abs(current_rsi - 50) * 0.5)
        scores.append(rsi_score)
        
        # 2. Moving Average Position (25 points)
        ma_7 = close.rolling(7).mean()
        ma_30 = close.rolling(30).mean()
        ma_90 = close.rolling(90).mean()
        
        current_price = close.iloc[-1]
        ma_score = 0
        
        # Price above MAs = bullish structure
        if current_price > ma_7.iloc[-1]:
            ma_score += 8
        if current_price > ma_30.iloc[-1]:
            ma_score += 8
        if current_price > ma_90.iloc[-1]:
            ma_score += 9
        
        # Bonus for 7 > 30 > 90 alignment
        if ma_7.iloc[-1] > ma_30.iloc[-1] > ma_90.iloc[-1]:
            ma_score += 5
        
        scores.append(min(25, ma_score))
        
        # 3. Bollinger Band Squeeze (25 points)
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # Bollinger Band Width (squeeze indicator)
        bb_width = (bb_upper - bb_lower) / bb_middle
        current_width = bb_width.iloc[-1]
        width_20d_avg = bb_width.tail(20).mean()
        
        # Low width = potential breakout
        if current_width < width_20d_avg * 0.8:
            bb_score = 25  # Squeeze setup
        elif current_width > width_20d_avg * 1.2:
            bb_score = 10  # Too wide, already moving
        else:
            bb_score = 17
        scores.append(bb_score)
        
        # 4. Volume-Price Divergence (25 points)
        price_change = (close.iloc[-1] / close.iloc[-5] - 1) * 100
        volume_change = (df['volume'].tail(5).mean() / df['volume'].tail(20).mean() - 1) * 100
        
        # Confirming move: price up + volume up = bullish
        if price_change > 0 and volume_change > 10:
            div_score = 25
        elif price_change < 0 and volume_change > 10:
            div_score = 15  # Distribution
        elif price_change > 0 and volume_change < -10:
            div_score = 10  # Weak move
        else:
            div_score = 18
        scores.append(div_score)
        
        return sum(scores)
    
    def _calc_regime(self, df, market_context=None):
        """
        Regime Score (0-100)
        Crypto adaptation of macro context:
        - BTC dominance trends (alt season vs BTC season)
        - Funding rates (perpetual futures sentiment)
        - Market correlation (is this coin moving with or against BTC?)
        """
        if market_context is None:
            # Default neutral if no context provided
            return 50
        
        scores = []
        
        # 1. BTC Dominance (25 points)
        btc_dom = market_context.get('btc_dominance', 50)
        # Alts do better when BTC dominance < 50
        if btc_dom < 45:
            btc_score = 25  # Alt season
        elif btc_dom > 60:
            btc_score = 10  # BTC season
        else:
            btc_score = 20
        scores.append(btc_score)
        
        # 2. Funding Rates (25 points)
        funding = market_context.get('funding_rate', 0)
        # Negative funding = shorts paying longs = bullish
        if funding < -0.01:
            funding_score = 25
        elif funding > 0.01:
            funding_score = 10
        else:
            funding_score = 20
        scores.append(funding_score)
        
        # 3. Correlation Regime (25 points)
        corr = market_context.get('btc_correlation', 0.7)
        # High correlation = follow BTC; Low = independent
        if corr > 0.8:
            corr_score = 15  # Just follows BTC
        elif corr < 0.4:
            corr_score = 25  # Independent mover
        else:
            corr_score = 20
        scores.append(corr_score)
        
        # 4. Market Momentum (25 points)
        btc_trend = market_context.get('btc_7d_change', 0)
        if btc_trend > 10:
            mom_score = 25  # Strong uptrend
        elif btc_trend < -10:
            mom_score = 10  # Downtrend
        else:
            mom_score = 20
        scores.append(mom_score)
        
        return sum(scores)
    
    def rank_signals(self, signals_list, ohlcv_dict, market_context=None):
        """
        Rank a list of signals by composite score
        Returns sorted list with scores attached
        """
        scored_signals = []
        
        for signal in signals_list:
            symbol = signal['symbol']
            ohlcv = ohlcv_dict.get(symbol)
            
            if ohlcv is None:
                continue
            
            scores = self.score_coin(symbol, ohlcv, market_context)
            signal['scores'] = scores
            signal['composite_score'] = scores['composite']
            signal['signal_strength'] = scores['signal_strength']
            
            scored_signals.append(signal)
        
        # Sort by composite score descending
        scored_signals.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return scored_signals
    
    def save_scores(self, scores):
        """Save scores to database for tracking"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                vol_edge REAL,
                quality REAL,
                setup REAL,
                regime REAL,
                composite REAL,
                signal_strength TEXT
            )
        ''')
        
        c.execute('''
            INSERT INTO scores 
            (timestamp, symbol, vol_edge, quality, setup, regime, composite, signal_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scores['timestamp'],
            scores['symbol'],
            scores['vol_edge'],
            scores['quality'],
            scores['setup'],
            scores['regime'],
            scores['composite'],
            scores['signal_strength']
        ))
        
        conn.commit()
        conn.close()

# Integration with existing Sentry class
def enhance_sentry_with_scoring():
    """
    Call this to add scoring to the existing Sentry class
    Usage: In iron_dragoon.py, add scoring before alerting
    """
    pass  # Placeholder for integration instructions

if __name__ == '__main__':
    # Test the scoring engine
    print("🧪 Testing Crypto Scoring Engine...")
    
    # Generate synthetic test data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    test_data = []
    price = 100
    for i in range(100):
        change = np.random.normal(0, 0.02)
        price *= (1 + change)
        test_data.append([
            int(dates[i].timestamp() * 1000),
            price * (1 - abs(np.random.normal(0, 0.01))),
            price * (1 + abs(np.random.normal(0, 0.01))),
            price * (1 - abs(np.random.normal(0, 0.005))),
            price,
            np.random.normal(1000000, 200000)
        ])
    
    engine = CryptoScoringEngine()
    scores = engine.score_coin('TEST/USD', test_data)
    
    print(f"\n📊 Test Results for TEST/USD:")
    print(f"  Vol-Edge: {scores['vol_edge']:.1f}")
    print(f"  Quality: {scores['quality']:.1f}")
    print(f"  Setup: {scores['setup']:.1f}")
    print(f"  Regime: {scores['regime']:.1f}")
    print(f"  Composite: {scores['composite']:.1f}")
    print(f"  Signal: {scores['signal_strength']}")

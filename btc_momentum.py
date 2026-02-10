"""
BTC Momentum Scanner — The Signal Engine
=========================================
Bitcoin is the signal, not the trade.
This module continuously monitors BTC's momentum across multiple timeframes
and outputs a single score (0-100) that tells the rest of the system:
"How hot is Bitcoin right now?"

When this score is high → alts are amplifying → GO
When this score drops  → alts will crash harder → EXIT
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("btc_momentum")


@dataclass
class MomentumSignal:
    """Output of the momentum scanner."""
    timestamp: datetime
    score: float                  # 0-100 composite momentum score
    regime: str                   # "dead", "cool", "warm", "hot", "fire"
    btc_price: float
    btc_change_1h: float          # % change
    btc_change_4h: float
    btc_change_24h: float
    volume_ratio: float           # current vol / avg vol
    rsi_1h: float
    rsi_4h: float
    trend_alignment: float        # how many timeframes agree on direction
    details: dict = field(default_factory=dict)

    @property
    def is_go(self) -> bool:
        """Should we be trading alts right now?"""
        return self.score >= 55

    @property
    def is_fire(self) -> bool:
        """Bitcoin is absolutely ripping — max alt exposure."""
        return self.score >= 80

    @property
    def is_exit(self) -> bool:
        """Bitcoin momentum dying — get out of alts."""
        return self.score < 30


class BTCMomentumScanner:
    """
    Watches Bitcoin across multiple timeframes and produces a single
    momentum score that drives the entire alt trading system.
    """

    def __init__(self, exchange_id: str = "binance", config: dict = None):
        self.exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
        })
        self.config = config or {}
        self.timeframes = ["5m", "15m", "1h", "4h", "1d"]
        self.symbol = "BTC/USDT"
        self._cache = {}

    def fetch_ohlcv(self, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV candles for BTC."""
        try:
            raw = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {timeframe} OHLCV: {e}")
            return pd.DataFrame()

    @staticmethod
    def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calc_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calc_macd(series: pd.Series) -> tuple:
        """Calculate MACD line, signal, histogram."""
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal
        return macd_line, signal, histogram

    def score_price_momentum(self, df: pd.DataFrame) -> float:
        """Score based on recent price changes (0-25 points)."""
        if df.empty or len(df) < 2:
            return 0.0

        close = df["close"]
        # Recent returns
        ret_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(df) > 5 else 0
        ret_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(df) > 20 else 0

        score = 0.0
        # Positive short-term momentum
        if ret_5 > 0:
            score += min(ret_5 * 2, 10)  # Up to 10 points for 5% gain
        # Positive medium-term momentum
        if ret_20 > 0:
            score += min(ret_20, 10)  # Up to 10 points
        # Acceleration (short > long term = accelerating)
        if ret_5 > ret_20 / 4 and ret_5 > 0:
            score += 5

        return min(score, 25.0)

    def score_rsi(self, df: pd.DataFrame) -> float:
        """Score based on RSI positioning (0-20 points).
        We want RSI showing bullish momentum but not overbought.
        Sweet spot: 55-75 = maximum score."""
        if df.empty or len(df) < 15:
            return 0.0

        rsi = self.calc_rsi(df["close"]).iloc[-1]
        if pd.isna(rsi):
            return 0.0

        if 55 <= rsi <= 75:
            return 20.0  # Sweet spot — strong momentum, not overbought
        elif 45 <= rsi < 55:
            return 12.0  # Building momentum
        elif 75 < rsi <= 85:
            return 14.0  # Very strong but caution
        elif 85 < rsi:
            return 6.0   # Overbought — momentum may fade
        elif 35 <= rsi < 45:
            return 5.0   # Weak
        else:
            return 0.0   # Bearish

    def score_volume(self, df: pd.DataFrame) -> float:
        """Score based on volume confirmation (0-20 points).
        High volume on up moves = real momentum, not fake."""
        if df.empty or len(df) < 21:
            return 0.0

        vol = df["volume"]
        close = df["close"]

        avg_vol = vol.rolling(20).mean().iloc[-1]
        current_vol = vol.iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        price_up = close.iloc[-1] > close.iloc[-2]

        score = 0.0
        if price_up and vol_ratio > 1.5:
            score = min(vol_ratio * 5, 20)  # High volume + price up = conviction
        elif price_up and vol_ratio > 1.0:
            score = 10.0  # Normal volume up move
        elif not price_up and vol_ratio > 2.0:
            score = 2.0   # Selling pressure — bad
        else:
            score = 5.0   # Neutral

        return min(score, 20.0)

    def score_trend_alignment(self, candles_by_tf: dict) -> float:
        """Score based on multi-timeframe trend alignment (0-20 points).
        When ALL timeframes agree BTC is trending up = maximum signal."""

        bullish_count = 0
        total = 0

        for tf, df in candles_by_tf.items():
            if df.empty or len(df) < 50:
                continue
            total += 1

            close = df["close"]
            ema20 = self.calc_ema(close, 20).iloc[-1]
            ema50 = self.calc_ema(close, 50).iloc[-1]

            # Bullish: price above both EMAs and short EMA above long EMA
            if close.iloc[-1] > ema20 and ema20 > ema50:
                bullish_count += 1

        if total == 0:
            return 0.0

        alignment = bullish_count / total
        return alignment * 20.0

    def score_macd(self, df: pd.DataFrame) -> float:
        """Score based on MACD momentum (0-15 points)."""
        if df.empty or len(df) < 35:
            return 0.0

        macd_line, signal, histogram = self.calc_macd(df["close"])

        score = 0.0
        # MACD above signal = bullish
        if macd_line.iloc[-1] > signal.iloc[-1]:
            score += 7
        # Histogram increasing = momentum accelerating
        if len(histogram) > 1 and histogram.iloc[-1] > histogram.iloc[-2]:
            score += 5
        # MACD above zero = overall bullish
        if macd_line.iloc[-1] > 0:
            score += 3

        return min(score, 15.0)

    def get_momentum_signal(self) -> MomentumSignal:
        """
        Main method: Fetch data, calculate all scores, produce final signal.
        """
        logger.info("Scanning BTC momentum...")

        # Fetch candles for all timeframes
        candles = {}
        for tf in self.timeframes:
            candles[tf] = self.fetch_ohlcv(tf, limit=200)

        # Use 1h as primary timeframe for most calculations
        df_1h = candles.get("1h", pd.DataFrame())
        df_4h = candles.get("4h", pd.DataFrame())
        df_1d = candles.get("1d", pd.DataFrame())

        if df_1h.empty:
            logger.error("No 1h data available")
            return MomentumSignal(
                timestamp=datetime.utcnow(), score=0, regime="dead",
                btc_price=0, btc_change_1h=0, btc_change_4h=0,
                btc_change_24h=0, volume_ratio=0, rsi_1h=0, rsi_4h=0,
                trend_alignment=0
            )

        # Calculate individual scores
        price_score = self.score_price_momentum(df_1h)
        rsi_score = self.score_rsi(df_1h)
        volume_score = self.score_volume(df_1h)
        trend_score = self.score_trend_alignment(candles)
        macd_score = self.score_macd(df_1h)

        # Composite score (0-100)
        total_score = price_score + rsi_score + volume_score + trend_score + macd_score

        # Determine regime
        if total_score >= 80:
            regime = "fire"
        elif total_score >= 65:
            regime = "hot"
        elif total_score >= 45:
            regime = "warm"
        elif total_score >= 25:
            regime = "cool"
        else:
            regime = "dead"

        # Calculate supplementary metrics
        btc_price = df_1h["close"].iloc[-1] if not df_1h.empty else 0
        btc_1h = ((df_1h["close"].iloc[-1] / df_1h["close"].iloc[-2]) - 1) * 100 if len(df_1h) > 1 else 0
        btc_4h = ((df_4h["close"].iloc[-1] / df_4h["close"].iloc[-2]) - 1) * 100 if not df_4h.empty and len(df_4h) > 1 else 0
        btc_24h = ((df_1d["close"].iloc[-1] / df_1d["close"].iloc[-2]) - 1) * 100 if not df_1d.empty and len(df_1d) > 1 else 0

        rsi_1h = self.calc_rsi(df_1h["close"]).iloc[-1] if not df_1h.empty else 0
        rsi_4h = self.calc_rsi(df_4h["close"]).iloc[-1] if not df_4h.empty else 0

        vol_avg = df_1h["volume"].rolling(20).mean().iloc[-1] if len(df_1h) > 20 else 1
        vol_ratio = df_1h["volume"].iloc[-1] / vol_avg if vol_avg > 0 else 1

        signal = MomentumSignal(
            timestamp=datetime.utcnow(),
            score=round(total_score, 1),
            regime=regime,
            btc_price=round(btc_price, 2),
            btc_change_1h=round(btc_1h, 2),
            btc_change_4h=round(btc_4h, 2),
            btc_change_24h=round(btc_24h, 2),
            volume_ratio=round(vol_ratio, 2),
            rsi_1h=round(rsi_1h, 1) if not pd.isna(rsi_1h) else 0,
            rsi_4h=round(rsi_4h, 1) if not pd.isna(rsi_4h) else 0,
            trend_alignment=round(trend_score / 20, 2),
            details={
                "price_score": round(price_score, 1),
                "rsi_score": round(rsi_score, 1),
                "volume_score": round(volume_score, 1),
                "trend_score": round(trend_score, 1),
                "macd_score": round(macd_score, 1),
            }
        )

        logger.info(
            f"BTC Momentum: {signal.score}/100 [{signal.regime.upper()}] "
            f"Price: ${signal.btc_price:,.0f} | "
            f"1h: {signal.btc_change_1h:+.2f}% | "
            f"Vol Ratio: {signal.volume_ratio:.1f}x"
        )

        return signal


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    scanner = BTCMomentumScanner("binance")
    signal = scanner.get_momentum_signal()

    print(f"\n{'='*60}")
    print(f"  BTC MOMENTUM REPORT — {signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")
    print(f"  Score:        {signal.score}/100  [{signal.regime.upper()}]")
    print(f"  BTC Price:    ${signal.btc_price:,.2f}")
    print(f"  1h Change:    {signal.btc_change_1h:+.2f}%")
    print(f"  4h Change:    {signal.btc_change_4h:+.2f}%")
    print(f"  24h Change:   {signal.btc_change_24h:+.2f}%")
    print(f"  Volume Ratio: {signal.volume_ratio:.1f}x average")
    print(f"  RSI (1h):     {signal.rsi_1h}")
    print(f"  RSI (4h):     {signal.rsi_4h}")
    print(f"  Trend Align:  {signal.trend_alignment:.0%}")
    print(f"{'='*60}")
    print(f"  SIGNAL: {'🟢 GO — Trade alts!' if signal.is_go else '🔴 WAIT — BTC momentum too weak'}")
    if signal.is_fire:
        print(f"  🔥🔥🔥 BITCOIN IS ON FIRE — MAX ALT EXPOSURE 🔥🔥🔥")
    print(f"{'='*60}")
    print(f"\n  Score Breakdown:")
    for k, v in signal.details.items():
        bar = "█" * int(v) + "░" * (25 - int(v))
        print(f"    {k:20s}: {bar} {v}")

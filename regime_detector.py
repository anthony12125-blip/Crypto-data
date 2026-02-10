"""
Regime Detector
================
Determines the current market regime and tells the system what to do:

  BEAR       → Accumulate, collect data, train models. DO NOT trade alts.
  TRANSITION → Getting ready. Tighten scans, paper trade, build positions slowly.
  BULL       → Full activation. BTC is hot, alts are amplifying. GO GO GO.
  OVERHEATED → Take profits. Reduce exposure. The top is near.

Uses multiple signals:
  - BTC vs 200-day MA
  - Weekly trend structure (higher highs/lows)
  - Fear & Greed Index
  - BTC dominance trend
  - Volume profile
  - On-chain signals (when available)
"""

import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("regime_detector")


@dataclass
class RegimeSignal:
    timestamp: datetime
    regime: str              # "bear", "transition", "early_bull", "bull", "overheated"
    confidence: float        # 0-1
    btc_price: float
    btc_vs_200d_ma: float    # % above/below 200d MA
    weekly_trend: str        # "down", "sideways", "up"
    fear_greed: int          # 0-100
    btc_dominance: float     # %
    signals: dict            # individual signal details

    @property
    def should_trade_alts(self) -> bool:
        return self.regime in ("early_bull", "bull")

    @property
    def should_accumulate(self) -> bool:
        return self.regime in ("bear", "transition")

    @property
    def should_exit(self) -> bool:
        return self.regime == "overheated"

    @property
    def emoji(self) -> str:
        return {
            "bear": "🐻",
            "transition": "🌅",
            "early_bull": "🐂",
            "bull": "🚀",
            "overheated": "🌋",
        }.get(self.regime, "❓")


class RegimeDetector:

    def __init__(self, exchange_id: str = "binance"):
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.symbol = "BTC/USDT"

    def fetch_daily_candles(self, limit: int = 250) -> pd.DataFrame:
        try:
            raw = self.exchange.fetch_ohlcv(self.symbol, "1d", limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch daily candles: {e}")
            return pd.DataFrame()

    def fetch_weekly_candles(self, limit: int = 52) -> pd.DataFrame:
        try:
            raw = self.exchange.fetch_ohlcv(self.symbol, "1w", limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch weekly candles: {e}")
            return pd.DataFrame()

    def get_fear_greed(self) -> int:
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            return int(resp.json()["data"][0]["value"])
        except:
            return 50  # Neutral default

    def analyze_200d_ma(self, df: pd.DataFrame) -> dict:
        """BTC position relative to 200-day moving average."""
        if len(df) < 200:
            return {"above_200d": False, "pct_diff": 0, "score": 0}

        ma200 = df["close"].rolling(200).mean().iloc[-1]
        price = df["close"].iloc[-1]
        pct_diff = ((price / ma200) - 1) * 100

        # Score: -50 to +50
        if pct_diff > 30:
            score = 45  # Well above — strong bull
        elif pct_diff > 10:
            score = 35
        elif pct_diff > 0:
            score = 20  # Just above — early bull or transition
        elif pct_diff > -10:
            score = -10  # Just below — possible transition
        elif pct_diff > -30:
            score = -30
        else:
            score = -45  # Deep below — deep bear

        return {
            "above_200d": pct_diff > 0,
            "pct_diff": round(pct_diff, 1),
            "ma200": round(ma200, 0),
            "score": score
        }

    def analyze_weekly_trend(self, df: pd.DataFrame) -> dict:
        """Weekly higher highs / higher lows structure."""
        if len(df) < 8:
            return {"trend": "unknown", "score": 0}

        highs = df["high"].iloc[-8:]
        lows = df["low"].iloc[-8:]

        # Count higher highs and higher lows
        hh = sum(1 for i in range(1, len(highs)) if highs.iloc[i] > highs.iloc[i-1])
        hl = sum(1 for i in range(1, len(lows)) if lows.iloc[i] > lows.iloc[i-1])

        # Count lower highs and lower lows
        lh = sum(1 for i in range(1, len(highs)) if highs.iloc[i] < highs.iloc[i-1])
        ll = sum(1 for i in range(1, len(lows)) if lows.iloc[i] < lows.iloc[i-1])

        bullish_count = hh + hl
        bearish_count = lh + ll
        total = bullish_count + bearish_count

        if total == 0:
            trend = "sideways"
            score = 0
        elif bullish_count / total >= 0.65:
            trend = "up"
            score = 30
        elif bearish_count / total >= 0.65:
            trend = "down"
            score = -30
        else:
            trend = "sideways"
            score = 0

        return {
            "trend": trend,
            "higher_highs": hh,
            "higher_lows": hl,
            "lower_highs": lh,
            "lower_lows": ll,
            "score": score
        }

    def analyze_fear_greed(self, fg: int) -> dict:
        """Fear & Greed positioning."""
        if fg >= 80:
            return {"zone": "extreme_greed", "score": -15, "value": fg}  # Overheated signal
        elif fg >= 60:
            return {"zone": "greed", "score": 10, "value": fg}  # Bull confirmation
        elif fg >= 40:
            return {"zone": "neutral", "score": 0, "value": fg}
        elif fg >= 20:
            return {"zone": "fear", "score": -10, "value": fg}  # Bear
        else:
            return {"zone": "extreme_fear", "score": -20, "value": fg}  # Deep bear but opportunity

    def analyze_volume_trend(self, df: pd.DataFrame) -> dict:
        """Is volume expanding (bullish) or contracting (bearish)?"""
        if len(df) < 30:
            return {"trend": "unknown", "score": 0}

        recent_vol = df["volume"].iloc[-7:].mean()
        older_vol = df["volume"].iloc[-30:-7].mean()

        if older_vol == 0:
            return {"trend": "unknown", "score": 0}

        vol_ratio = recent_vol / older_vol

        if vol_ratio > 1.5:
            return {"trend": "expanding_fast", "ratio": round(vol_ratio, 2), "score": 15}
        elif vol_ratio > 1.1:
            return {"trend": "expanding", "ratio": round(vol_ratio, 2), "score": 8}
        elif vol_ratio > 0.9:
            return {"trend": "stable", "ratio": round(vol_ratio, 2), "score": 0}
        else:
            return {"trend": "contracting", "ratio": round(vol_ratio, 2), "score": -10}

    def analyze_momentum(self, df: pd.DataFrame) -> dict:
        """30-day and 90-day returns."""
        if len(df) < 90:
            return {"ret_30d": 0, "ret_90d": 0, "score": 0}

        ret_30 = ((df["close"].iloc[-1] / df["close"].iloc[-30]) - 1) * 100
        ret_90 = ((df["close"].iloc[-1] / df["close"].iloc[-90]) - 1) * 100

        score = 0
        if ret_30 > 20:
            score += 15
        elif ret_30 > 5:
            score += 8
        elif ret_30 > 0:
            score += 3
        elif ret_30 > -10:
            score -= 5
        else:
            score -= 15

        if ret_90 > 50:
            score += 10
        elif ret_90 > 20:
            score += 5
        elif ret_90 < -20:
            score -= 10

        return {
            "ret_30d": round(ret_30, 1),
            "ret_90d": round(ret_90, 1),
            "score": score
        }

    def detect(self) -> RegimeSignal:
        """
        MAIN METHOD: Analyze all signals and determine current market regime.
        """
        logger.info("Detecting market regime...")

        # Fetch data
        daily = self.fetch_daily_candles(250)
        weekly = self.fetch_weekly_candles(52)
        fear_greed = self.get_fear_greed()

        if daily.empty:
            return RegimeSignal(
                timestamp=datetime.utcnow(), regime="unknown", confidence=0,
                btc_price=0, btc_vs_200d_ma=0, weekly_trend="unknown",
                fear_greed=50, btc_dominance=0, signals={}
            )

        # Individual analyses
        ma_analysis = self.analyze_200d_ma(daily)
        trend_analysis = self.analyze_weekly_trend(weekly)
        fg_analysis = self.analyze_fear_greed(fear_greed)
        vol_analysis = self.analyze_volume_trend(daily)
        mom_analysis = self.analyze_momentum(daily)

        # Composite score: sum of all individual scores
        # Range roughly -160 to +160
        total_score = (
            ma_analysis["score"] +
            trend_analysis["score"] +
            fg_analysis["score"] +
            vol_analysis["score"] +
            mom_analysis["score"]
        )

        # Classify regime
        if total_score >= 70:
            regime = "bull"
            confidence = min(total_score / 100, 1.0)
        elif total_score >= 40:
            regime = "early_bull"
            confidence = (total_score - 40) / 40
        elif total_score >= 0:
            regime = "transition"
            confidence = 0.5
        elif total_score >= -40:
            regime = "bear"
            confidence = abs(total_score) / 50
        else:
            regime = "bear"
            confidence = min(abs(total_score) / 80, 1.0)

        # Override: extreme greed + high score = overheated
        if fear_greed >= 85 and total_score > 50:
            regime = "overheated"
            confidence = 0.8

        btc_price = daily["close"].iloc[-1]

        signal = RegimeSignal(
            timestamp=datetime.utcnow(),
            regime=regime,
            confidence=round(confidence, 2),
            btc_price=round(btc_price, 2),
            btc_vs_200d_ma=ma_analysis["pct_diff"],
            weekly_trend=trend_analysis["trend"],
            fear_greed=fear_greed,
            btc_dominance=0,  # Would need additional API call
            signals={
                "200d_ma": ma_analysis,
                "weekly_trend": trend_analysis,
                "fear_greed": fg_analysis,
                "volume": vol_analysis,
                "momentum": mom_analysis,
                "total_score": total_score,
            }
        )

        logger.info(
            f"Regime: {signal.emoji} {signal.regime.upper()} "
            f"(confidence: {signal.confidence:.0%}) | "
            f"BTC: ${signal.btc_price:,.0f} | "
            f"vs 200d MA: {signal.btc_vs_200d_ma:+.1f}% | "
            f"F&G: {signal.fear_greed}"
        )

        return signal


def print_regime_report(signal: RegimeSignal):
    """Pretty print regime analysis."""
    print(f"""
{'='*65}
  MARKET REGIME REPORT — {signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}
{'='*65}

  {signal.emoji}  REGIME: {signal.regime.upper()}
  Confidence: {signal.confidence:.0%}

  BTC Price:       ${signal.btc_price:,.2f}
  vs 200d MA:      {signal.btc_vs_200d_ma:+.1f}%
  Weekly Trend:    {signal.weekly_trend}
  Fear & Greed:    {signal.fear_greed}/100

{'─'*65}
  SIGNAL BREAKDOWN:
{'─'*65}""")

    for name, data in signal.signals.items():
        if isinstance(data, dict):
            score = data.get("score", 0)
            bar_pos = max(0, score)
            bar_neg = max(0, -score)
            bar = f"{'█' * int(bar_pos)}{'░' * (30 - int(bar_pos))}" if score >= 0 else \
                  f"{'░' * (30 - int(bar_neg))}{'█' * int(bar_neg)}"
            print(f"  {name:20s}: {score:+6.0f}  {bar}")
        else:
            print(f"  {name:20s}: {data}")

    total = signal.signals.get("total_score", 0)
    print(f"\n  {'TOTAL SCORE':20s}: {total:+6.0f}")

    print(f"""
{'─'*65}
  ACTION:
{'─'*65}""")

    if signal.should_trade_alts:
        print("  🟢 TRADE ALTS — Bitcoin momentum supports alt amplification")
        print("     Activate the alt beta scanner and trading engine")
    elif signal.should_accumulate:
        print("  🔵 ACCUMULATE — Bear market, collect data and DCA")
        print("     Keep data collector running, train models, paper trade")
    elif signal.should_exit:
        print("  🔴 EXIT — Market overheated, take profits systematically")
        print("     Reduce positions, tighten stops, move to stables")
    else:
        print("  ⚪ HOLD — Uncertain regime, maintain current positions")

    print(f"{'='*65}\n")


if __name__ == "__main__":
    detector = RegimeDetector("binance")
    signal = detector.detect()
    print_regime_report(signal)

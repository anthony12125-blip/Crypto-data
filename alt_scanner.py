"""
Alt Beta Amplification Scanner
================================
This is the money-maker. 

When Bitcoin moves 10%, there are alts moving 100-1000% on the SAME DAY.
This module scans hundreds of alts in real-time and finds the ones that are
amplifying Bitcoin's move the most — ranked by beta, volume surge, and
relative strength.

The thesis: Don't trade Bitcoin. Use Bitcoin as the signal.
Trade the alts that are riding the same wave but 10-50x harder.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alt_scanner")


@dataclass
class AltSignal:
    """A single alt coin's amplification data."""
    symbol: str
    price: float
    beta_vs_btc: float            # How many x it moves relative to BTC
    change_1h: float              # % change last hour
    change_4h: float              # % change last 4h
    change_24h: float             # % change last 24h
    btc_change_24h: float         # BTC's change for comparison
    amplification_ratio: float    # alt_change / btc_change (the key metric)
    volume_surge: float           # current vol / avg vol
    volume_usd_24h: float         # absolute volume
    correlation_btc: float        # correlation with BTC (0-1)
    rsi: float
    relative_strength_vs_btc: float  # outperformance vs BTC
    score: float                  # composite ranking score (0-100)

    @property
    def is_amplifying(self) -> bool:
        """Is this alt meaningfully amplifying BTC's move?"""
        return self.beta_vs_btc > 2.0 and self.correlation_btc > 0.4

    @property
    def grade(self) -> str:
        if self.score >= 85:
            return "🔥 S-TIER"
        elif self.score >= 70:
            return "🟢 A-TIER"
        elif self.score >= 55:
            return "🟡 B-TIER"
        elif self.score >= 40:
            return "🟠 C-TIER"
        else:
            return "⚪ SKIP"


class AltBetaScanner:
    """
    Scans the entire alt universe and ranks coins by how much they're
    amplifying Bitcoin's momentum right now.
    """

    def __init__(self, exchange_id: str = "binance", config: dict = None):
        self.exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
        })
        self.config = config or {}
        self.quote = self.config.get("quote_currency", "USDT")
        self.min_volume = self.config.get("min_volume_usd", 1_000_000)
        self.exclude = set(self.config.get("exclude", [
            "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD",
            "WBTC", "WETH", "stETH", "WBETH"
        ]))

    def get_tradeable_alts(self, top_n: int = 200) -> List[str]:
        """Get the top N alts by volume, excluding stables and BTC."""
        logger.info(f"Fetching tradeable alts (top {top_n})...")
        try:
            tickers = self.exchange.fetch_tickers()
        except Exception as e:
            logger.error(f"Failed to fetch tickers: {e}")
            return []

        alt_volumes = []
        for symbol, ticker in tickers.items():
            if not symbol.endswith(f"/{self.quote}"):
                continue
            base = symbol.split("/")[0]
            if base in self.exclude or base == "BTC":
                continue

            vol_usd = (ticker.get("quoteVolume") or 0)
            if vol_usd >= self.min_volume:
                alt_volumes.append((symbol, vol_usd))

        # Sort by volume descending, take top N
        alt_volumes.sort(key=lambda x: x[1], reverse=True)
        symbols = [s for s, v in alt_volumes[:top_n]]
        logger.info(f"Found {len(symbols)} tradeable alts")
        return symbols

    def fetch_btc_data(self, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetch BTC candles."""
        try:
            raw = self.exchange.fetch_ohlcv(f"BTC/{self.quote}", timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df["returns"] = df["close"].pct_change()
            return df
        except Exception as e:
            logger.error(f"Failed to fetch BTC data: {e}")
            return pd.DataFrame()

    def fetch_alt_data(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetch alt candles."""
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df["returns"] = df["close"].pct_change()
            return df
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def calc_beta(alt_returns: pd.Series, btc_returns: pd.Series) -> float:
        """
        Calculate beta of alt relative to BTC.
        Beta > 1 means alt moves MORE than BTC.
        Beta of 5 means when BTC moves 1%, alt moves 5%.
        THIS IS THE KEY METRIC.
        """
        # Align the series
        combined = pd.concat([alt_returns, btc_returns], axis=1).dropna()
        if len(combined) < 10:
            return 0.0

        alt_r = combined.iloc[:, 0]
        btc_r = combined.iloc[:, 1]

        covariance = alt_r.cov(btc_r)
        btc_variance = btc_r.var()

        if btc_variance == 0:
            return 0.0

        return covariance / btc_variance

    @staticmethod
    def calc_correlation(alt_returns: pd.Series, btc_returns: pd.Series) -> float:
        """Correlation between alt and BTC returns."""
        combined = pd.concat([alt_returns, btc_returns], axis=1).dropna()
        if len(combined) < 10:
            return 0.0
        return combined.iloc[:, 0].corr(combined.iloc[:, 1])

    @staticmethod
    def calc_rsi(series: pd.Series, period: int = 14) -> float:
        """Calculate current RSI value."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return round(val, 1) if not pd.isna(val) else 50.0

    def score_alt(self, beta: float, volume_surge: float, correlation: float,
                  amplification: float, rsi: float, change_24h: float) -> float:
        """
        Composite scoring of an alt's tradability right now.
        Weights what matters most for the strategy:
        1. Beta (how much it amplifies BTC) — 30%
        2. Volume surge (is real money flowing in) — 25%
        3. Amplification ratio (actual current outperformance) — 25%
        4. Correlation (is it moving WITH btc, not independently) — 10%
        5. RSI positioning (momentum sweet spot) — 10%
        """
        score = 0.0

        # Beta score (0-30) — sweet spot is 3-10x
        if 3 <= beta <= 10:
            score += 30
        elif 2 <= beta < 3:
            score += 20
        elif 10 < beta <= 20:
            score += 22  # Very high beta, more risk but huge potential
        elif beta > 20:
            score += 10  # Extremely volatile, risky
        elif 1 < beta < 2:
            score += 8   # Low amplification, why bother
        else:
            score += 0   # Not amplifying or inverse

        # Volume surge score (0-25) — need real money behind the move
        if volume_surge >= 5.0:
            score += 25
        elif volume_surge >= 3.0:
            score += 20
        elif volume_surge >= 2.0:
            score += 15
        elif volume_surge >= 1.5:
            score += 10
        elif volume_surge >= 1.0:
            score += 5
        else:
            score += 0

        # Amplification ratio score (0-25) — how much it's actually outperforming right now
        if amplification >= 10:
            score += 25
        elif amplification >= 5:
            score += 20
        elif amplification >= 3:
            score += 15
        elif amplification >= 2:
            score += 10
        elif amplification >= 1.5:
            score += 5

        # Correlation score (0-10) — we want it correlated, not random
        if 0.5 <= correlation <= 0.85:
            score += 10  # Sweet spot: moves with BTC but amplified
        elif 0.3 <= correlation < 0.5:
            score += 5
        elif correlation > 0.85:
            score += 7   # Too correlated, might not amplify much
        else:
            score += 0   # Uncorrelated = unreliable

        # RSI score (0-10) — we want momentum, not overbought
        if 50 <= rsi <= 70:
            score += 10  # Building momentum
        elif 70 < rsi <= 80:
            score += 7   # Strong but watch out
        elif 40 <= rsi < 50:
            score += 5   # Starting to turn
        elif rsi > 80:
            score += 3   # Overbought risk
        else:
            score += 0   # Weak

        return min(score, 100.0)

    def scan(self, btc_momentum_score: float = 50, top_n_alts: int = 200,
             timeframe: str = "1h", lookback: int = 48) -> List[AltSignal]:
        """
        MAIN METHOD: Scan the alt universe and rank by beta amplification.

        Args:
            btc_momentum_score: Current BTC momentum (from BTCMomentumScanner)
            top_n_alts: How many alts to scan
            timeframe: Candle timeframe for beta calculation
            lookback: Number of candles for rolling beta

        Returns:
            List of AltSignal, sorted by score descending
        """
        logger.info(f"Starting alt scan (BTC momentum: {btc_momentum_score})...")

        # Get BTC data first
        btc_df = self.fetch_btc_data(timeframe, limit=lookback + 50)
        if btc_df.empty:
            logger.error("No BTC data — aborting scan")
            return []

        btc_change_24h = 0
        if len(btc_df) >= 24:
            btc_change_24h = ((btc_df["close"].iloc[-1] / btc_df["close"].iloc[-25]) - 1) * 100

        # Get tradeable alts
        alt_symbols = self.get_tradeable_alts(top_n_alts)

        signals = []
        scanned = 0
        errors = 0

        for symbol in alt_symbols:
            try:
                alt_df = self.fetch_alt_data(symbol, timeframe, limit=lookback + 50)
                if alt_df.empty or len(alt_df) < 30:
                    continue

                # Core calculations
                beta = self.calc_beta(alt_df["returns"], btc_df["returns"])
                correlation = self.calc_correlation(alt_df["returns"], btc_df["returns"])

                # Price changes
                change_1h = ((alt_df["close"].iloc[-1] / alt_df["close"].iloc[-2]) - 1) * 100 if len(alt_df) > 1 else 0
                change_4h = ((alt_df["close"].iloc[-1] / alt_df["close"].iloc[-5]) - 1) * 100 if len(alt_df) > 4 else 0
                change_24h = ((alt_df["close"].iloc[-1] / alt_df["close"].iloc[-25]) - 1) * 100 if len(alt_df) > 24 else 0

                # Amplification ratio — THE KEY NUMBER
                # If BTC is up 5% and this alt is up 50%, amplification = 10x
                if btc_change_24h != 0:
                    amplification = abs(change_24h / btc_change_24h)
                else:
                    amplification = abs(change_24h) if change_24h > 0 else 0

                # Volume analysis
                vol_avg = alt_df["volume"].rolling(20).mean().iloc[-1]
                vol_current = alt_df["volume"].iloc[-1]
                vol_surge = vol_current / vol_avg if vol_avg > 0 else 1.0
                vol_usd = vol_current * alt_df["close"].iloc[-1]

                # RSI
                rsi = self.calc_rsi(alt_df["close"])

                # Composite score
                score = self.score_alt(beta, vol_surge, correlation,
                                       amplification, rsi, change_24h)

                # Only include coins moving in same direction as BTC
                if btc_change_24h > 0 and change_24h < 0:
                    score *= 0.3  # Penalize if moving opposite to BTC

                signal = AltSignal(
                    symbol=symbol,
                    price=round(alt_df["close"].iloc[-1], 8),
                    beta_vs_btc=round(beta, 2),
                    change_1h=round(change_1h, 2),
                    change_4h=round(change_4h, 2),
                    change_24h=round(change_24h, 2),
                    btc_change_24h=round(btc_change_24h, 2),
                    amplification_ratio=round(amplification, 1),
                    volume_surge=round(vol_surge, 1),
                    volume_usd_24h=round(vol_usd, 0),
                    correlation_btc=round(correlation, 2),
                    rsi=rsi,
                    relative_strength_vs_btc=round(change_24h - btc_change_24h, 2),
                    score=round(score, 1),
                )
                signals.append(signal)
                scanned += 1

                # Rate limit respect
                if scanned % 10 == 0:
                    logger.info(f"  Scanned {scanned}/{len(alt_symbols)} alts...")
                    time.sleep(0.5)

            except Exception as e:
                errors += 1
                logger.debug(f"Error scanning {symbol}: {e}")
                continue

        # Sort by score
        signals.sort(key=lambda s: s.score, reverse=True)
        logger.info(f"Scan complete: {scanned} alts scanned, {errors} errors, {len(signals)} signals")

        return signals


def print_scan_results(signals: List[AltSignal], top_n: int = 20):
    """Pretty print the top alt signals."""
    print(f"\n{'='*100}")
    print(f"  ALT BETA AMPLIFICATION SCAN — Top {top_n} Coins")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*100}")
    print(f"  {'Rank':>4} {'Symbol':<14} {'Score':>6} {'Grade':<12} {'Beta':>6} "
          f"{'Amp':>6} {'24h%':>8} {'BTC%':>7} {'VolSurge':>9} {'RSI':>5} {'Corr':>6}")
    print(f"  {'-'*94}")

    for i, s in enumerate(signals[:top_n], 1):
        print(
            f"  {i:>4} {s.symbol:<14} {s.score:>6.1f} {s.grade:<12} "
            f"{s.beta_vs_btc:>6.1f}x {s.amplification_ratio:>5.1f}x "
            f"{s.change_24h:>+7.1f}% {s.btc_change_24h:>+6.1f}% "
            f"{s.volume_surge:>8.1f}x {s.rsi:>5.0f} {s.correlation_btc:>5.2f}"
        )

    print(f"{'='*100}")

    # Summary stats
    amplifying = [s for s in signals if s.is_amplifying]
    print(f"\n  Summary:")
    print(f"    Total alts scanned:        {len(signals)}")
    print(f"    Actively amplifying BTC:   {len(amplifying)}")
    if signals:
        avg_beta = np.mean([s.beta_vs_btc for s in signals[:20]])
        avg_amp = np.mean([s.amplification_ratio for s in signals[:20]])
        print(f"    Avg beta (top 20):         {avg_beta:.1f}x")
        print(f"    Avg amplification (top 20): {avg_amp:.1f}x")
    print()


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    scanner = AltBetaScanner("binance")
    signals = scanner.scan(
        btc_momentum_score=50,
        top_n_alts=50,     # Start small for testing
        timeframe="1h",
        lookback=48
    )
    print_scan_results(signals, top_n=20)

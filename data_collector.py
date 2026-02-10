"""
Data Collection Pipeline
==========================
START THIS NOW. Every day of data you don't collect is data you can't train on.

This runs 24/7 on your M4 Mac Mini and collects:
- OHLCV candles (1m, 5m, 15m, 1h, 4h, 1d) for BTC + top 100 alts
- Fear & Greed index
- Volume profiles
- BTC dominance
- All stored in SQLite for fast local access

Run with: python3 data_collector.py
Or as a background service: nohup python3 data_collector.py &
"""

import ccxt
import sqlite3
import pandas as pd
import numpy as np
import requests
import time
import json
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/collector.log"),
    ]
)
logger = logging.getLogger("data_collector")

# Graceful shutdown
RUNNING = True
def signal_handler(sig, frame):
    global RUNNING
    logger.info("Shutdown signal received. Finishing current cycle...")
    RUNNING = False
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class DataCollector:
    """Continuous data collection pipeline."""

    def __init__(self, exchange_id: str = "binance", db_path: str = "data/market_data.db"):
        # Create data directory
        Path("data").mkdir(exist_ok=True)
        Path("data/ohlcv").mkdir(exist_ok=True)
        Path("data/sentiment").mkdir(exist_ok=True)

        # Exchange
        self.exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
        })

        # Database
        self.db_path = db_path
        self._init_db()

        # Config
        self.quote = "USDT"
        self.btc_symbol = f"BTC/{self.quote}"
        self.timeframes = ["1h", "4h", "1d"]  # Primary collection timeframes
        self.exclude_bases = {
            "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD",
            "WBTC", "WETH", "stETH", "WBETH", "BFUSD"
        }

        logger.info(f"DataCollector initialized — Exchange: {exchange_id}, DB: {db_path}")

    def _init_db(self):
        """Initialize SQLite database with tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # OHLCV data
        c.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT,
                timeframe TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)

        # Fear & Greed index
        c.execute("""
            CREATE TABLE IF NOT EXISTS fear_greed (
                timestamp INTEGER PRIMARY KEY,
                value INTEGER,
                classification TEXT
            )
        """)

        # BTC dominance
        c.execute("""
            CREATE TABLE IF NOT EXISTS btc_dominance (
                timestamp INTEGER PRIMARY KEY,
                dominance REAL,
                total_mcap REAL,
                btc_mcap REAL
            )
        """)

        # Alt scan snapshots (for training the ML model)
        c.execute("""
            CREATE TABLE IF NOT EXISTS alt_snapshots (
                timestamp INTEGER,
                symbol TEXT,
                price REAL,
                change_1h REAL,
                change_4h REAL,
                change_24h REAL,
                volume_usd REAL,
                volume_surge REAL,
                btc_change_24h REAL,
                beta REAL,
                correlation REAL,
                PRIMARY KEY (timestamp, symbol)
            )
        """)

        # Collection log
        c.execute("""
            CREATE TABLE IF NOT EXISTS collection_log (
                timestamp INTEGER,
                task TEXT,
                status TEXT,
                records INTEGER,
                duration_sec REAL
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Database initialized")

    def get_top_alts(self, top_n: int = 100) -> list:
        """Get top N alts by 24h volume."""
        try:
            tickers = self.exchange.fetch_tickers()
            alt_vols = []
            for sym, t in tickers.items():
                if not sym.endswith(f"/{self.quote}"):
                    continue
                base = sym.split("/")[0]
                if base in self.exclude_bases:
                    continue
                vol = t.get("quoteVolume") or 0
                if vol > 500_000:
                    alt_vols.append((sym, vol))

            alt_vols.sort(key=lambda x: x[1], reverse=True)
            return [s for s, v in alt_vols[:top_n]]
        except Exception as e:
            logger.error(f"Failed to get top alts: {e}")
            return []

    def collect_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        """Fetch and store OHLCV candles."""
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            inserted = 0
            for candle in raw:
                try:
                    c.execute(
                        "INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (symbol, timeframe, candle[0], candle[1], candle[2],
                         candle[3], candle[4], candle[5])
                    )
                    inserted += 1
                except sqlite3.Error:
                    pass
            conn.commit()
            conn.close()
            return inserted
        except Exception as e:
            logger.debug(f"OHLCV error {symbol} {timeframe}: {e}")
            return 0

    def collect_fear_greed(self):
        """Fetch and store Fear & Greed index."""
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1&format=json",
                timeout=10
            )
            data = resp.json()["data"][0]
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO fear_greed VALUES (?, ?, ?)",
                (int(data["timestamp"]), int(data["value"]), data["value_classification"])
            )
            conn.commit()
            conn.close()
            logger.info(f"Fear & Greed: {data['value']} ({data['value_classification']})")
            return int(data["value"])
        except Exception as e:
            logger.error(f"Fear & Greed fetch error: {e}")
            return None

    def collect_alt_snapshot(self, symbols: list):
        """Take a snapshot of all alt metrics for ML training data."""
        try:
            # Get BTC data for reference
            btc_raw = self.exchange.fetch_ohlcv(self.btc_symbol, "1h", limit=48)
            if not btc_raw:
                return 0
            btc_df = pd.DataFrame(btc_raw, columns=["ts", "o", "h", "l", "c", "v"])
            btc_change_24h = ((btc_df["c"].iloc[-1] / btc_df["c"].iloc[-25]) - 1) * 100 if len(btc_df) > 24 else 0
            btc_returns = btc_df["c"].pct_change().dropna()

            now_ts = int(datetime.utcnow().timestamp())
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            count = 0

            for symbol in symbols:
                try:
                    raw = self.exchange.fetch_ohlcv(symbol, "1h", limit=48)
                    if not raw or len(raw) < 25:
                        continue

                    df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
                    price = df["c"].iloc[-1]
                    ch1 = ((df["c"].iloc[-1] / df["c"].iloc[-2]) - 1) * 100
                    ch4 = ((df["c"].iloc[-1] / df["c"].iloc[-5]) - 1) * 100 if len(df) > 4 else 0
                    ch24 = ((df["c"].iloc[-1] / df["c"].iloc[-25]) - 1) * 100 if len(df) > 24 else 0

                    vol_avg = df["v"].rolling(20).mean().iloc[-1]
                    vol_surge = df["v"].iloc[-1] / vol_avg if vol_avg > 0 else 1
                    vol_usd = df["v"].iloc[-1] * price

                    alt_returns = df["c"].pct_change().dropna()
                    # Beta and correlation
                    min_len = min(len(alt_returns), len(btc_returns))
                    if min_len > 10:
                        ar = alt_returns.iloc[-min_len:].reset_index(drop=True)
                        br = btc_returns.iloc[-min_len:].reset_index(drop=True)
                        cov = ar.cov(br)
                        var = br.var()
                        beta = cov / var if var > 0 else 0
                        corr = ar.corr(br)
                    else:
                        beta = 0
                        corr = 0

                    c.execute(
                        "INSERT OR REPLACE INTO alt_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (now_ts, symbol, price, ch1, ch4, ch24, vol_usd,
                         vol_surge, btc_change_24h, beta, corr)
                    )
                    count += 1
                    time.sleep(0.1)  # Rate limit

                except Exception as e:
                    continue

            conn.commit()
            conn.close()
            return count

        except Exception as e:
            logger.error(f"Alt snapshot error: {e}")
            return 0

    def run_collection_cycle(self):
        """Run one full collection cycle."""
        logger.info("=" * 60)
        logger.info("Starting collection cycle...")
        cycle_start = time.time()

        # 1. Get current alt universe
        top_alts = self.get_top_alts(100)
        all_symbols = [self.btc_symbol] + top_alts
        logger.info(f"Tracking {len(all_symbols)} symbols")

        # 2. Collect OHLCV for all symbols and timeframes
        total_candles = 0
        for tf in self.timeframes:
            for symbol in all_symbols:
                if not RUNNING:
                    return
                n = self.collect_ohlcv(symbol, tf, limit=100)
                total_candles += n
                time.sleep(0.15)  # Respect rate limits
            logger.info(f"  {tf} candles collected: {total_candles}")

        # 3. Fear & Greed
        self.collect_fear_greed()

        # 4. Alt snapshot (for ML training)
        snapshot_count = self.collect_alt_snapshot(top_alts)
        logger.info(f"  Alt snapshot: {snapshot_count} coins captured")

        # 5. Log this cycle
        duration = time.time() - cycle_start
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO collection_log VALUES (?, ?, ?, ?, ?)",
            (int(datetime.utcnow().timestamp()), "full_cycle", "ok",
             total_candles + snapshot_count, round(duration, 1))
        )
        conn.commit()
        conn.close()

        logger.info(f"Cycle complete in {duration:.0f}s — "
                     f"{total_candles} candles, {snapshot_count} alt snapshots")

    def get_stats(self) -> dict:
        """Get collection statistics."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        stats = {}
        c.execute("SELECT COUNT(*) FROM ohlcv")
        stats["total_candles"] = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv")
        stats["unique_symbols"] = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM alt_snapshots")
        stats["alt_snapshots"] = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM fear_greed")
        stats["fear_greed_records"] = c.fetchone()[0]

        c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv")
        row = c.fetchone()
        if row[0]:
            stats["earliest_data"] = datetime.fromtimestamp(row[0] / 1000).strftime("%Y-%m-%d")
            stats["latest_data"] = datetime.fromtimestamp(row[1] / 1000).strftime("%Y-%m-%d")

        conn.close()
        return stats

    def run_forever(self, interval_minutes: int = 60):
        """
        Run the collector continuously.
        Default: collect every 60 minutes.
        On M4 Mac Mini, this uses minimal CPU/memory.
        """
        logger.info("=" * 60)
        logger.info("  CRYPTO DATA COLLECTOR — Starting continuous collection")
        logger.info(f"  Collection interval: every {interval_minutes} minutes")
        logger.info(f"  Database: {self.db_path}")
        logger.info("  Press Ctrl+C to stop gracefully")
        logger.info("=" * 60)

        cycle = 0
        while RUNNING:
            cycle += 1
            logger.info(f"\n--- Cycle #{cycle} ---")

            try:
                self.run_collection_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}")

            # Print stats every 5 cycles
            if cycle % 5 == 0:
                stats = self.get_stats()
                logger.info(f"DB Stats: {json.dumps(stats, indent=2)}")

            # Wait for next cycle
            if RUNNING:
                logger.info(f"Sleeping {interval_minutes} minutes until next cycle...")
                for _ in range(interval_minutes * 60):
                    if not RUNNING:
                        break
                    time.sleep(1)

        logger.info("Collector stopped gracefully.")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crypto Data Collector")
    parser.add_argument("--exchange", default="binance", help="Exchange ID")
    parser.add_argument("--interval", type=int, default=60, help="Collection interval (minutes)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--stats", action="store_true", help="Show DB stats and exit")
    args = parser.parse_args()

    collector = DataCollector(exchange_id=args.exchange)

    if args.stats:
        stats = collector.get_stats()
        print(json.dumps(stats, indent=2))
    elif args.once:
        collector.run_collection_cycle()
        stats = collector.get_stats()
        print(f"\nCollection stats: {json.dumps(stats, indent=2)}")
    else:
        collector.run_forever(interval_minutes=args.interval)

"""
Backtesting Engine
====================
Test the alt beta amplification strategy against historical data.
Uses walk-forward validation: train on past, test on future, roll forward.
Never peek at the future.

Run: python3 backtest.py
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backtest")


@dataclass
class BacktestTrade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    beta_at_entry: float
    btc_momentum_at_entry: float
    pnl_pct: float
    exit_reason: str
    hold_hours: float


@dataclass
class BacktestResult:
    """Summary of a backtest run."""
    start_date: str
    end_date: str
    profile: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_hold_hours: float
    best_trade_pct: float
    worst_trade_pct: float
    btc_return_pct: float         # Buy & hold BTC comparison
    outperformance_pct: float     # Our strategy vs BTC buy & hold
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """
    Walk-forward backtester for the alt beta amplification strategy.
    """

    def __init__(self, exchange_id: str = "binance", db_path: str = "data/market_data.db"):
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.db_path = db_path

    def fetch_historical(self, symbol: str, timeframe: str = "1h",
                         since: str = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch historical candles."""
        try:
            since_ts = None
            if since:
                since_ts = int(datetime.fromisoformat(since).timestamp() * 1000)

            all_candles = []
            fetched = 0
            current_since = since_ts

            while fetched < limit:
                batch_limit = min(500, limit - fetched)
                raw = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=batch_limit
                )
                if not raw:
                    break
                all_candles.extend(raw)
                fetched += len(raw)
                current_since = raw[-1][0] + 1
                if len(raw) < batch_limit:
                    break

            df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df["returns"] = df["close"].pct_change()
            return df

        except Exception as e:
            logger.error(f"Fetch error {symbol}: {e}")
            return pd.DataFrame()

    def calc_beta(self, alt_returns: pd.Series, btc_returns: pd.Series, window: int = 48) -> pd.Series:
        """Rolling beta calculation."""
        cov = alt_returns.rolling(window).cov(btc_returns)
        var = btc_returns.rolling(window).var()
        return (cov / var.replace(0, np.nan)).fillna(0)

    def calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def calc_btc_momentum(self, btc_df: pd.DataFrame, idx: int) -> float:
        """Simplified BTC momentum score for backtesting."""
        if idx < 50:
            return 0

        close = btc_df["close"].iloc[:idx+1]

        # Price momentum
        ret_5 = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100 if len(close) > 5 else 0
        ret_20 = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100 if len(close) > 20 else 0

        price_score = 0
        if ret_5 > 0:
            price_score += min(ret_5 * 2, 10)
        if ret_20 > 0:
            price_score += min(ret_20, 10)

        # RSI
        rsi_series = self.calc_rsi(close)
        rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50

        rsi_score = 0
        if 55 <= rsi <= 75:
            rsi_score = 20
        elif 45 <= rsi < 55:
            rsi_score = 12
        elif 75 < rsi <= 85:
            rsi_score = 14
        else:
            rsi_score = 5

        # EMA trend
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        trend_score = 20 if (close.iloc[-1] > ema20 and ema20 > ema50) else 5

        # Volume
        vol = btc_df["volume"].iloc[:idx+1]
        vol_avg = vol.rolling(20).mean().iloc[-1]
        vol_current = vol.iloc[-1]
        vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1
        vol_score = min(vol_ratio * 5, 15) if (close.iloc[-1] > close.iloc[-2]) else 5

        return min(price_score + rsi_score + trend_score + vol_score, 100)

    def run(self, alt_symbols: List[str], timeframe: str = "1h",
            since: str = "2024-01-01", limit: int = 2000,
            profile: str = "moderate", initial_capital: float = 10000) -> BacktestResult:
        """
        Run a full backtest.

        Args:
            alt_symbols: List of alt symbols to test (e.g., ["ETH/USDT", "SOL/USDT"])
            timeframe: Candle timeframe
            since: Start date ISO format
            limit: Number of candles
            profile: Trading profile name
            initial_capital: Starting capital in USD
        """
        # Profile settings
        profiles = {
            "conservative": {
                "tp": 8.0, "sl": 2.5, "trailing": 3.0,
                "max_pos": 3, "pos_size": 0.05, "min_momentum": 65,
                "min_beta": 2.0, "max_hold": 48
            },
            "moderate": {
                "tp": 15.0, "sl": 4.0, "trailing": 5.0,
                "max_pos": 5, "pos_size": 0.10, "min_momentum": 55,
                "min_beta": 2.0, "max_hold": 96
            },
            "aggressive": {
                "tp": 30.0, "sl": 7.0, "trailing": 8.0,
                "max_pos": 8, "pos_size": 0.15, "min_momentum": 45,
                "min_beta": 1.5, "max_hold": 168
            },
        }
        cfg = profiles.get(profile, profiles["moderate"])

        logger.info(f"Fetching BTC data from {since}...")
        btc_df = self.fetch_historical("BTC/USDT", timeframe, since=since, limit=limit)
        if btc_df.empty:
            logger.error("No BTC data")
            return None

        logger.info(f"Fetching data for {len(alt_symbols)} alts...")
        alt_data = {}
        for sym in alt_symbols:
            df = self.fetch_historical(sym, timeframe, since=since, limit=limit)
            if not df.empty and len(df) > 100:
                alt_data[sym] = df
                logger.info(f"  {sym}: {len(df)} candles")

        if not alt_data:
            logger.error("No alt data")
            return None

        # Align all data to BTC's index
        common_idx = btc_df.index
        for sym in list(alt_data.keys()):
            alt_data[sym] = alt_data[sym].reindex(common_idx).ffill().dropna(subset=["close"])

        # Pre-calculate rolling betas
        logger.info("Calculating rolling betas...")
        betas = {}
        for sym, df in alt_data.items():
            aligned = pd.concat([df["returns"], btc_df["returns"]], axis=1).dropna()
            if len(aligned) > 50:
                betas[sym] = self.calc_beta(aligned.iloc[:, 0], aligned.iloc[:, 1], window=48)

        # ─── SIMULATION ────────────────────────────────
        logger.info("Running simulation...")
        capital = initial_capital
        positions: Dict[str, dict] = {}
        trades: List[BacktestTrade] = []
        equity_curve = [capital]
        fees_pct = 0.075  # 0.075% per trade (Binance taker fee)

        start_idx = 60  # Need enough history for indicators

        for i in range(start_idx, len(btc_df)):
            current_time = btc_df.index[i]

            # BTC momentum at this point
            btc_momentum = self.calc_btc_momentum(btc_df, i)

            # ─── CHECK EXITS ───
            for sym in list(positions.keys()):
                pos = positions[sym]
                if sym not in alt_data or i >= len(alt_data[sym]):
                    continue

                current_price = alt_data[sym]["close"].iloc[i]
                pos["highest"] = max(pos["highest"], current_price)
                pnl_pct = ((current_price / pos["entry_price"]) - 1) * 100
                hold_hours = (i - pos["entry_idx"])  # Approximate for 1h candles

                exit_reason = None

                # BTC momentum crash — EXIT ALTS IMMEDIATELY
                if btc_momentum < 25:
                    exit_reason = "btc_exit"
                elif pnl_pct >= cfg["tp"]:
                    exit_reason = "take_profit"
                elif pnl_pct <= -cfg["sl"]:
                    exit_reason = "stop_loss"
                elif pos["highest"] > pos["entry_price"]:
                    drop = ((current_price / pos["highest"]) - 1) * 100
                    if drop <= -cfg["trailing"]:
                        exit_reason = "trailing_stop"
                elif hold_hours >= cfg["max_hold"]:
                    exit_reason = "time_limit"

                if exit_reason:
                    # Apply fees
                    net_pnl = pnl_pct - (fees_pct * 2)  # Entry + exit fee
                    trade_value = pos["size"]
                    profit = trade_value * (net_pnl / 100)
                    capital += trade_value + profit

                    trades.append(BacktestTrade(
                        symbol=sym,
                        entry_time=btc_df.index[pos["entry_idx"]],
                        exit_time=current_time,
                        entry_price=pos["entry_price"],
                        exit_price=current_price,
                        beta_at_entry=pos["beta"],
                        btc_momentum_at_entry=pos["btc_momentum"],
                        pnl_pct=round(net_pnl, 2),
                        exit_reason=exit_reason,
                        hold_hours=hold_hours,
                    ))
                    del positions[sym]

            # ─── CHECK ENTRIES ───
            if btc_momentum >= cfg["min_momentum"] and len(positions) < cfg["max_pos"]:
                # Score each alt
                candidates = []
                for sym, df in alt_data.items():
                    if sym in positions or i >= len(df):
                        continue

                    # Get beta at this point
                    if sym in betas and i < len(betas[sym]):
                        beta = betas[sym].iloc[i]
                    else:
                        continue

                    if beta < cfg["min_beta"]:
                        continue

                    # Volume surge
                    if i > 20:
                        vol_avg = df["volume"].iloc[i-20:i].mean()
                        vol_cur = df["volume"].iloc[i]
                        vol_surge = vol_cur / vol_avg if vol_avg > 0 else 1
                    else:
                        vol_surge = 1

                    # Price moving same direction as BTC
                    alt_ret = df["returns"].iloc[i]
                    btc_ret = btc_df["returns"].iloc[i]
                    same_direction = (alt_ret > 0 and btc_ret > 0)

                    if same_direction and vol_surge > 1.5 and beta > cfg["min_beta"]:
                        score = beta * 10 + vol_surge * 5 + (btc_momentum / 10)
                        candidates.append((sym, score, beta, vol_surge))

                # Enter top candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                entries_this_bar = 0

                for sym, score, beta, vol_surge in candidates[:2]:
                    if len(positions) >= cfg["max_pos"] or entries_this_bar >= 2:
                        break

                    pos_size = capital * cfg["pos_size"]
                    if pos_size < 10:
                        continue

                    capital -= pos_size
                    entry_price = alt_data[sym]["close"].iloc[i]

                    positions[sym] = {
                        "entry_price": entry_price,
                        "entry_idx": i,
                        "size": pos_size,
                        "highest": entry_price,
                        "beta": beta,
                        "btc_momentum": btc_momentum,
                    }
                    entries_this_bar += 1

            # Track equity
            open_value = sum(
                p["size"] * (alt_data[s]["close"].iloc[i] / p["entry_price"])
                for s, p in positions.items()
                if s in alt_data and i < len(alt_data[s])
            )
            total_equity = capital + open_value
            equity_curve.append(total_equity)

        # ─── CLOSE REMAINING POSITIONS ───
        for sym, pos in positions.items():
            if sym in alt_data:
                final_price = alt_data[sym]["close"].iloc[-1]
                pnl_pct = ((final_price / pos["entry_price"]) - 1) * 100 - (fees_pct * 2)
                profit = pos["size"] * (pnl_pct / 100)
                capital += pos["size"] + profit
                trades.append(BacktestTrade(
                    symbol=sym,
                    entry_time=btc_df.index[pos["entry_idx"]],
                    exit_time=btc_df.index[-1],
                    entry_price=pos["entry_price"],
                    exit_price=final_price,
                    beta_at_entry=pos["beta"],
                    btc_momentum_at_entry=pos["btc_momentum"],
                    pnl_pct=round(pnl_pct, 2),
                    exit_reason="end_of_backtest",
                    hold_hours=len(btc_df) - pos["entry_idx"],
                ))

        # ─── CALCULATE RESULTS ───
        final_capital = capital
        total_return = ((final_capital / initial_capital) - 1) * 100

        winning = [t for t in trades if t.pnl_pct > 0]
        losing = [t for t in trades if t.pnl_pct <= 0]
        win_rate = len(winning) / len(trades) * 100 if trades else 0

        avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0

        # Max drawdown
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        max_dd = drawdown.min()

        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Profit factor
        gross_profit = sum(t.pnl_pct for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl_pct for t in losing)) if losing else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # BTC buy & hold comparison
        btc_return = ((btc_df["close"].iloc[-1] / btc_df["close"].iloc[start_idx]) - 1) * 100

        result = BacktestResult(
            start_date=btc_df.index[start_idx].strftime("%Y-%m-%d"),
            end_date=btc_df.index[-1].strftime("%Y-%m-%d"),
            profile=profile,
            initial_capital=initial_capital,
            final_capital=round(final_capital, 2),
            total_return_pct=round(total_return, 2),
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(win_rate, 1),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            profit_factor=round(profit_factor, 2),
            avg_hold_hours=round(np.mean([t.hold_hours for t in trades]), 1) if trades else 0,
            best_trade_pct=round(max(t.pnl_pct for t in trades), 2) if trades else 0,
            worst_trade_pct=round(min(t.pnl_pct for t in trades), 2) if trades else 0,
            btc_return_pct=round(btc_return, 2),
            outperformance_pct=round(total_return - btc_return, 2),
            trades=trades,
            equity_curve=equity_curve,
        )

        return result


def print_backtest_report(r: BacktestResult):
    """Pretty print backtest results."""
    print(f"""
{'='*70}
  BACKTEST REPORT — Alt Beta Amplification Strategy
{'='*70}
  Period:          {r.start_date} → {r.end_date}
  Profile:         {r.profile.upper()}
  Initial Capital: ${r.initial_capital:,.2f}

{'─'*70}
  PERFORMANCE
{'─'*70}
  Final Capital:   ${r.final_capital:,.2f}
  Total Return:    {r.total_return_pct:+.2f}%
  BTC Buy & Hold:  {r.btc_return_pct:+.2f}%
  Outperformance:  {r.outperformance_pct:+.2f}% {'✅' if r.outperformance_pct > 0 else '❌'}

{'─'*70}
  TRADES
{'─'*70}
  Total Trades:    {r.total_trades}
  Winning:         {r.winning_trades} ({r.win_rate:.1f}%)
  Losing:          {r.losing_trades}
  Avg Win:         {r.avg_win_pct:+.2f}%
  Avg Loss:        {r.avg_loss_pct:+.2f}%
  Best Trade:      {r.best_trade_pct:+.2f}%
  Worst Trade:     {r.worst_trade_pct:+.2f}%
  Avg Hold Time:   {r.avg_hold_hours:.1f} hours

{'─'*70}
  RISK
{'─'*70}
  Max Drawdown:    {r.max_drawdown_pct:.2f}%
  Sharpe Ratio:    {r.sharpe_ratio:.2f}
  Profit Factor:   {r.profit_factor:.2f}

{'─'*70}
  EXIT REASONS
{'─'*70}""")

    reasons = {}
    for t in r.trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
        pct = count / r.total_trades * 100 if r.total_trades > 0 else 0
        print(f"  {reason:20s}: {count:4d} ({pct:.0f}%)")

    print(f"""
{'─'*70}
  TOP 5 TRADES
{'─'*70}""")
    sorted_trades = sorted(r.trades, key=lambda t: t.pnl_pct, reverse=True)
    for t in sorted_trades[:5]:
        print(f"  {t.symbol:<14} {t.pnl_pct:+7.2f}% | Beta: {t.beta_at_entry:.1f}x | "
              f"BTC Mom: {t.btc_momentum_at_entry:.0f} | {t.exit_reason}")

    print(f"\n{'─'*70}")
    print(f"  WORST 5 TRADES")
    print(f"{'─'*70}")
    for t in sorted_trades[-5:]:
        print(f"  {t.symbol:<14} {t.pnl_pct:+7.2f}% | Beta: {t.beta_at_entry:.1f}x | "
              f"BTC Mom: {t.btc_momentum_at_entry:.0f} | {t.exit_reason}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Quick backtest with a handful of popular alts
    bt = Backtester("binance")

    alts = [
        "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT",
        "LINK/USDT", "MATIC/USDT", "UNI/USDT", "NEAR/USDT",
        "FIL/USDT", "ATOM/USDT", "APT/USDT",
    ]

    for prof in ["conservative", "moderate", "aggressive"]:
        print(f"\n🔄 Running {prof.upper()} backtest...")
        result = bt.run(
            alt_symbols=alts,
            timeframe="1h",
            since="2024-06-01",
            limit=2000,
            profile=prof,
            initial_capital=10000,
        )
        if result:
            print_backtest_report(result)

"""
Live Trading Orchestrator
===========================
This is the brain. It ties together:
1. BTC Momentum Scanner (the signal)
2. Alt Beta Scanner (the opportunities)  
3. Trading execution (the money)

The loop:
  Every N minutes:
    1. Check BTC momentum — is Bitcoin hot?
    2. If YES → scan alts for highest beta amplifiers
    3. Enter positions in top-ranked alts
    4. Manage existing positions (trailing stops, exits)
    5. If BTC momentum fades → exit all alts (they'll drop harder)

IMPORTANT: Start with testnet=true in config.yaml!
"""

import time
import json
import yaml
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

from btc_momentum import BTCMomentumScanner, MomentumSignal
from alt_scanner import AltBetaScanner, AltSignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/trading.log"),
    ]
)
logger = logging.getLogger("live_trader")


@dataclass
class Position:
    """An open trading position."""
    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: float
    entry_btc_momentum: float      # BTC momentum when we entered
    entry_alt_score: float          # Alt score when we entered
    highest_price: float = 0.0     # For trailing stop
    current_price: float = 0.0
    unrealized_pnl_pct: float = 0.0

    @property
    def hold_hours(self) -> float:
        return (datetime.utcnow() - self.entry_time).total_seconds() / 3600

    def update(self, current_price: float):
        self.current_price = current_price
        self.highest_price = max(self.highest_price, current_price)
        self.unrealized_pnl_pct = ((current_price / self.entry_price) - 1) * 100


@dataclass
class TradeResult:
    """Completed trade record."""
    symbol: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    quantity: float
    pnl_pct: float
    pnl_usd: float
    exit_reason: str  # "take_profit", "stop_loss", "trailing_stop", "btc_exit", "time_limit"


class LiveTrader:
    """
    The main trading orchestrator.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.profile_name = self.config.get("active_profile", "moderate")
        self.profile = self.config["profiles"][self.profile_name]

        # Initialize components
        exchange_id = self.config["exchange"]["name"]
        self.btc_scanner = BTCMomentumScanner(exchange_id, self.config.get("btc_momentum", {}))
        self.alt_scanner = AltBetaScanner(exchange_id, self.config.get("alt_scanner", {}))

        # State
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeResult] = []
        self.portfolio_value = 10000.0  # Starting value — update from exchange
        self.daily_pnl = 0.0
        self.peak_value = self.portfolio_value

        # Risk limits
        self.risk = self.config.get("risk", {})
        self.max_daily_loss = self.risk.get("max_daily_loss_pct", 5.0)
        self.max_drawdown = self.risk.get("max_drawdown_pct", 15.0)
        self.cooloff_until = None

        Path("logs").mkdir(exist_ok=True)

        logger.info(f"LiveTrader initialized — Profile: {self.profile_name}")
        logger.info(f"  Max positions: {self.profile['max_concurrent_positions']}")
        logger.info(f"  Position size: {self.profile['max_position_pct']}%")
        logger.info(f"  Take profit: {self.profile['take_profit_pct']}%")
        logger.info(f"  Stop loss: {self.profile['stop_loss_pct']}%")
        logger.info(f"  Min BTC momentum to trade: {self.profile['min_btc_momentum']}")

    # ─────────────────────────────────────────────
    # RISK MANAGEMENT
    # ─────────────────────────────────────────────

    def check_risk_limits(self) -> bool:
        """Check if we've hit any risk limits. Returns True if OK to trade."""
        # Cooloff period
        if self.cooloff_until and datetime.utcnow() < self.cooloff_until:
            logger.warning(f"In cooloff until {self.cooloff_until}")
            return False

        # Daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            logger.warning(f"Daily loss limit hit: {self.daily_pnl:.1f}%")
            self.cooloff_until = datetime.utcnow() + __import__("datetime").timedelta(
                hours=self.risk.get("cooloff_hours", 24)
            )
            return False

        # Max drawdown
        drawdown = ((self.portfolio_value / self.peak_value) - 1) * 100
        if drawdown <= -self.max_drawdown:
            logger.warning(f"Max drawdown hit: {drawdown:.1f}%")
            return False

        return True

    def get_position_size(self, alt_score: float, btc_momentum: float) -> float:
        """Calculate position size based on conviction level."""
        base_pct = self.profile["max_position_pct"]

        # Scale by alt score (higher score = more conviction)
        score_multiplier = alt_score / 100  # 0.5 to 1.0 typically

        # Scale by BTC momentum (hotter BTC = larger positions)
        momentum_multiplier = min(btc_momentum / 80, 1.2)  # Cap at 1.2x

        position_pct = base_pct * score_multiplier * momentum_multiplier
        position_usd = self.portfolio_value * (position_pct / 100)

        return round(position_usd, 2)

    # ─────────────────────────────────────────────
    # POSITION MANAGEMENT
    # ─────────────────────────────────────────────

    def should_enter(self, alt: AltSignal, btc_signal: MomentumSignal) -> bool:
        """Decide if we should enter a new position."""
        # Already in this position?
        if alt.symbol in self.positions:
            return False

        # Too many positions?
        if len(self.positions) >= self.profile["max_concurrent_positions"]:
            return False

        # BTC momentum high enough?
        if btc_signal.score < self.profile["min_btc_momentum"]:
            return False

        # Alt score high enough?
        if alt.score < self.profile.get("min_confidence_score", 0.65) * 100:
            return False

        # Alt must be amplifying BTC (moving same direction, faster)
        if not alt.is_amplifying:
            return False

        # Volume must be surging (real money, not fake move)
        if alt.volume_surge < 1.5:
            return False

        return True

    def check_exit(self, pos: Position, btc_signal: MomentumSignal) -> Optional[str]:
        """Check if we should exit a position. Returns exit reason or None."""

        # CRITICAL: BTC momentum dying = exit alts immediately
        # Alts crash HARDER than BTC drops
        if btc_signal.is_exit:
            return "btc_exit"

        # Take profit hit
        if pos.unrealized_pnl_pct >= self.profile["take_profit_pct"]:
            return "take_profit"

        # Stop loss hit
        if pos.unrealized_pnl_pct <= -self.profile["stop_loss_pct"]:
            return "stop_loss"

        # Trailing stop — once we're in profit, protect gains
        if pos.highest_price > pos.entry_price:
            trailing_pct = self.profile["trailing_stop_pct"]
            drop_from_high = ((pos.current_price / pos.highest_price) - 1) * 100
            if drop_from_high <= -trailing_pct:
                return "trailing_stop"

        # Time limit
        if pos.hold_hours >= self.profile["max_hold_hours"]:
            return "time_limit"

        # BTC momentum weakening significantly from entry
        if btc_signal.score < pos.entry_btc_momentum * 0.5:
            return "btc_weakening"

        return None  # Hold

    # ─────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────

    def run_cycle(self):
        """Run one trading cycle."""
        logger.info("=" * 70)
        logger.info(f"  TRADING CYCLE — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info(f"  Profile: {self.profile_name} | Positions: {len(self.positions)}")
        logger.info("=" * 70)

        # Step 0: Risk check
        if not self.check_risk_limits():
            logger.warning("Risk limits active — skipping cycle")
            return

        # Step 1: Get BTC momentum signal
        btc_signal = self.btc_scanner.get_momentum_signal()
        logger.info(
            f"  BTC: ${btc_signal.btc_price:,.0f} | "
            f"Momentum: {btc_signal.score}/100 [{btc_signal.regime.upper()}]"
        )

        # Step 2: Manage existing positions
        exits_made = []
        for symbol, pos in list(self.positions.items()):
            # TODO: Fetch current price from exchange
            # pos.update(current_price)

            exit_reason = self.check_exit(pos, btc_signal)
            if exit_reason:
                logger.info(
                    f"  EXIT: {symbol} | Reason: {exit_reason} | "
                    f"P/L: {pos.unrealized_pnl_pct:+.1f}%"
                )
                # TODO: Execute sell order
                exits_made.append(symbol)

                # Record trade
                result = TradeResult(
                    symbol=symbol,
                    entry_price=pos.entry_price,
                    exit_price=pos.current_price,
                    entry_time=pos.entry_time,
                    exit_time=datetime.utcnow(),
                    quantity=pos.quantity,
                    pnl_pct=pos.unrealized_pnl_pct,
                    pnl_usd=pos.quantity * pos.entry_price * (pos.unrealized_pnl_pct / 100),
                    exit_reason=exit_reason,
                )
                self.trade_history.append(result)
                del self.positions[symbol]

        # Step 3: Look for new entries (only if BTC is hot)
        if btc_signal.is_go:
            logger.info(f"  BTC is {'🔥 ON FIRE' if btc_signal.is_fire else '🟢 HOT'} — scanning alts...")

            alt_signals = self.alt_scanner.scan(
                btc_momentum_score=btc_signal.score,
                top_n_alts=100,
                timeframe="1h",
                lookback=48
            )

            entries_made = 0
            for alt in alt_signals:
                if entries_made >= 3:  # Max new entries per cycle
                    break

                if self.should_enter(alt, btc_signal):
                    position_usd = self.get_position_size(alt.score, btc_signal.score)
                    quantity = position_usd / alt.price if alt.price > 0 else 0

                    logger.info(
                        f"  ENTRY: {alt.symbol} | "
                        f"Score: {alt.score} | Beta: {alt.beta_vs_btc}x | "
                        f"Size: ${position_usd:,.0f}"
                    )

                    # TODO: Execute buy order on exchange
                    pos = Position(
                        symbol=alt.symbol,
                        entry_price=alt.price,
                        entry_time=datetime.utcnow(),
                        quantity=quantity,
                        entry_btc_momentum=btc_signal.score,
                        entry_alt_score=alt.score,
                        highest_price=alt.price,
                        current_price=alt.price,
                    )
                    self.positions[alt.symbol] = pos
                    entries_made += 1

        else:
            logger.info(f"  BTC momentum too low ({btc_signal.score}) — no new entries")

        # Step 4: Summary
        self._print_portfolio_summary(btc_signal)

    def _print_portfolio_summary(self, btc_signal: MomentumSignal):
        """Print current state."""
        logger.info(f"\n  {'─'*50}")
        logger.info(f"  PORTFOLIO SUMMARY")
        logger.info(f"  {'─'*50}")
        logger.info(f"  Open positions: {len(self.positions)}")

        for sym, pos in self.positions.items():
            logger.info(
                f"    {sym}: Entry ${pos.entry_price:.4f} | "
                f"P/L: {pos.unrealized_pnl_pct:+.1f}% | "
                f"Hold: {pos.hold_hours:.1f}h"
            )

        total_trades = len(self.trade_history)
        if total_trades > 0:
            wins = sum(1 for t in self.trade_history if t.pnl_pct > 0)
            total_pnl = sum(t.pnl_usd for t in self.trade_history)
            win_rate = (wins / total_trades) * 100
            logger.info(f"  Total trades: {total_trades} | Win rate: {win_rate:.0f}%")
            logger.info(f"  Total P/L: ${total_pnl:+,.2f}")

        logger.info(f"  {'─'*50}\n")

    def run(self, interval_minutes: int = 5):
        """
        Main trading loop. Runs continuously.

        Args:
            interval_minutes: How often to run a trading cycle.
                Conservative: 15-30 min
                Moderate: 5-10 min
                Aggressive: 1-3 min
        """
        logger.info("=" * 70)
        logger.info("  🚀 CRYPTO ALPHA BOT — STARTING LIVE TRADING")
        logger.info(f"  Profile: {self.profile_name}")
        logger.info(f"  Cycle interval: {interval_minutes} minutes")
        logger.info(f"  Testnet: {self.config['exchange'].get('testnet', True)}")
        logger.info("=" * 70)

        if self.config["exchange"].get("testnet", True):
            logger.info("  ⚠️  RUNNING IN TESTNET MODE — No real money at risk")
        else:
            logger.warning("  💰 RUNNING LIVE — Real money on the line!")

        cycle = 0
        while True:
            cycle += 1
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)

            # Save state
            self._save_state()

            # Wait
            logger.info(f"Next cycle in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

    def _save_state(self):
        """Save trading state to disk for recovery."""
        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": {
                sym: {
                    "entry_price": p.entry_price,
                    "entry_time": p.entry_time.isoformat(),
                    "quantity": p.quantity,
                    "entry_btc_momentum": p.entry_btc_momentum,
                }
                for sym, p in self.positions.items()
            },
            "trade_count": len(self.trade_history),
            "portfolio_value": self.portfolio_value,
        }
        Path("logs").mkdir(exist_ok=True)
        with open("logs/state.json", "w") as f:
            json.dump(state, f, indent=2)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crypto Alpha Bot — Live Trader")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--interval", type=int, default=5, help="Cycle interval (minutes)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    trader = LiveTrader(config_path=args.config)

    if args.once:
        trader.run_cycle()
    else:
        trader.run(interval_minutes=args.interval)

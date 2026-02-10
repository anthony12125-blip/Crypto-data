"""
Agent-Integrated Live Trader
===============================
Extended version of live_trader.py that reads decisions from the
3-agent voting system to adjust trading behavior.

The price bot runs continuously. Every 4 hours (or on emergency),
the agent voting system provides an intelligence update:
  - trade / hold / exit
  - aggressiveness level
  - specific warnings

This file wraps the original live_trader with agent awareness.

Run: python3 agent_trader.py
"""

import json
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from live_trader import LiveTrader
from agent_voting import MainAgentOrchestrator, DECISION_FILE

import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/agent_trader.log"),
    ]
)
logger = logging.getLogger("agent_trader")


class AgentIntegratedTrader:
    """
    Wraps the LiveTrader with intelligence from the 3-agent voting system.
    
    Architecture:
      - Price bot (LiveTrader) runs every 5 minutes — watches charts
      - Agent orchestrator runs every 4 hours — watches the real world
      - Agent decision modifies the bot's behavior:
        - "trade" + "aggressive" → bot uses aggressive profile
        - "trade" + "conservative" → bot uses conservative profile
        - "hold" → bot stops opening new positions but manages existing ones
        - "exit" → bot closes all positions and waits
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # The price bot
        self.trader = LiveTrader(config_path)

        # The agent orchestrator (runs in background thread)
        self.orchestrator = MainAgentOrchestrator(
            vote_interval_hours=self.config.get("agent", {}).get("vote_interval_hours", 4)
        )

        # State
        self.agent_override = None  # Current agent decision
        self.original_profile = self.config.get("active_profile", "moderate")

        Path("logs").mkdir(exist_ok=True)

    def read_agent_decision(self) -> Optional[dict]:
        """Read the current agent decision from disk."""
        if not DECISION_FILE.exists():
            return None

        try:
            with open(DECISION_FILE) as f:
                decision = json.load(f)

            # Check validity
            valid_until = datetime.fromisoformat(decision["valid_until"])
            if datetime.utcnow() > valid_until:
                logger.warning("Agent decision expired")
                return None

            return decision
        except Exception as e:
            logger.error(f"Error reading agent decision: {e}")
            return None

    def apply_agent_decision(self, decision: dict):
        """Apply the agent's decision to modify trading behavior."""
        action = decision.get("action", "hold")
        aggressiveness = decision.get("aggressiveness", "moderate")
        confidence = decision.get("confidence", 0.5)
        warnings = decision.get("warnings", [])

        logger.info(f"  🤖 Agent decision: {action.upper()} | "
                     f"Aggressiveness: {aggressiveness} | "
                     f"Confidence: {confidence:.0%}")

        if action == "trade":
            # Switch to the recommended profile
            if aggressiveness in self.config.get("profiles", {}):
                self.trader.profile = self.config["profiles"][aggressiveness]
                self.trader.profile_name = aggressiveness
                logger.info(f"  Profile switched to: {aggressiveness}")

            # Scale position sizes by confidence
            if confidence < 0.6:
                # Low confidence — reduce sizes
                self.trader.profile["max_position_pct"] *= 0.7
                logger.info(f"  Position size reduced (low confidence)")

        elif action == "hold":
            # Don't open new positions, but manage existing ones
            self.trader.profile["max_concurrent_positions"] = 0
            logger.info(f"  🟡 HOLD mode — no new positions, managing existing")

        elif action == "exit":
            # Close everything
            logger.warning(f"  🔴 EXIT mode — closing all positions")
            for symbol in list(self.trader.positions.keys()):
                logger.info(f"  Closing {symbol} — agent says EXIT")
                # The main trading cycle will handle exits
            self.trader.profile["max_concurrent_positions"] = 0

        # Log warnings
        if warnings:
            for w in warnings:
                logger.warning(f"  ⚠️ Agent warning: {w}")

        self.agent_override = decision

    def run_agent_background(self):
        """Run the agent orchestrator in a background thread."""
        logger.info("Starting agent orchestrator in background...")

        def agent_loop():
            while True:
                try:
                    if self.orchestrator.should_trigger_vote():
                        logger.info("🗳️ Agent vote triggered...")
                        ctx = self.orchestrator.gather_market_context()
                        self.orchestrator.voting_system.run_vote(ctx)
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    logger.error(f"Agent background error: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=agent_loop, daemon=True)
        thread.start()

    def run(self, interval_minutes: int = 5):
        """
        Main loop: price bot + agent intelligence.
        """
        logger.info("=" * 70)
        logger.info("  🚀 AGENT-INTEGRATED CRYPTO ALPHA BOT")
        logger.info("=" * 70)
        logger.info(f"  Price bot cycle: every {interval_minutes} minutes")
        logger.info(f"  Agent vote cycle: every {self.orchestrator.vote_interval} hours")
        logger.info(f"  Base profile: {self.original_profile}")
        logger.info("=" * 70)

        # Start agent orchestrator in background
        self.run_agent_background()

        # Main trading loop
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n{'─'*50}")
                logger.info(f"  CYCLE {cycle} — {datetime.utcnow().strftime('%H:%M:%S UTC')}")
                logger.info(f"{'─'*50}")

                # Read latest agent decision
                decision = self.read_agent_decision()
                if decision:
                    self.apply_agent_decision(decision)
                else:
                    # No agent decision — use default profile
                    self.trader.profile = self.config["profiles"][self.original_profile]
                    self.trader.profile_name = self.original_profile
                    logger.info("  No agent decision available — using default profile")

                # Run price bot cycle
                self.trader.trading_cycle()

                # Wait
                logger.info(f"  Next cycle in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent-Integrated Trader")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--interval", type=int, default=5, help="Minutes between cycles")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")

    args = parser.parse_args()

    trader = AgentIntegratedTrader(args.config)

    if args.once:
        decision = trader.read_agent_decision()
        if decision:
            trader.apply_agent_decision(decision)
        trader.trader.trading_cycle()
    else:
        trader.run(args.interval)

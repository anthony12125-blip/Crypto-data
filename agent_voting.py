"""
3-Agent Voting System
=======================
The intelligence layer for the crypto trading bot.

Architecture:
  MAIN AGENT (always running on M4):
    - Monitors market in real-time via price bot
    - Watches for decision points (BTC momentum shifts, alt signals)
    - When a trade decision is needed, spins up the VOTE

  VOTE SYSTEM (on-demand):
    - Agent A: 7B model fine-tuned on crypto data (training variant 1)
    - Agent B: 7B model fine-tuned on crypto data (training variant 2)
    - Main Agent: Acts as tiebreaker/orchestrator
    - All three analyze current conditions
    - Majority vote determines action

  OUTPUT → single decision to the price bot:
    - trade/no-trade
    - aggressiveness level (conservative/moderate/aggressive)
    - specific warnings or context
    - valid for next 4 hours until next vote

Run: python3 agent_voting.py
"""

import json
import subprocess
import sys
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import threading
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/agent_voting.log"),
    ]
)
logger = logging.getLogger("agent_voting")

# Model paths
MODEL_DIR = Path("models/llm")
AGENT_A_MODEL = MODEL_DIR / "crypto_analyst_a_fused"
AGENT_B_MODEL = MODEL_DIR / "crypto_analyst_b_fused"
FALLBACK_MODEL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

# Decision output file — the price bot reads this
DECISION_FILE = Path("data/agent_decision.json")


@dataclass
class MarketContext:
    """Current market state passed to all agents for analysis."""
    timestamp: str
    btc_price: float
    btc_momentum_score: float
    btc_regime: str
    btc_change_1h: float
    btc_change_24h: float
    fear_greed_index: int
    btc_dominance: float
    top_alts: List[Dict]  # Top amplifying alts from scanner
    recent_news: List[str]  # Recent headlines
    regime: str  # bear/transition/early_bull/bull/overheated
    regime_confidence: float


@dataclass
class AgentVote:
    """A single agent's vote on what to do."""
    agent_id: str  # "agent_a", "agent_b", "main_agent"
    model_path: str
    vote: str  # "trade", "hold", "exit"
    confidence: float  # 0-1
    aggressiveness: str  # "conservative", "moderate", "aggressive"
    reasoning: str
    warnings: List[str]
    timestamp: str
    inference_time_ms: float


@dataclass
class VoteDecision:
    """The final consensus decision from all three agents."""
    timestamp: str
    valid_until: str  # 4 hours from now
    action: str  # "trade", "hold", "exit"
    aggressiveness: str
    confidence: float
    votes: List[AgentVote]
    vote_count: Dict[str, int]  # {"trade": 2, "hold": 1}
    unanimous: bool
    summary: str
    warnings: List[str]
    market_context: Dict


class LLMInference:
    """Run inference on a local MLX model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.available = self._check_model()

    def _check_model(self) -> bool:
        """Check if model exists locally."""
        if Path(self.model_path).exists():
            return True
        # Check if it's a HuggingFace model ID (will be cached)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mlx_lm.generate",
                 "--model", self.model_path,
                 "--prompt", "test",
                 "--max-tokens", "1"],
                capture_output=True, text=True, timeout=60,
            )
            return result.returncode == 0
        except:
            return False

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        """Generate a response from the model."""
        full_prompt = f"[INST] {prompt} [/INST]"

        cmd = [
            sys.executable, "-m", "mlx_lm.generate",
            "--model", self.model_path,
            "--prompt", full_prompt,
            "--max-tokens", str(max_tokens),
            "--temp", str(temperature),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"Generation error: {result.stderr}")
                return ""
        except subprocess.TimeoutExpired:
            logger.error("Model inference timed out (120s)")
            return ""
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return ""


class AgentVotingSystem:
    """
    Three-agent voting system for crypto trading decisions.
    """

    def __init__(self):
        self.decision_interval_hours = 4
        self.last_decision: Optional[VoteDecision] = None
        self.decision_history: List[VoteDecision] = []

        # Initialize agents
        logger.info("Initializing 3-Agent Voting System...")
        self._init_agents()

    def _init_agents(self):
        """Set up the three agents."""
        # Agent A: Primary fine-tuned model
        a_path = str(AGENT_A_MODEL) if AGENT_A_MODEL.exists() else FALLBACK_MODEL
        self.agent_a = LLMInference(a_path)
        logger.info(f"  Agent A: {a_path} ({'✅' if self.agent_a.available else '❌'})")

        # Agent B: Secondary fine-tuned model (different training variant)
        b_path = str(AGENT_B_MODEL) if AGENT_B_MODEL.exists() else FALLBACK_MODEL
        self.agent_b = LLMInference(b_path)
        logger.info(f"  Agent B: {b_path} ({'✅' if self.agent_b.available else '❌'})")

        # Main Agent: Uses same model as A but with different prompting strategy
        self.main_agent = self.agent_a
        logger.info(f"  Main Agent: orchestrator (uses Agent A model)")

    def build_analysis_prompt(self, ctx: MarketContext, agent_role: str) -> str:
        """Build the analysis prompt for each agent."""

        # Top alts summary
        alt_summary = ""
        for alt in ctx.top_alts[:5]:
            alt_summary += (
                f"  - {alt.get('symbol', '?')}: "
                f"score={alt.get('score', 0):.0f}, "
                f"beta={alt.get('beta', 0):.1f}x, "
                f"24h={alt.get('change_24h', 0):+.1f}%\n"
            )

        # News summary
        news_summary = "\n".join(f"  - {h}" for h in ctx.recent_news[:10]) if ctx.recent_news else "  No recent news available"

        # Role-specific instructions
        if agent_role == "agent_a":
            role_instruction = (
                "You are Agent A, a conservative crypto market analyst. "
                "You prioritize risk management and capital preservation. "
                "You only recommend trading when signals are very strong. "
                "You tend to err on the side of caution."
            )
        elif agent_role == "agent_b":
            role_instruction = (
                "You are Agent B, an aggressive crypto market analyst. "
                "You look for opportunities and amplification potential. "
                "You're willing to take calculated risks when the setup is good. "
                "You focus on maximizing upside during bull momentum."
            )
        else:
            role_instruction = (
                "You are the Main Agent, a balanced crypto market analyst. "
                "You weigh both risk and opportunity equally. "
                "You focus on whether the current setup matches historical "
                "patterns of successful alt amplification trades."
            )

        prompt = f"""{role_instruction}

CURRENT MARKET STATE ({ctx.timestamp}):

Bitcoin:
  Price: ${ctx.btc_price:,.2f}
  Momentum Score: {ctx.btc_momentum_score:.0f}/100
  Regime: {ctx.btc_regime}
  1h Change: {ctx.btc_change_1h:+.2f}%
  24h Change: {ctx.btc_change_24h:+.2f}%

Market:
  Fear & Greed Index: {ctx.fear_greed_index}/100
  Market Regime: {ctx.regime} (confidence: {ctx.regime_confidence:.0%})
  BTC Dominance: {ctx.btc_dominance:.1f}%

Top Amplifying Altcoins:
{alt_summary}

Recent News Headlines:
{news_summary}

Based on this data, provide your analysis in EXACTLY this JSON format:
{{
  "vote": "trade" or "hold" or "exit",
  "confidence": 0.0 to 1.0,
  "aggressiveness": "conservative" or "moderate" or "aggressive",
  "reasoning": "One paragraph explaining your decision",
  "warnings": ["list", "of", "specific", "risks"]
}}

IMPORTANT: Respond ONLY with the JSON. No other text."""

        return prompt

    def query_agent(self, agent: LLMInference, ctx: MarketContext,
                    agent_id: str) -> AgentVote:
        """Query a single agent for its vote."""
        prompt = self.build_analysis_prompt(ctx, agent_id)

        start_time = time.time()
        response = agent.generate(prompt, max_tokens=500, temperature=0.3)
        inference_ms = (time.time() - start_time) * 1000

        # Parse response
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
            else:
                raise ValueError("No JSON found in response")

            return AgentVote(
                agent_id=agent_id,
                model_path=agent.model_path,
                vote=parsed.get("vote", "hold"),
                confidence=float(parsed.get("confidence", 0.5)),
                aggressiveness=parsed.get("aggressiveness", "moderate"),
                reasoning=parsed.get("reasoning", "No reasoning provided"),
                warnings=parsed.get("warnings", []),
                timestamp=datetime.utcnow().isoformat(),
                inference_time_ms=round(inference_ms, 1),
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"  {agent_id} response parse failed: {e}")
            logger.debug(f"  Raw response: {response[:200]}")

            # Fallback: conservative hold
            return AgentVote(
                agent_id=agent_id,
                model_path=agent.model_path,
                vote="hold",
                confidence=0.3,
                aggressiveness="conservative",
                reasoning=f"Parse error, defaulting to hold. Raw: {response[:100]}",
                warnings=["Agent response could not be parsed"],
                timestamp=datetime.utcnow().isoformat(),
                inference_time_ms=round(inference_ms, 1),
            )

    def run_vote(self, ctx: MarketContext) -> VoteDecision:
        """
        Run the 3-agent vote.
        Spins up agents, collects votes, determines consensus.
        """
        logger.info("=" * 60)
        logger.info("  🗳️  3-AGENT VOTE INITIATED")
        logger.info("=" * 60)
        logger.info(f"  BTC: ${ctx.btc_price:,.0f} | Momentum: {ctx.btc_momentum_score:.0f} | "
                     f"Regime: {ctx.regime}")

        votes: List[AgentVote] = []

        # Query all three agents (sequentially to conserve RAM)
        for agent_id, agent in [
            ("agent_a", self.agent_a),
            ("agent_b", self.agent_b),
            ("main_agent", self.main_agent),
        ]:
            logger.info(f"\n  Querying {agent_id}...")
            vote = self.query_agent(agent, ctx, agent_id)
            votes.append(vote)
            logger.info(f"  {agent_id}: {vote.vote.upper()} "
                        f"(confidence: {vote.confidence:.0%}, "
                        f"aggressiveness: {vote.aggressiveness}) "
                        f"[{vote.inference_time_ms:.0f}ms]")

        # Count votes
        vote_counts = {}
        for v in votes:
            vote_counts[v.vote] = vote_counts.get(v.vote, 0) + 1

        # Determine winning action (majority)
        winning_action = max(vote_counts, key=vote_counts.get)
        unanimous = vote_counts[winning_action] == 3

        # Determine aggressiveness (average of agreeing agents)
        agreeing = [v for v in votes if v.vote == winning_action]
        agg_map = {"conservative": 1, "moderate": 2, "aggressive": 3}
        agg_reverse = {1: "conservative", 2: "moderate", 3: "aggressive"}
        avg_agg = sum(agg_map.get(v.aggressiveness, 2) for v in agreeing) / len(agreeing)
        final_aggressiveness = agg_reverse[round(avg_agg)]

        # Average confidence of agreeing agents
        avg_confidence = sum(v.confidence for v in agreeing) / len(agreeing)

        # If split vote (2-1), reduce confidence
        if not unanimous:
            avg_confidence *= 0.8

        # Collect all warnings
        all_warnings = []
        for v in votes:
            all_warnings.extend(v.warnings)
        all_warnings = list(set(all_warnings))

        # Build summary
        vote_str = ", ".join(f"{v.agent_id}={v.vote}" for v in votes)
        summary = (
            f"Vote result: {winning_action.upper()} "
            f"({'UNANIMOUS' if unanimous else f'{vote_counts[winning_action]}-{3 - vote_counts[winning_action]}'}) | "
            f"Aggressiveness: {final_aggressiveness} | "
            f"Confidence: {avg_confidence:.0%} | "
            f"Votes: [{vote_str}]"
        )

        valid_until = (datetime.utcnow() + timedelta(hours=self.decision_interval_hours)).isoformat()

        decision = VoteDecision(
            timestamp=datetime.utcnow().isoformat(),
            valid_until=valid_until,
            action=winning_action,
            aggressiveness=final_aggressiveness,
            confidence=round(avg_confidence, 3),
            votes=votes,
            vote_count=vote_counts,
            unanimous=unanimous,
            summary=summary,
            warnings=all_warnings,
            market_context={
                "btc_price": ctx.btc_price,
                "btc_momentum": ctx.btc_momentum_score,
                "regime": ctx.regime,
                "fear_greed": ctx.fear_greed_index,
            }
        )

        # Log the decision
        logger.info(f"\n{'='*60}")
        logger.info(f"  🗳️  VOTE RESULT: {winning_action.upper()}")
        logger.info(f"  {'='*60}")
        logger.info(f"  {summary}")
        if all_warnings:
            logger.info(f"  ⚠️  Warnings: {', '.join(all_warnings)}")
        logger.info(f"  Valid until: {valid_until}")
        logger.info(f"{'='*60}\n")

        # Save decision for the price bot to read
        self._save_decision(decision)

        self.last_decision = decision
        self.decision_history.append(decision)

        return decision

    def _save_decision(self, decision: VoteDecision):
        """Save decision to JSON file for the price bot to read."""
        DECISION_FILE.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "timestamp": decision.timestamp,
            "valid_until": decision.valid_until,
            "action": decision.action,
            "aggressiveness": decision.aggressiveness,
            "confidence": decision.confidence,
            "unanimous": decision.unanimous,
            "vote_count": decision.vote_count,
            "summary": decision.summary,
            "warnings": decision.warnings,
            "market_context": decision.market_context,
            "votes": [
                {
                    "agent_id": v.agent_id,
                    "vote": v.vote,
                    "confidence": v.confidence,
                    "aggressiveness": v.aggressiveness,
                    "reasoning": v.reasoning,
                    "warnings": v.warnings,
                    "inference_time_ms": v.inference_time_ms,
                }
                for v in decision.votes
            ],
        }

        with open(DECISION_FILE, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Decision saved to {DECISION_FILE}")

    def get_current_decision(self) -> Optional[dict]:
        """Read the current valid decision (used by the price bot)."""
        if not DECISION_FILE.exists():
            return None

        with open(DECISION_FILE) as f:
            decision = json.load(f)

        # Check if still valid
        valid_until = datetime.fromisoformat(decision["valid_until"])
        if datetime.utcnow() > valid_until:
            logger.warning("Decision expired. Need new vote.")
            return None

        return decision


class MainAgentOrchestrator:
    """
    The always-on main agent that monitors the market and triggers votes.
    """

    def __init__(self, vote_interval_hours: int = 4):
        self.vote_interval = vote_interval_hours
        self.voting_system = AgentVotingSystem()
        self.running = False

    def gather_market_context(self) -> MarketContext:
        """Collect current market state from all scanners."""
        from btc_momentum import BTCMomentumScanner
        from alt_scanner import AltBetaScanner
        from regime_detector import RegimeDetector

        # BTC momentum
        btc_scanner = BTCMomentumScanner("binance")
        btc_signal = btc_scanner.get_momentum_signal()

        # Regime
        regime_detector = RegimeDetector("binance")
        regime = regime_detector.detect()

        # Alt scan
        alt_scanner = AltBetaScanner("binance")
        alt_signals = alt_scanner.scan(
            btc_momentum_score=btc_signal.score, top_n_alts=30
        )

        # Format alt data for agents
        top_alts = []
        for s in alt_signals[:10]:
            top_alts.append({
                "symbol": s.symbol,
                "score": s.score,
                "beta": s.beta_vs_btc,
                "change_24h": s.change_24h,
                "volume_surge": s.volume_surge,
            })

        # Recent news (placeholder — would integrate with news scraper)
        recent_news = self._fetch_recent_news()

        return MarketContext(
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            btc_price=btc_signal.btc_price,
            btc_momentum_score=btc_signal.score,
            btc_regime=btc_signal.regime,
            btc_change_1h=btc_signal.btc_change_1h,
            btc_change_24h=btc_signal.btc_change_24h,
            fear_greed_index=regime.fear_greed,
            btc_dominance=regime.btc_dominance if hasattr(regime, 'btc_dominance') else 0,
            top_alts=top_alts,
            recent_news=recent_news,
            regime=regime.regime,
            regime_confidence=regime.confidence,
        )

    def _fetch_recent_news(self) -> List[str]:
        """Fetch recent crypto headlines."""
        import requests
        try:
            resp = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/",
                params={"lang": "EN", "sortOrder": "latest"},
                timeout=10,
            )
            if resp.status_code == 200:
                articles = resp.json().get("Data", [])
                return [a["title"] for a in articles[:15]]
        except:
            pass
        return []

    def should_trigger_vote(self) -> bool:
        """Determine if it's time for a new vote."""
        current = self.voting_system.get_current_decision()

        # No current decision — need one
        if current is None:
            return True

        # Check if decision is about to expire (within 30 min)
        valid_until = datetime.fromisoformat(current["valid_until"])
        if datetime.utcnow() > valid_until - timedelta(minutes=30):
            return True

        # Emergency: BTC momentum crashed since last vote
        try:
            from btc_momentum import BTCMomentumScanner
            scanner = BTCMomentumScanner("binance")
            signal = scanner.get_momentum_signal()

            last_momentum = current.get("market_context", {}).get("btc_momentum", 50)
            if signal.score < last_momentum * 0.5:
                logger.warning(f"⚠️ BTC momentum crashed: {last_momentum:.0f} → {signal.score:.0f}")
                return True
        except:
            pass

        return False

    def run(self):
        """
        Main loop: monitor market, trigger votes when needed.
        The price bot keeps running independently.
        This just updates the intelligence layer.
        """
        self.running = True
        logger.info("=" * 60)
        logger.info("  🤖 MAIN AGENT ORCHESTRATOR STARTED")
        logger.info(f"  Vote interval: every {self.vote_interval} hours")
        logger.info(f"  Decision file: {DECISION_FILE}")
        logger.info("=" * 60)

        while self.running:
            try:
                if self.should_trigger_vote():
                    logger.info("\n🗳️ Triggering vote cycle...")

                    # Gather market context
                    logger.info("Gathering market context...")
                    ctx = self.gather_market_context()

                    # Run the 3-agent vote
                    decision = self.voting_system.run_vote(ctx)

                    logger.info(f"\nNext vote in ~{self.vote_interval} hours "
                                f"(or sooner if market changes dramatically)")

                else:
                    current = self.voting_system.get_current_decision()
                    if current:
                        valid_until = datetime.fromisoformat(current["valid_until"])
                        remaining = valid_until - datetime.utcnow()
                        logger.debug(f"Current decision: {current['action']} "
                                     f"(expires in {remaining})")

                # Check every 5 minutes
                time.sleep(300)

            except KeyboardInterrupt:
                logger.info("\nShutting down orchestrator...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Orchestrator error: {e}")
                time.sleep(60)  # Wait and retry

        logger.info("Orchestrator stopped.")


def print_current_decision():
    """Print the current active decision."""
    system = AgentVotingSystem()
    decision = system.get_current_decision()

    if not decision:
        print("\n  ❌ No active decision. Run the orchestrator to generate one.\n")
        return

    print(f"""
{'='*65}
  CURRENT AGENT DECISION
{'='*65}
  Action:         {decision['action'].upper()}
  Aggressiveness: {decision['aggressiveness']}
  Confidence:     {decision['confidence']:.0%}
  Unanimous:      {'Yes ✅' if decision['unanimous'] else 'No (split vote)'}
  Valid Until:     {decision['valid_until']}

  Vote Breakdown:""")

    for v in decision.get("votes", []):
        print(f"    {v['agent_id']:15s}: {v['vote']:6s} "
              f"(conf: {v['confidence']:.0%}, agg: {v['aggressiveness']})")
        print(f"      Reasoning: {v['reasoning'][:100]}...")

    if decision.get("warnings"):
        print(f"\n  ⚠️  Warnings:")
        for w in decision["warnings"]:
            print(f"    - {w}")

    print(f"\n  Summary: {decision['summary']}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3-Agent Voting System")
    parser.add_argument("--mode", choices=["run", "vote-now", "status"],
                        default="run", help="Operation mode")
    parser.add_argument("--interval", type=int, default=4,
                        help="Hours between votes")

    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    if args.mode == "run":
        orchestrator = MainAgentOrchestrator(vote_interval_hours=args.interval)
        orchestrator.run()

    elif args.mode == "vote-now":
        orchestrator = MainAgentOrchestrator()
        ctx = orchestrator.gather_market_context()
        decision = orchestrator.voting_system.run_vote(ctx)
        print_current_decision()

    elif args.mode == "status":
        print_current_decision()

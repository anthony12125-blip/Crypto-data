#!/usr/bin/env python3
"""
Crypto Training Data Formatter
Converts CryptoPanic news articles into MLX LoRA fine-tuning JSONL format.

Generates 3 types of training pairs per article:
  1. Sentiment Analysis
  2. Market Context Analysis
  3. Event Classification

Creates 3 agent training sets with different orderings:
  - Agent A: Category-chronological (grouped by category, chronological within)
  - Agent B: Reverse chronological
  - Agent C: Random shuffle

Each split 80/10/10 into train/valid/test.
"""

import json
import os
import random
import re
import sys
import glob
from collections import defaultdict
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────
RAW_DATA_DIR = "data/raw"
OUTPUT_BASE = "data/training"
SEED = 42
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# ─── Keyword-based sentiment classification ──────────────────────────
BULLISH_KEYWORDS = [
    "rally", "surge", "soar", "all-time high", "ath", "breakout", "adoption",
    "approve", "partnership", "launch", "institutional", "accumulation",
    "upgrade", "mainnet", "record", "bullish", "gains", "pump", "moon",
    "spike", "jump", "milestone", "inflow", "buy", "growth",
]
BEARISH_KEYWORDS = [
    "crash", "plunge", "dump", "hack", "exploit", "ban", "crackdown",
    "sec charges", "lawsuit", "fraud", "scam", "collapse", "bankruptcy",
    "liquidat", "freeze", "halt", "suspend", "bearish", "drop", "fall",
    "decline", "sell-off", "selloff", "fear", "panic", "loss", "theft",
    "vulnerable", "attack", "rug pull", "rugpull", "ponzi",
]
NEUTRAL_KEYWORDS = [
    "report", "analysis", "update", "review", "regulation", "stablecoin",
    "development", "roadmap", "announce", "plan", "consider", "explore",
    "discuss", "proposal",
]

# ─── Event category classification ───────────────────────────────────
CATEGORY_KEYWORDS = {
    "regulation": [
        "regulation", "regulate", "sec", "cftc", "law", "legal", "compliance",
        "policy", "government", "ban", "crackdown", "sanction", "license",
        "framework", "legislation", "congress", "senate", "bill",
    ],
    "security": [
        "hack", "exploit", "breach", "vulnerability", "attack", "theft",
        "steal", "stolen", "drain", "rug pull", "rugpull", "scam", "fraud",
        "phishing", "malware", "ransomware",
    ],
    "price_action": [
        "rally", "surge", "crash", "plunge", "all-time high", "ath",
        "breakout", "dump", "pump", "price", "market cap", "volume",
        "liquidat", "bull", "bear", "correction", "dip",
    ],
    "defi": [
        "defi", "decentralized finance", "yield", "lending", "borrow",
        "liquidity", "amm", "dex", "swap", "pool", "stake", "staking",
        "farm", "vault", "protocol", "tvl",
    ],
    "adoption": [
        "adoption", "accept", "payment", "partner", "partnership",
        "institutional", "launch", "integrate", "mainstream", "retail",
        "merchant", "corporate", "enterprise",
    ],
    "technology": [
        "upgrade", "mainnet", "fork", "merge", "layer 2", "l2",
        "scalability", "throughput", "consensus", "proof of", "rollup",
        "shard", "bridge", "interop", "cross-chain",
    ],
    "macro": [
        "fed", "federal reserve", "interest rate", "inflation", "cpi",
        "gdp", "recession", "economy", "monetary", "fiscal", "treasury",
        "dollar", "fomc",
    ],
    "nft_metaverse": [
        "nft", "metaverse", "opensea", "gaming", "play-to-earn", "p2e",
        "virtual land", "avatar", "collectible",
    ],
    "exchange": [
        "binance", "coinbase", "kraken", "ftx", "exchange", "listing",
        "delist", "withdrawal", "deposit", "trading pair",
    ],
    "stablecoin": [
        "stablecoin", "usdt", "usdc", "dai", "tether", "peg", "depeg",
    ],
}

# ─── Prompt templates for variety ────────────────────────────────────
SENTIMENT_PROMPTS = [
    'Analyze this crypto news from {date}: "{title}". What is the sentiment and likely market impact?',
    'What is the market sentiment of this {date} crypto headline: "{title}"? Assess the potential impact.',
    'Read this crypto news ({date}): "{title}". Classify the sentiment and predict short-term market effect.',
    'Evaluate the sentiment and trading implications of this {date} headline: "{title}".',
    'From a trading perspective, analyze the sentiment of: "{title}" ({date}).',
]

CONTEXT_PROMPTS_WITH_BODY = [
    'You are a crypto market analyst. On {date}, this news broke: "{title}". The article states: {body} What does this tell us about the current market regime and what should a trader do?',
    'As a crypto analyst, interpret this {date} news: "{title}". Context: {body} What are the trading implications?',
    'Analyze the market significance of this report from {date}: "{title}". Details: {body} What should traders watch for?',
    'On {date}, this was reported: "{title}". Summary: {body} Break down the market implications and recommended trading action.',
]

CONTEXT_PROMPTS_NO_BODY = [
    'You are a crypto market analyst. On {date}, this news broke: "{title}". What does this tell us about the current market regime and what should a trader do?',
    'As a crypto analyst, interpret this {date} headline: "{title}". What are the trading implications?',
    'Analyze the market significance of this {date} headline: "{title}". What should traders watch for?',
    'On {date}, this headline appeared: "{title}". Break down the market implications and recommended trading action.',
]

EVENT_PROMPTS = [
    'Classify this crypto event: "{title}" ({date}). Category, severity, and expected duration of impact?',
    'Categorize and assess: "{title}" from {date}. What type of event is this, how severe, and how long will it affect markets?',
    'Event analysis for "{title}" ({date}): What category does this fall into, what severity level, and what is the expected impact duration?',
    'Provide an event classification for this {date} headline: "{title}". Include category, severity rating, and duration of market impact.',
]


def classify_sentiment(title: str, body: str) -> str:
    text = (title + " " + body).lower()
    bullish_score = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
    bearish_score = sum(1 for kw in BEARISH_KEYWORDS if kw in text)

    if bullish_score > bearish_score + 1:
        return "bullish"
    elif bullish_score > bearish_score:
        return "slightly_bullish"
    elif bearish_score > bullish_score + 1:
        return "bearish"
    elif bearish_score > bullish_score:
        return "slightly_bearish"
    else:
        return "neutral"


def classify_category(title: str, body: str) -> str:
    text = (title + " " + body).lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general"
    return best


def severity_from_sentiment(sentiment: str) -> str:
    if sentiment in ("bullish", "bearish"):
        return "high"
    elif sentiment in ("slightly_bullish", "slightly_bearish"):
        return "moderate"
    return "low"


def duration_from_category(category: str) -> str:
    durations = {
        "regulation": "weeks to months depending on regulatory specifics",
        "security": "days to weeks depending on severity and funds affected",
        "price_action": "hours to days, driven by momentum and follow-through",
        "defi": "days to weeks depending on protocol significance",
        "adoption": "weeks to months as adoption effects compound",
        "technology": "days to weeks for immediate impact, months for full effect",
        "macro": "weeks to months as macro trends unfold",
        "nft_metaverse": "days to weeks, often short-lived hype cycles",
        "exchange": "hours to days for exchange-specific events",
        "stablecoin": "hours to days if peg risk, weeks if regulatory",
        "general": "varies depending on specifics",
    }
    return durations.get(category, "varies depending on specifics")


def generate_sentiment_completion(title: str, body: str, sentiment: str, category: str) -> str:
    severity = severity_from_sentiment(sentiment)
    sentiment_label = sentiment.replace("_", " ")

    impact_map = {
        "bullish": "Strong positive signal. Expect upward price pressure and potential momentum continuation. Traders should watch for volume confirmation before entering. Likely to attract more buying interest.",
        "slightly_bullish": "Moderately positive. Market may see a small uptick but the signal is not strong enough alone to drive a sustained move. Watch for confirmation from other indicators.",
        "bearish": "Significant negative signal. Expect selling pressure and potential cascade of liquidations. Risk-off positioning recommended. Watch for support levels and potential dead cat bounces.",
        "slightly_bearish": "Mildly negative signal. May cause short-term weakness but unlikely to trigger a major move on its own. Monitor for additional bearish catalysts that could amplify this.",
        "neutral": "Informational — no strong directional bias. Market impact depends on subsequent developments. This is more of a background event that traders should monitor but not react to immediately.",
    }

    base = impact_map.get(sentiment, impact_map["neutral"])

    category_notes = {
        "regulation": "Regulatory news tends to have outsized impact on sentiment and can set the tone for weeks.",
        "security": "Security events erode confidence and often cause broader market fear beyond the affected protocol.",
        "price_action": "Price-driven headlines tend to be lagging indicators — the move has often already started.",
        "defi": "DeFi-specific events may have concentrated impact on related tokens and protocols.",
        "adoption": "Adoption news builds long-term bullish narrative but rarely moves markets sharply in the short term.",
        "technology": "Technical upgrades are typically priced in before execution; watch for sell-the-news patterns.",
        "macro": "Macro events affect all risk assets including crypto — watch correlation with equities.",
        "exchange": "Exchange-specific news can cause rapid short-term volatility, especially around listings and delistings.",
    }
    note = category_notes.get(category, "")

    return f"Sentiment: {sentiment_label}. {base} {note}".strip()


def generate_context_completion(title: str, body: str, sentiment: str, category: str, date: str) -> str:
    regime_hints = {
        "bullish": "The market appears to be in a constructive phase with buying interest. Traders should maintain long exposure but use trailing stops to protect gains.",
        "slightly_bullish": "The market is leaning positive but caution is warranted. A trader should hold current positions with moderate stops and selectively add on dips.",
        "bearish": "This signals a risk-off environment. A trader should reduce exposure, tighten stops on existing positions, and consider increasing stablecoin allocation until momentum confirms a reversal.",
        "slightly_bearish": "The market shows some weakness. A trader should avoid adding new positions, tighten stops on existing trades, and watch for further deterioration.",
        "neutral": "The market lacks clear direction. A trader should stay patient, maintain current positions with reasonable stops, and wait for a clearer signal before making moves.",
    }

    base_advice = regime_hints.get(sentiment, regime_hints["neutral"])

    if body and body not in ("NULL", "-", ""):
        context_addition = f"The additional context — {body[:200]} — provides nuance to the headline."
    else:
        context_addition = "Without additional context, the headline alone suggests monitoring for follow-up developments."

    category_advice = {
        "regulation": "Regulation headlines often create uncertainty. Watch for specifics — vague threats cause more fear than actual rules. Markets historically adapt to regulatory clarity within 2-4 weeks.",
        "security": "Security incidents require immediate risk assessment. Check if the affected protocol has significant TVL and whether the exploit is isolated or systemic. Expect contagion fear in the short term.",
        "price_action": "Price action headlines confirm what charts already show. Focus on whether the move has exhausted itself (check volume, RSI, funding rates) or still has momentum.",
        "defi": "DeFi events can create opportunities in related protocols. Check for yield changes, liquidity shifts, and potential cascading effects across interconnected protocols.",
        "adoption": "Adoption milestones build the long-term bull case. They rarely cause immediate moves but shift the baseline sentiment. Good for accumulation strategies.",
        "technology": "Technical developments are fundamental catalysts. Pre-upgrade usually bullish, post-upgrade often sell-the-news. Check if the upgrade addresses real bottlenecks.",
        "macro": "Macro events ripple through all markets. Crypto often moves in tandem with risk assets during macro shifts. Watch BTC dominance for rotation signals.",
        "exchange": "Exchange events can disrupt trading patterns. Watch for unusual volume and spread changes across venues.",
    }
    cat_advice = category_advice.get(category, "Monitor for additional signals to confirm direction.")

    return f"{context_addition} {base_advice} {cat_advice}"


def generate_event_completion(title: str, body: str, sentiment: str, category: str) -> str:
    severity = severity_from_sentiment(sentiment)
    duration = duration_from_category(category)

    severity_details = {
        "high": "This is a high-impact event that could cause 5-20% moves in affected assets. Broad market impact likely.",
        "moderate": "Moderate impact expected — 2-10% moves possible in affected assets. Broader market may see some volatility.",
        "low": "Low severity — unlikely to cause significant market moves on its own but worth monitoring for cumulative effect.",
    }
    detail = severity_details.get(severity, severity_details["low"])

    return f"Category: {category}. Severity: {severity}. Expected duration: {duration}. {detail}"


RECORD_START_RE = re.compile(r"^[a-zA-Z0-9_]+\|\d{10,}\|")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return HTML_TAG_RE.sub("", text).strip()


def read_records(filepath: str):
    """Read a file and yield complete records, handling multi-line bodies."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        current = None
        for line in f:
            if RECORD_START_RE.match(line):
                if current is not None:
                    yield current
                current = line.rstrip("\n")
            else:
                # Continuation of previous record's body
                if current is not None:
                    current += " " + line.strip()
        if current is not None:
            yield current


def parse_line(line: str):
    """Parse a pipe-delimited line. Returns dict or None if invalid."""
    parts = line.split("|")
    if len(parts) < 10:
        return None

    record_id = parts[0].strip()
    timestamp_ms = parts[1].strip()
    date = parts[5].strip()
    source = parts[6].strip()
    title = parts[7].strip()
    body = strip_html(parts[8].strip()) if len(parts) > 8 else ""
    url = parts[9].strip() if len(parts) > 9 else ""

    # Filtering rules
    if len(title) < 15:
        return None
    if title.startswith("RT @") and len(title) < 50:
        return None
    if body in ("NULL", "-", ""):
        body = ""

    # Parse timestamp
    try:
        ts = int(timestamp_ms)
    except (ValueError, TypeError):
        ts = 0

    if not date:
        if ts > 0:
            date = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        else:
            return None

    return {
        "id": record_id,
        "timestamp_ms": ts,
        "date": date,
        "source": source,
        "title": title,
        "body": body,
        "url": url,
    }


def generate_pairs(record: dict) -> list:
    """Generate 3 training pairs for a single record."""
    title = record["title"]
    body = record["body"]
    date = record["date"]

    sentiment = classify_sentiment(title, body)
    category = classify_category(title, body)

    pairs = []

    # Type 1: Sentiment Analysis
    prompt_template = random.choice(SENTIMENT_PROMPTS)
    prompt = prompt_template.format(date=date, title=title)
    completion = generate_sentiment_completion(title, body, sentiment, category)
    pairs.append({
        "text": f"<s>[INST] {prompt} [/INST] {completion} </s>",
        "category": category,
        "timestamp_ms": record["timestamp_ms"],
    })

    # Type 2: Market Context Analysis
    if body:
        prompt_template = random.choice(CONTEXT_PROMPTS_WITH_BODY)
        prompt = prompt_template.format(date=date, title=title, body=body)
    else:
        prompt_template = random.choice(CONTEXT_PROMPTS_NO_BODY)
        prompt = prompt_template.format(date=date, title=title)
    completion = generate_context_completion(title, body, sentiment, category, date)
    pairs.append({
        "text": f"<s>[INST] {prompt} [/INST] {completion} </s>",
        "category": category,
        "timestamp_ms": record["timestamp_ms"],
    })

    # Type 3: Event Classification
    prompt_template = random.choice(EVENT_PROMPTS)
    prompt = prompt_template.format(date=date, title=title)
    completion = generate_event_completion(title, body, sentiment, category)
    pairs.append({
        "text": f"<s>[INST] {prompt} [/INST] {completion} </s>",
        "category": category,
        "timestamp_ms": record["timestamp_ms"],
    })

    return pairs


def split_data(pairs: list) -> tuple:
    """Split into train/valid/test at 80/10/10."""
    n = len(pairs)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)
    return pairs[:train_end], pairs[train_end:valid_end], pairs[valid_end:]


def write_jsonl(pairs: list, filepath: str):
    """Write pairs to a JSONL file (only the 'text' field)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps({"text": pair["text"]}, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)

    # Find all raw data files
    data_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.txt")))
    if not data_files:
        print(f"ERROR: No .txt files found in {RAW_DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(data_files)} data files:")
    for f in data_files:
        print(f"  {f}")

    # ─── Phase 1: Parse all records ──────────────────────────────────
    all_records = []
    total_records_read = 0
    skipped = 0
    encoding_errors = 0

    for filepath in data_files:
        filename = os.path.basename(filepath)
        file_records = 0
        file_skipped = 0
        for raw_record in read_records(filepath):
            total_records_read += 1
            try:
                record = parse_line(raw_record)
                if record:
                    all_records.append(record)
                    file_records += 1
                else:
                    file_skipped += 1
                    skipped += 1
            except Exception:
                encoding_errors += 1
        print(f"  {filename}: {file_records} valid, {file_skipped} skipped")

    print(f"\n--- Parsing Summary ---")
    print(f"Total records read:   {total_records_read}")
    print(f"Valid records:        {len(all_records)}")
    print(f"Skipped (filtered):   {skipped}")
    print(f"Encoding errors:      {encoding_errors}")

    # ─── Phase 2: Generate training pairs ────────────────────────────
    print(f"\nGenerating training pairs (3 per article)...")
    all_pairs = []
    for i, record in enumerate(all_records):
        pairs = generate_pairs(record)
        all_pairs.extend(pairs)
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i + 1}/{len(all_records)} articles ({len(all_pairs)} pairs so far)")

    print(f"Total training pairs: {len(all_pairs)}")

    # ─── Phase 3: Agent A — Category-chronological ───────────────────
    print(f"\n--- Agent A: Category-chronological ordering ---")
    by_category = defaultdict(list)
    for pair in all_pairs:
        by_category[pair["category"]].append(pair)

    # Sort each category by timestamp
    for cat in by_category:
        by_category[cat].sort(key=lambda x: x["timestamp_ms"])

    print(f"  Categories found: {list(by_category.keys())}")
    for cat, items in sorted(by_category.items()):
        print(f"    {cat}: {len(items)} pairs")

    # Interleave categories round-robin
    agent_a_pairs = []
    category_iters = {cat: iter(pairs) for cat, pairs in sorted(by_category.items())}
    exhausted = set()
    while len(exhausted) < len(category_iters):
        for cat in sorted(category_iters.keys()):
            if cat in exhausted:
                continue
            try:
                agent_a_pairs.append(next(category_iters[cat]))
            except StopIteration:
                exhausted.add(cat)

    train_a, valid_a, test_a = split_data(agent_a_pairs)
    write_jsonl(train_a, os.path.join(OUTPUT_BASE, "agent_a", "train.jsonl"))
    write_jsonl(valid_a, os.path.join(OUTPUT_BASE, "agent_a", "valid.jsonl"))
    write_jsonl(test_a, os.path.join(OUTPUT_BASE, "agent_a", "test.jsonl"))
    print(f"  Agent A: train={len(train_a)}, valid={len(valid_a)}, test={len(test_a)}")

    # ─── Phase 4: Agent B — Reverse chronological ────────────────────
    print(f"\n--- Agent B: Reverse chronological ordering ---")
    agent_b_pairs = sorted(all_pairs, key=lambda x: x["timestamp_ms"], reverse=True)
    train_b, valid_b, test_b = split_data(agent_b_pairs)
    write_jsonl(train_b, os.path.join(OUTPUT_BASE, "agent_b", "train.jsonl"))
    write_jsonl(valid_b, os.path.join(OUTPUT_BASE, "agent_b", "valid.jsonl"))
    write_jsonl(test_b, os.path.join(OUTPUT_BASE, "agent_b", "test.jsonl"))
    print(f"  Agent B: train={len(train_b)}, valid={len(valid_b)}, test={len(test_b)}")

    # ─── Phase 5: Agent C — Random shuffle ───────────────────────────
    print(f"\n--- Agent C: Random shuffle ordering ---")
    agent_c_pairs = all_pairs.copy()
    random.shuffle(agent_c_pairs)
    train_c, valid_c, test_c = split_data(agent_c_pairs)
    write_jsonl(train_c, os.path.join(OUTPUT_BASE, "agent_c", "train.jsonl"))
    write_jsonl(valid_c, os.path.join(OUTPUT_BASE, "agent_c", "valid.jsonl"))
    write_jsonl(test_c, os.path.join(OUTPUT_BASE, "agent_c", "test.jsonl"))
    print(f"  Agent C: train={len(train_c)}, valid={len(valid_c)}, test={len(test_c)}")

    # ─── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Articles processed:     {len(all_records)}")
    print(f"Training pairs total:   {len(all_pairs)}")
    print(f"Pairs per agent:        {len(all_pairs)}")
    print(f"Total across 3 agents:  {len(all_pairs) * 3}")
    print(f"")
    print(f"Output directories:")
    print(f"  Agent A (category-chrono): {os.path.join(OUTPUT_BASE, 'agent_a')}/")
    print(f"  Agent B (reverse-chrono):  {os.path.join(OUTPUT_BASE, 'agent_b')}/")
    print(f"  Agent C (random shuffle):  {os.path.join(OUTPUT_BASE, 'agent_c')}/")
    print(f"")
    print(f"Next steps:")
    print(f"  1. Fine-tune Agent A: python -m mlx_lm.lora --data ./data/training/agent_a ...")
    print(f"  2. Fine-tune Agent B: python -m mlx_lm.lora --data ./data/training/agent_b ...")
    print(f"  3. Fine-tune Agent C: python -m mlx_lm.lora --data ./data/training/agent_c ...")


if __name__ == "__main__":
    main()

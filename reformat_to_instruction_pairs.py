#!/usr/bin/env python3
"""
Reformat train_combined.jsonl files from nested JSON → instruction/response text pairs.
Processes all three agents (agent_a, agent_b, agent_c).
Output: one JSONL per agent with {"instruction": "...", "response": "..."} per line.
"""

import json
from pathlib import Path

TRAINING_DIR = Path('data/training')
AGENT_ROLES = {
    'agent_a': 'Technical Analyst',
    'agent_b': 'Fundamental Analyst',
    'agent_c': 'Risk Manager',
}


def format_instruction(entry: dict, role: str) -> str:
    """Build a natural-language instruction from the nested JSON."""
    fin = entry.get('financial', {})
    art = entry.get('article', {})
    date = entry.get('date', 'unknown')

    price = fin.get('btc_price', 0)
    change_24h = fin.get('btc_change_24h', 0) * 100  # decimal → %
    change_7d = fin.get('btc_change_7d', 0) * 100
    volume = fin.get('btc_volume', 0)
    ma7 = fin.get('btc_ma_7', 0)
    ma30 = fin.get('btc_ma_30', 0)
    rsi = fin.get('btc_rsi', 0)
    volatility = fin.get('btc_volatility', 0) * 100

    title = art.get('title', '')
    content = art.get('content', '')

    instruction = (
        f"You are a crypto {role}. Analyze the following market snapshot for {date} "
        f"and provide your assessment.\n\n"
        f"Market Data:\n"
        f"- BTC Price: ${price:,.2f}\n"
        f"- 24h Change: {change_24h:+.2f}%\n"
        f"- 7d Change: {change_7d:+.2f}%\n"
        f"- 24h Volume: ${volume:,.0f}\n"
        f"- 7-day MA: ${ma7:,.2f}\n"
        f"- 30-day MA: ${ma30:,.2f}\n"
        f"- RSI: {rsi:.1f}\n"
        f"- Volatility: {volatility:.2f}%\n\n"
        f"Latest News:\n"
        f"Headline: {title}\n"
        f"Summary: {content}"
    )
    return instruction


def format_response(entry: dict, role: str) -> str:
    """Build a structured response from the article metadata."""
    fin = entry.get('financial', {})
    art = entry.get('article', {})

    sentiment = art.get('sentiment', 'neutral')
    categories = art.get('categories', '')
    coins = art.get('coins', '')

    price = fin.get('btc_price', 0)
    rsi = fin.get('btc_rsi', 50)
    change_24h = fin.get('btc_change_24h', 0) * 100
    volatility = fin.get('btc_volatility', 0) * 100

    # Derive a simple signal from the data
    if rsi < 35 and change_24h < -2:
        signal = "BUY — oversold conditions with potential for mean reversion"
    elif rsi > 70 and change_24h > 3:
        signal = "CAUTION — overbought, risk of pullback"
    elif sentiment == 'bullish':
        signal = "LEAN LONG — bullish sentiment with supportive fundamentals"
    elif sentiment == 'bearish':
        signal = "LEAN SHORT / WAIT — bearish sentiment, wait for confirmation"
    else:
        signal = "NEUTRAL / WAIT — mixed signals, monitor for clearer setup"

    response = (
        f"Sentiment: {sentiment.capitalize()}\n"
        f"Categories: {categories}\n"
        f"Relevant Coins: {coins}\n\n"
        f"Signal: {signal}\n\n"
        f"Key Observations:\n"
        f"- RSI at {rsi:.1f} {'(oversold)' if rsi < 35 else '(overbought)' if rsi > 70 else '(neutral range)'}\n"
        f"- 24h momentum: {change_24h:+.2f}%\n"
        f"- Volatility: {volatility:.2f}%\n"
        f"- Price vs 7d MA: {'above' if price > fin.get('btc_ma_7', price) else 'below'} "
        f"(${abs(price - fin.get('btc_ma_7', price)):,.2f} delta)\n"
        f"- Price vs 30d MA: {'above' if price > fin.get('btc_ma_30', price) else 'below'} "
        f"(${abs(price - fin.get('btc_ma_30', price)):,.2f} delta)"
    )
    return response


def process_agent(agent_name: str, role: str) -> int:
    input_file = TRAINING_DIR / agent_name / 'train_combined.jsonl'
    output_file = TRAINING_DIR / agent_name / 'train_instruction_pairs.jsonl'

    if not input_file.exists():
        print(f"  ⚠️  {input_file} not found, skipping")
        return 0

    count = 0
    errors = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                pair = {
                    'instruction': format_instruction(entry, role),
                    'response': format_response(entry, role),
                }
                fout.write(json.dumps(pair, ensure_ascii=False) + '\n')
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                errors += 1
                if errors <= 3:
                    print(f"  ⚠️  Line {line_num}: {e}")

    print(f"  ✅ {count} pairs written → {output_file}")
    if errors:
        print(f"  ⚠️  {errors} lines skipped due to errors")
    return count


def main():
    print("=" * 60)
    print("Reformatting train_combined.jsonl → instruction/response pairs")
    print("=" * 60)

    total = 0
    for agent, role in AGENT_ROLES.items():
        print(f"\n🔄 {agent} ({role})")
        total += process_agent(agent, role)

    print(f"\n{'=' * 60}")
    print(f"✅ Done — {total} total instruction/response pairs created")
    print("=" * 60)


if __name__ == '__main__':
    main()

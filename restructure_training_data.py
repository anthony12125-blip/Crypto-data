#!/usr/bin/env python3
"""
Restructure Iron Dragoon training data for decision-making
Converts sentiment analysis format → trade decision format
"""

import json
import random
from pathlib import Path
import re

TRAINING_DIR = Path('data/training')
OUTPUT_DIR = Path('data/training_restructured')

def parse_existing_format(line):
    """Extract fields from existing Mistral format"""
    try:
        data = json.loads(line)
        text = data.get('text', '')
        
        # Extract components using regex
        # Format: <s>[INST] Analyze crypto market conditions on DATE: 'TITLE' Market Context: ... [/INST] Sentiment: X. Analysis: ... </s>
        
        date_match = re.search(r"on (\d{4}-\d{2}-\d{2})", text)
        title_match = re.search(r"'([^']+)'", text)
        
        # Extract market context
        btc_price_match = re.search(r"BTC Price: \$(\d+)", text)
        change_match = re.search(r"24h Change: ([+-]?\d+\.?\d*)%", text)
        rsi_match = re.search(r"RSI: (\d+\.?\d*)", text)
        vol_match = re.search(r"Volatility: (\d+\.?\d*)%", text)
        
        # Extract sentiment and analysis
        sentiment_match = re.search(r"Sentiment: (\w+)", text)
        analysis_match = re.search(r"Analysis: (.+?)(?:Categories:|Related Assets:|$)", text, re.DOTALL)
        categories_match = re.search(r"Categories: ([^\n]+)", text)
        
        return {
            'date': date_match.group(1) if date_match else '2024-01-01',
            'title': title_match.group(1) if title_match else 'Crypto News',
            'btc_price': int(btc_price_match.group(1)) if btc_price_match else 50000,
            'btc_change': float(change_match.group(1)) if change_match else 0.0,
            'rsi': float(rsi_match.group(1)) if rsi_match else 50.0,
            'volatility': float(vol_match.group(1)) if vol_match else 2.0,
            'sentiment': sentiment_match.group(1).lower() if sentiment_match else 'neutral',
            'analysis': analysis_match.group(1).strip() if analysis_match else '',
            'categories': categories_match.group(1).strip() if categories_match else 'general'
        }
    except Exception as e:
        return None

def generate_trade_decision(parsed):
    """Generate realistic trade decision based on market conditions"""
    
    # Decision logic based on conditions
    rsi = parsed['rsi']
    change = parsed['btc_change']
    volatility = parsed['volatility']
    sentiment = parsed['sentiment']
    
    # Scoring for decision
    score = 50  # Neutral baseline
    
    # RSI component (30 points)
    if rsi < 30:
        score += 25  # Strong oversold bounce potential
    elif rsi < 40:
        score += 15  # Oversold
    elif rsi > 70:
        score -= 20  # Overbought
    elif rsi > 60:
        score -= 10  # Getting overbought
    
    # Price momentum component (20 points)
    if change < -5:
        score += 10  # Big dip = potential bounce
    elif change < -3:
        score += 5
    elif change > 5:
        score -= 10  # Extended, caution
    elif change > 3:
        score -= 5
    
    # Volatility component (15 points)
    if volatility > 5:
        score += 10  # High vol = opportunity
    elif volatility > 3:
        score += 5
    elif volatility < 1.5:
        score -= 5  # Too quiet
    
    # Sentiment component (15 points)
    if sentiment == 'bullish':
        score += 10
    elif sentiment == 'bearish':
        score -= 10
    
    # Determine action based on score
    if score >= 65:
        action = 'BUY'
        confidence = min(95, 60 + (score - 65) * 1.5)
    elif score >= 45:
        action = 'WAIT'
        confidence = 50 + (score - 45)
    else:
        action = 'SKIP'
        confidence = max(30, 50 - (45 - score))
    
    # Generate trade parameters based on action
    btc_price = parsed['btc_price']
    
    if action == 'BUY':
        stop_loss = btc_price * 0.97  # 3% stop
        take_profit = btc_price * 1.05  # 5% target
        position_size = min(33, max(15, int(score / 3)))
        horizon = '3-7 days'
        
        reasoning = generate_buy_reasoning(parsed, score)
        risk_flags = generate_risk_flags(parsed)
        
    elif action == 'WAIT':
        stop_loss = btc_price * 0.96
        take_profit = btc_price * 1.04
        position_size = 20
        horizon = 'Monitor for 24-48h'
        
        reasoning = generate_wait_reasoning(parsed, score)
        risk_flags = generate_risk_flags(parsed)
        
    else:  # SKIP
        stop_loss = btc_price * 0.95
        take_profit = btc_price * 1.03
        position_size = 0
        horizon = 'N/A'
        
        reasoning = generate_skip_reasoning(parsed, score)
        risk_flags = generate_risk_flags(parsed)
    
    return {
        'action': action,
        'confidence': round(confidence),
        'entry': btc_price,
        'stop_loss': round(stop_loss, 2),
        'take_profit': round(take_profit, 2),
        'position_size': position_size,
        'horizon': horizon,
        'reasoning': reasoning,
        'risk_flags': risk_flags
    }

def generate_buy_reasoning(parsed, score):
    """Generate BUY reasoning"""
    reasons = []
    
    if parsed['rsi'] < 35:
        reasons.append(f"RSI {parsed['rsi']:.0f} indicates oversold conditions with mean reversion potential")
    elif parsed['rsi'] < 45:
        reasons.append(f"RSI {parsed['rsi']:.0f} showing early momentum recovery")
    
    if parsed['btc_change'] < -3:
        reasons.append(f"Price down {abs(parsed['btc_change']):.1f}% presents dip-buying opportunity")
    
    if parsed['volatility'] > 4:
        reasons.append(f"Elevated volatility ({parsed['volatility']:.1f}%) creating entry windows")
    
    if 'adoption' in parsed['categories'].lower() or 'institutional' in parsed['categories'].lower():
        reasons.append("Positive fundamental catalyst in adoption/institutional category")
    
    if parsed['sentiment'] == 'bullish':
        reasons.append("Market sentiment aligned with bullish structure")
    
    if not reasons:
        reasons.append("Technical setup shows favorable risk/reward profile")
    
    return " ".join(reasons[:3])  # Top 3 reasons

def generate_wait_reasoning(parsed, score):
    """Generate WAIT reasoning"""
    reasons = []
    
    if 45 <= parsed['rsi'] <= 55:
        reasons.append(f"RSI {parsed['rsi']:.0f} in neutral zone - await directional confirmation")
    
    if abs(parsed['btc_change']) < 2:
        reasons.append("Low momentum environment - patience for breakout")
    
    if 'regulation' in parsed['categories'].lower() or 'security' in parsed['categories'].lower():
        reasons.append("Regulatory uncertainty requires monitoring before entry")
    
    if parsed['volatility'] > 6:
        reasons.append(f"Extreme volatility ({parsed['volatility']:.1f}%) suggests waiting for stabilization")
    
    if not reasons:
        reasons.append("Setup developing but needs confirmation - monitor for trigger")
    
    return " ".join(reasons[:2])

def generate_skip_reasoning(parsed, score):
    """Generate SKIP reasoning"""
    reasons = []
    
    if parsed['rsi'] > 70:
        reasons.append(f"RSI {parsed['rsi']:.0f} overbought - avoid chasing")
    elif parsed['rsi'] > 65:
        reasons.append(f"RSI {parsed['rsi']:.0f} extended - risk of pullback high")
    
    if parsed['btc_change'] > 8:
        reasons.append(f"Price up {parsed['btc_change']:.1f}% - extended, wait for consolidation")
    
    if 'hack' in parsed['analysis'].lower() or 'security' in parsed['categories'].lower():
        reasons.append("Security incident in news - avoid until clarity")
    
    if parsed['sentiment'] == 'bearish' and parsed['btc_change'] < -5:
        reasons.append("Bearish sentiment with accelerating downside - don't catch falling knife")
    
    if not reasons:
        reasons.append("Risk/reward unfavorable - better opportunities elsewhere")
    
    return " ".join(reasons[:2])

def generate_risk_flags(parsed):
    """Generate risk flags based on conditions"""
    flags = []
    
    if parsed['volatility'] > 6:
        flags.append("High volatility - reduce position size")
    
    if 'regulation' in parsed['categories'].lower():
        flags.append("Regulatory headline risk")
    
    if 'hack' in parsed['analysis'].lower() or 'exploit' in parsed['analysis'].lower():
        flags.append("Security incident reported")
    
    if parsed['rsi'] > 75 or parsed['rsi'] < 25:
        flags.append("Extreme RSI - potential reversal zone")
    
    if abs(parsed['btc_change']) > 10:
        flags.append("Large daily move - volatility risk")
    
    return flags if flags else ["None critical"]

def restructure_example(line, agent_type='technician'):
    """Convert one training example to new format"""
    parsed = parse_existing_format(line)
    if not parsed:
        return None
    
    decision = generate_trade_decision(parsed)
    
    # Build instruction (input)
    instruction = f"""Market Snapshot:
- Symbol: BTC/USD
- Price: ${parsed['btc_price']:,} ({parsed['btc_change']:+.1f}% 24h)
- RSI: {parsed['rsi']:.0f}
- Volatility: {parsed['volatility']:.1f}%
- News: "{parsed['title'][:80]}"

Agent Role: {agent_type}
Evaluate trade opportunity."""

    # Build response (output)
    risk_flags_text = "; ".join(decision['risk_flags'])
    
    response = f"""Action: {decision['action']}

Trade Card:
- Entry: ${decision['entry']:,}
- Stop Loss: ${decision['stop_loss']:,.2f}
- Take Profit: ${decision['take_profit']:,.2f}
- Position Size: {decision['position_size']}% max
- Time Horizon: {decision['horizon']}

Reasoning:
{decision['reasoning']}

Risk Flags: {risk_flags_text}
Confidence: {decision['confidence']}/100"""

    return {
        'text': f"<s>[INST] {instruction} [/INST] {response} </s>"
    }

def process_agent(agent_name, agent_type, sample_size=100):
    """Process training data for one agent"""
    input_file = TRAINING_DIR / agent_name / 'train.jsonl'
    output_dir = OUTPUT_DIR / agent_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'train.jsonl'
    
    print(f"\n🔄 Processing {agent_name} ({agent_type})...")
    
    processed = 0
    skipped = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # Sample for demonstration
        lines = f_in.readlines()
        sample_lines = random.sample(lines, min(sample_size, len(lines)))
        
        for line in sample_lines:
            restructured = restructure_example(line, agent_type)
            if restructured:
                f_out.write(json.dumps(restructured, ensure_ascii=False) + '\n')
                processed += 1
            else:
                skipped += 1
    
    print(f"  ✅ Processed: {processed} | ⏭️ Skipped: {skipped}")
    print(f"  📄 Output: {output_file}")
    
    return processed

def show_samples():
    """Show sample restructured examples"""
    print("\n" + "="*80)
    print("SAMPLE RESTRUCTURED EXAMPLES")
    print("="*80)
    
    sample_file = OUTPUT_DIR / 'agent_a' / 'train.jsonl'
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            samples = [json.loads(line) for line in f.readlines()[:3]]
        
        for i, sample in enumerate(samples, 1):
            print(f"\n--- Example {i} ---")
            print(sample['text'][:1500] + "..." if len(sample['text']) > 1500 else sample['text'])
            print()

if __name__ == "__main__":
    print("="*80)
    print("IRON DRAGOON TRAINING DATA RESTRUCTURING")
    print("="*80)
    print("\nConverting sentiment analysis → trade decision format")
    print("Processing sample of 100 examples per agent...")
    
    # Process each agent
    agents = [
        ('agent_a', 'Technician'),
        ('agent_b', 'Fundamentalist'),
        ('agent_c', 'Risk Manager')
    ]
    
    total = 0
    for agent, agent_type in agents:
        count = process_agent(agent, agent_type, sample_size=100)
        total += count
    
    print("\n" + "="*80)
    print(f"✅ Restructured {total} training examples")
    print("="*80)
    
    # Show samples
    show_samples()
    
    # Create manifest
    manifest = {
        'restructured_date': '2026-02-19',
        'format': 'Mistral instruction tuning - Trade Decision Format',
        'agents': {
            'agent_a': {'type': 'Technician', 'focus': 'Technical analysis, RSI, momentum'},
            'agent_b': {'type': 'Fundamentalist', 'focus': 'News catalysts, sentiment'},
            'agent_c': {'type': 'Risk Manager', 'focus': 'Risk assessment, position sizing'}
        },
        'output_structure': {
            'Action': 'BUY/WAIT/SKIP',
            'Trade Card': 'Entry, Stop Loss, Take Profit, Position Size, Horizon',
            'Reasoning': 'Plain English explanation',
            'Risk Flags': 'Potential issues',
            'Confidence': '0-100 score'
        },
        'sample_size': 100,
        'note': 'Full dataset (137K per agent) can be processed by removing sample_size limit'
    }
    
    with open(OUTPUT_DIR / 'restructure_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📄 Manifest: {OUTPUT_DIR / 'restructure_manifest.json'}")
    print("\nTo process full dataset, edit sample_size parameter")

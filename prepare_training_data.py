#!/usr/bin/env python3
"""
PROJECT IRON DRAGOON - Complete Training Data Preparation
Combines YOUR existing crypto article data with financial price data
"""

import os
import json
import sqlite3
import zipfile
import random
import re
from datetime import datetime
from pathlib import Path
import pandas as pd

# Configuration
DATA_DIR = Path('data')
FINANCIAL_DIR = DATA_DIR / 'financial'
TRAINING_DIR = DATA_DIR / 'training'
CRYPTO_DB = 'crypto_chronological_master.db'

# Ensure directories exist
for d in [TRAINING_DIR / 'agent_a', TRAINING_DIR / 'agent_b', TRAINING_DIR / 'agent_c']:
    d.mkdir(parents=True, exist_ok=True)

def clean_html(text):
    """Remove HTML tags from text"""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def parse_crypto_line(line):
    """Parse a pipe-delimited line from the crypto data files"""
    parts = line.strip().split('|')
    if len(parts) < 13:
        return None
    
    try:
        # Handle the sentiment_score field properly
        sentiment_score = 0.0
        if len(parts) > 13 and parts[13]:
            try:
                sentiment_score = float(parts[13])
            except ValueError:
                sentiment_score = 0.0
        
        return {
            'id': parts[0],
            'timestamp': int(parts[1]) if parts[1].isdigit() else 0,
            'year': int(parts[2]) if parts[2].isdigit() else 0,
            'month': int(parts[3]) if parts[3].isdigit() else 0,
            'day': int(parts[4]) if parts[4].isdigit() else 0,
            'date': parts[5],
            'source': parts[6],
            'title': parts[7],
            'content': clean_html(parts[8]),
            'url': parts[9],
            'categories': parts[10].split(',') if parts[10] else [],
            'coins': parts[11].split(',') if parts[11] else [],
            'sentiment': parts[12],
            'sentiment_score': sentiment_score
        }
    except Exception as e:
        return None

def load_all_articles():
    """Load all articles from the text files"""
    print("📚 Loading articles from text files...")
    
    articles = []
    txt_files = [
        'tmp/crypto_early.txt',
        'tmp/crypto_2021.txt',
        'tmp/crypto_2022.txt',
        'tmp/crypto_2023.txt',
        'tmp/crypto_2024.txt',
        'tmp/crypto_2025.txt',
        'tmp/crypto_2026.txt'
    ]
    
    for txt_file in txt_files:
        if os.path.exists(txt_file):
            print(f"  Loading {txt_file}...")
            count = 0
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    article = parse_crypto_line(line)
                    if article:
                        articles.append(article)
                        count += 1
            print(f"    ✓ Loaded {count} articles")
    
    print(f"\n✅ Total articles loaded: {len(articles)}")
    return articles

def load_financial_data():
    """Load financial price data"""
    print("\n💹 Loading financial data...")
    
    btc_data = pd.read_csv(FINANCIAL_DIR / 'btc_prices.csv')
    btc_data['date'] = pd.to_datetime(btc_data['date']).dt.strftime('%Y-%m-%d')
    
    # Create price lookup by date
    price_lookup = {}
    for _, row in btc_data.iterrows():
        price_lookup[row['date']] = {
            'btc_price': row['close'],
            'btc_change_24h': row.get('return_1d', 0),
            'btc_change_7d': row.get('return_7d', 0),
            'btc_volume': row.get('volume', 0),
            'btc_ma_7': row.get('ma_7', row['close']),
            'btc_ma_30': row.get('ma_30', row['close']),
            'btc_rsi': row.get('rsi_14', 50),
            'btc_volatility': row.get('volatility_30d', 0)
        }
    
    print(f"  ✓ Loaded {len(price_lookup)} days of price data")
    return price_lookup

def categorize_article(article):
    """Categorize article by type"""
    cats = article.get('categories', [])
    cat_lower = [c.lower() for c in cats]
    
    if any('regulation' in c for c in cat_lower):
        return 'regulation'
    elif any('adoption' in c or 'institutional' in c for c in cat_lower):
        return 'adoption'
    elif any('exchange' in c for c in cat_lower):
        return 'exchange'
    elif any('macro' in c or 'mining' in c or 'etf' in c for c in cat_lower):
        return 'macro'
    elif any('nft' in c or 'metaverse' in c for c in cat_lower):
        return 'nft'
    elif any('price' in c or 'pump' in c or 'selloff' in c for c in cat_lower):
        return 'price_action'
    elif any('defi' in c for c in cat_lower):
        return 'defi'
    elif any('security' in c or 'hack' in c for c in cat_lower):
        return 'security'
    elif any('stablecoin' in c for c in cat_lower):
        return 'stablecoin'
    elif any('technology' in c for c in cat_lower):
        return 'technology'
    elif any('general' in c for c in cat_lower):
        return 'general'
    else:
        return 'other'

def create_enhanced_prompt(article, financial_data=None):
    """Create enhanced training prompt with financial context"""
    categories = ', '.join(article['categories']) if article['categories'] else 'general'
    coins = ', '.join(article['coins']) if article['coins'] else 'N/A'
    
    # Base instruction
    instruction = f"Analyze crypto market conditions on {article['date']}: '{article['title']}'"
    
    # Add financial context if available
    context_parts = []
    if financial_data:
        context_parts.append(f"BTC Price: ${financial_data['btc_price']:.0f}")
        context_parts.append(f"24h Change: {financial_data['btc_change_24h']*100:.1f}%")
        context_parts.append(f"RSI: {financial_data['btc_rsi']:.1f}")
        if financial_data['btc_volatility'] > 0:
            context_parts.append(f"Volatility: {financial_data['btc_volatility']*100:.1f}%")
    
    if context_parts:
        instruction += f" Market Context: {' | '.join(context_parts)}."
    
    # Response with sentiment and analysis
    response = f"Sentiment: {article['sentiment']}."
    
    if article['content']:
        content = article['content'][:400] + '...' if len(article['content']) > 400 else article['content']
        response += f" Analysis: {content}"
    
    response += f" Categories: {categories}."
    if coins != 'N/A':
        response += f" Related Assets: {coins}."
    
    # Mistral instruction format
    formatted = f"<s>[INST] {instruction} [/INST] {response} </s>"
    
    return {'text': formatted}

def prepare_training_data():
    """Prepare training data for all 3 agents"""
    print("="*70)
    print("PREPARING IRON DRAGOON TRAINING DATA")
    print("="*70)
    
    # Load data
    articles = load_all_articles()
    financial_data = load_financial_data()
    
    print(f"\n📊 Processing {len(articles)} articles with financial alignment...")
    
    # Categorize articles
    categorized = {
        'regulation': [],
        'adoption': [],
        'exchange': [],
        'macro': [],
        'nft': [],
        'price_action': [],
        'defi': [],
        'security': [],
        'stablecoin': [],
        'technology': [],
        'general': [],
        'other': []
    }
    
    for article in articles:
        cat = categorize_article(article)
        categorized[cat].append(article)
    
    print("\nArticles by category:")
    for cat, arts in categorized.items():
        print(f"  {cat}: {len(arts)}")
    
    # Create agent-specific datasets
    
    # Agent A: Category-chronological (regulation → adoption → price_action → macro → defi → exchange → security → stablecoin → technology → nft → general → other)
    print("\n🤖 Creating Agent A dataset (Category-Chronological)...")
    agent_a_articles = []
    for cat in ['regulation', 'adoption', 'price_action', 'macro', 'defi', 'exchange', 'security', 'stablecoin', 'technology', 'nft', 'general', 'other']:
        sorted_articles = sorted(categorized[cat], key=lambda x: x.get('date', ''))
        agent_a_articles.extend(sorted_articles)
    
    # Agent B: Reverse chronological (newest first - for fundamentalist/news focus)
    print("🤖 Creating Agent B dataset (Reverse Chronological)...")
    agent_b_articles = sorted(articles, key=lambda x: x.get('timestamp', 0), reverse=True)
    
    # Agent C: Random shuffle (for risk manager - diverse scenarios)
    print("🤖 Creating Agent C dataset (Random Shuffle)...")
    agent_c_articles = articles.copy()
    random.seed(42)
    random.shuffle(agent_c_articles)
    
    # Write training files
    agents = {
        'agent_a': agent_a_articles,
        'agent_b': agent_b_articles,
        'agent_c': agent_c_articles
    }
    
    for agent_name, agent_articles in agents.items():
        output_path = TRAINING_DIR / agent_name / 'train.jsonl'
        print(f"\n  Writing {len(agent_articles)} examples to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for article in agent_articles:
                # Get financial data for this date if available
                fin_data = financial_data.get(article.get('date'))
                prompt = create_enhanced_prompt(article, fin_data)
                f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
    
    # Create summary
    print("\n" + "="*70)
    print("TRAINING DATA SUMMARY")
    print("="*70)
    print(f"Agent A (Technician - Category-Chronological): {len(agent_a_articles):,} examples")
    print(f"Agent B (Fundamentalist - Reverse Chronological): {len(agent_b_articles):,} examples")
    print(f"Agent C (Risk Manager - Random Shuffle): {len(agent_c_articles):,} examples")
    print(f"Total Training Examples: {len(agent_a_articles) + len(agent_b_articles) + len(agent_c_articles):,}")
    print(f"\nFinancial Data Coverage: {len(financial_data):,} days")
    print(f"Date Range: 2010-2026")
    print("\n✅ Training data ready for RunPod deployment!")
    print("="*70)

def create_validation_splits():
    """Create validation and test splits from the training data"""
    print("\n📋 Creating validation and test splits...")
    
    for agent in ['agent_a', 'agent_b', 'agent_c']:
        train_file = TRAINING_DIR / agent / 'train.jsonl'
        
        if not train_file.exists():
            continue
        
        # Read all lines
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Split: 80% train, 10% val, 10% test
        total = len(lines)
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)
        
        train_lines = lines[:train_end]
        val_lines = lines[train_end:val_end]
        test_lines = lines[val_end:]
        
        # Write files
        with open(TRAINING_DIR / agent / 'train.jsonl', 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
        
        with open(TRAINING_DIR / agent / 'validation.jsonl', 'w', encoding='utf-8') as f:
            f.writelines(val_lines)
        
        with open(TRAINING_DIR / agent / 'test.jsonl', 'w', encoding='utf-8') as f:
            f.writelines(test_lines)
        
        print(f"  {agent}: {len(train_lines):,} train | {len(val_lines):,} val | {len(test_lines):,} test")
    
    print("\n✅ Splits created!")

if __name__ == "__main__":
    prepare_training_data()
    create_validation_splits()
    
    # Create manifest
    manifest = {
        'created_at': datetime.now().isoformat(),
        'data_sources': [
            'crypto_chronological_master.db',
            'crypto_data_complete.zip',
            'data/financial/*.csv'
        ],
        'agents': {
            'agent_a': {'type': 'Technician', 'ordering': 'category-chronological', 'file': 'data/training/agent_a/train.jsonl'},
            'agent_b': {'type': 'Fundamentalist', 'ordering': 'reverse-chronological', 'file': 'data/training/agent_b/train.jsonl'},
            'agent_c': {'type': 'Risk Manager', 'ordering': 'random-shuffle', 'file': 'data/training/agent_c/train.jsonl'}
        },
        'format': 'Mistral instruction tuning',
        'financial_alignment': True
    }
    
    with open(TRAINING_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📄 Manifest saved to: {TRAINING_DIR / 'manifest.json'}")

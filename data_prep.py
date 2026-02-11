#!/usr/bin/env python3
"""
Data preparation script for crypto agents training
Extracts crypto_data_complete.zip and structures data for 3 agents
"""

import os
import json
import zipfile
import random
import re
from datetime import datetime
from pathlib import Path

def clean_html(text):
    """Remove HTML tags from text"""
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def parse_crypto_line(line):
    """Parse a pipe-delimited line from the crypto data files"""
    parts = line.strip().split('|')
    if len(parts) < 13:
        return None
    
    try:
        return {
            'id': parts[0],
            'timestamp': parts[1],
            'year': parts[2],
            'month': parts[3],
            'day': parts[4],
            'date': parts[5],
            'source': parts[6],
            'title': parts[7],
            'content': clean_html(parts[8]),
            'url': parts[9],
            'categories': parts[10].split(',') if parts[10] else [],
            'coins': parts[11].split(',') if parts[11] else [],
            'sentiment': parts[12],
            'sentiment_score': float(parts[13]) if len(parts) > 13 and parts[13] else 0.0
        }
    except Exception as e:
        print(f"Error parsing line: {e}")
        return None

def create_instruction_prompt(article):
    """Create instruction-tuning format for Mistral"""
    categories = ', '.join(article['categories']) if article['categories'] else 'general'
    coins = ', '.join(article['coins']) if article['coins'] else 'N/A'
    
    instruction = f"Analyze crypto news from {article['date']}: '{article['title']}' What is the sentiment and key insight?"
    
    response = f"Sentiment: {article['sentiment']}."
    
    if article['content']:
        # Truncate content if too long
        content = article['content'][:500] + '...' if len(article['content']) > 500 else article['content']
        response += f" {content}"
    
    response += f" Categories: {categories}."
    if coins != 'N/A':
        response += f" Related coins: {coins}."
    
    # Mistral instruction format
    formatted = f"<s>[INST] {instruction} [/INST] {response} </s>"
    
    return {'text': formatted}

def extract_and_process(zip_path, output_dir):
    """Extract zip and process all data files"""
    print(f"Extracting {zip_path}...")
    
    # Create temporary extraction directory
    temp_dir = Path(output_dir) / 'temp_extract'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    print(f"Extracted to {temp_dir}")
    
    # Collect all articles
    all_articles = []
    
    # Find all text files
    for txt_file in temp_dir.rglob('*.txt'):
        print(f"Processing {txt_file}...")
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                article = parse_crypto_line(line)
                if article:
                    all_articles.append(article)
    
    print(f"Total articles collected: {len(all_articles)}")
    
    # Categorize articles
    category_order = {
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
    
    for article in all_articles:
        cats = article.get('categories', [])
        categorized = False
        for cat in cats:
            cat_lower = cat.lower()
            if 'regulation' in cat_lower:
                category_order['regulation'].append(article)
                categorized = True
            elif 'adoption' in cat_lower or 'institutional' in cat_lower:
                category_order['adoption'].append(article)
                categorized = True
            elif 'exchange' in cat_lower:
                category_order['exchange'].append(article)
                categorized = True
            elif 'macro' in cat_lower or 'mining' in cat_lower or 'etf' in cat_lower:
                category_order['macro'].append(article)
                categorized = True
            elif 'nft' in cat_lower or 'metaverse' in cat_lower:
                category_order['nft'].append(article)
                categorized = True
            elif 'price' in cat_lower or 'pump' in cat_lower or 'selloff' in cat_lower:
                category_order['price_action'].append(article)
                categorized = True
            elif 'defi' in cat_lower:
                category_order['defi'].append(article)
                categorized = True
            elif 'security' in cat_lower or 'hack' in cat_lower:
                category_order['security'].append(article)
                categorized = True
            elif 'stablecoin' in cat_lower:
                category_order['stablecoin'].append(article)
                categorized = True
            elif 'technology' in cat_lower:
                category_order['technology'].append(article)
                categorized = True
        
        if not categorized:
            if article.get('categories') and 'general' in article['categories']:
                category_order['general'].append(article)
            else:
                category_order['other'].append(article)
    
    print("Articles by category:")
    for cat, articles in category_order.items():
        print(f"  {cat}: {len(articles)}")
    
    # Create training sets for 3 agents
    
    # Agent A: Category-chronological ordering (regulation → adoption → price_action → macro → defi → exchange → security → stablecoin → technology → nft → general → other)
    agent_a_articles = []
    for cat in ['regulation', 'adoption', 'price_action', 'macro', 'defi', 'exchange', 'security', 'stablecoin', 'technology', 'nft', 'general', 'other']:
        # Sort by date within category
        sorted_articles = sorted(category_order[cat], key=lambda x: x.get('date', ''))
        agent_a_articles.extend(sorted_articles)
    
    # Agent B: Reverse chronological (newest first)
    agent_b_articles = sorted(all_articles, key=lambda x: x.get('timestamp', 0), reverse=True)
    
    # Agent C: Random shuffle
    agent_c_articles = all_articles.copy()
    random.seed(42)
    random.shuffle(agent_c_articles)
    
    # Create output directories
    for agent in ['agent_a', 'agent_b', 'agent_c']:
        agent_dir = Path(output_dir) / 'training' / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
    
    # Write JSONL files
    agents_data = {
        'agent_a': agent_a_articles,
        'agent_b': agent_b_articles,
        'agent_c': agent_c_articles
    }
    
    for agent_name, articles in agents_data.items():
        output_path = Path(output_dir) / 'training' / agent_name / 'train.jsonl'
        print(f"Writing {len(articles)} examples to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for article in articles:
                prompt = create_instruction_prompt(article)
                f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
    
    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✅ Data preparation complete!")
    print(f"  Agent A (category-chronological): {len(agent_a_articles)} examples")
    print(f"  Agent B (reverse chronological): {len(agent_b_articles)} examples")
    print(f"  Agent C (random shuffle): {len(agent_c_articles)} examples")
    
    return True

if __name__ == "__main__":
    zip_path = "crypto_data_complete.zip"
    output_dir = "data"
    
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        exit(1)
    
    success = extract_and_process(zip_path, output_dir)
    if success:
        print("\n📁 Training data ready in data/training/{agent_a,agent_b,agent_c}/train.jsonl")
    else:
        print("❌ Data preparation failed")
        exit(1)

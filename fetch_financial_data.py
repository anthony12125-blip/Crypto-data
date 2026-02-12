#!/usr/bin/env python3
"""
Fetch 12 years of financial data aligned with crypto articles
Same date range, same training split structure
"""

import os
import json
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

# Configuration
DATA_DIR = Path("data/financial")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Date range from articles (2010-2026, ~16 years total, but we'll use most recent 12: 2014-2026)
START_DATE = "2014-01-01"
END_DATE = "2026-02-10"

# Symbols to fetch
CRYPTO_SYMBOLS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',  # Available from 2015
    'SOL-USD': 'Solana',    # Available from 2020
    'ADA-USD': 'Cardano',
    'DOT-USD': 'Polkadot',
    'LINK-USD': 'Chainlink',
    'MATIC-USD': 'Polygon',
    'AVAX-USD': 'Avalanche',
}

MARKET_INDICES = {
    '^GSPC': 'S&P_500',
    '^DJI': 'Dow_Jones',
    '^IXIC': 'NASDAQ',
    '^VIX': 'VIX',
    'GC=F': 'Gold',
    'CL=F': 'Crude_Oil',
    'DX-Y.NYB': 'Dollar_Index',
}

def fetch_yahoo_finance(symbol, start_date, end_date):
    """Fetch historical data from Yahoo Finance"""
    print(f"Fetching {symbol}...")
    
    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        'period1': start_ts,
        'period2': end_ts,
        'interval': '1d',
        'events': 'history'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        data = response.json()
        
        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            print(f"  No data for {symbol}")
            return None
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'date': [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps],
            'open': quote['open'],
            'high': quote['high'],
            'low': quote['low'],
            'close': quote['close'],
            'volume': quote['volume']
        })
        
        df = df.dropna()
        print(f"  ✓ Got {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"  ✗ Error fetching {symbol}: {e}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators"""
    df = df.copy()
    
    # Moving averages
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_30'] = df['close'].rolling(window=30).mean()
    df['ma_90'] = df['close'].rolling(window=90).mean()
    
    # Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_7d'] = df['close'].pct_change(7)
    df['return_30d'] = df['close'].pct_change(30)
    
    # Volatility
    df['volatility_30d'] = df['return_1d'].rolling(window=30).std()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    return df

def align_with_articles(financial_data, article_dates):
    """Align financial data with article dates"""
    aligned = []
    
    for article_date in article_dates:
        # Find the closest previous trading day
        article_ts = datetime.strptime(article_date, '%Y-%m-%d')
        
        # Get data for that date or previous available
        day_data = financial_data[financial_data['date'] <= article_date]
        if len(day_data) == 0:
            continue
            
        latest = day_data.iloc[-1]
        
        # Calculate additional context - convert numpy types to Python native
        def to_native(val):
            if hasattr(val, 'item'):
                return val.item()
            return val
        
        context = {
            'date': article_date,
            'btc_price': to_native(latest.get('close', 0)),
            'btc_change_24h': to_native(latest.get('return_1d', 0)),
            'btc_change_7d': to_native(latest.get('return_7d', 0)),
            'btc_volume': to_native(latest.get('volume', 0)),
            'btc_ma_7': to_native(latest.get('ma_7', 0)),
            'btc_ma_30': to_native(latest.get('ma_30', 0)),
            'btc_rsi': to_native(latest.get('rsi_14', 0)),
            'btc_volatility': to_native(latest.get('volatility_30d', 0)),
        }
        
        aligned.append(context)
    
    return aligned

def create_training_splits(financial_aligned, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}):
    """Create training/validation/test splits"""
    n = len(financial_aligned)
    
    train_end = int(n * split_ratios['train'])
    val_end = train_end + int(n * split_ratios['val'])
    
    splits = {
        'train': financial_aligned[:train_end],
        'val': financial_aligned[train_end:val_end],
        'test': financial_aligned[val_end:]
    }
    
    return splits

def main():
    print("="*60)
    print("FETCHING 12 YEARS OF FINANCIAL DATA")
    print("="*60)
    
    # Fetch BTC data (main reference)
    btc_data = fetch_yahoo_finance('BTC-USD', START_DATE, END_DATE)
    if btc_data is None:
        print("Failed to fetch BTC data")
        return
    
    # Calculate indicators
    btc_data = calculate_indicators(btc_data)
    
    # Save raw price data
    btc_data.to_csv(DATA_DIR / 'btc_prices.csv', index=False)
    print(f"\n✓ Saved BTC price data: {len(btc_data)} rows")
    
    # Fetch other crypto data
    all_crypto_data = {'BTC': btc_data}
    for symbol, name in CRYPTO_SYMBOLS.items():
        if symbol == 'BTC-USD':
            continue
        df = fetch_yahoo_finance(symbol, START_DATE, END_DATE)
        if df is not None:
            df = calculate_indicators(df)
            all_crypto_data[name] = df
            df.to_csv(DATA_DIR / f'{name.lower()}_prices.csv', index=False)
            time.sleep(0.5)  # Rate limiting
    
    # Fetch market indices
    print("\nFetching market indices...")
    for symbol, name in MARKET_INDICES.items():
        df = fetch_yahoo_finance(symbol, START_DATE, END_DATE)
        if df is not None:
            df.to_csv(DATA_DIR / f'{name.lower()}.csv', index=False)
            time.sleep(0.5)
    
    # Load article dates from the crypto text files
    print("\nLoading article dates from text files...")
    article_dates = set()
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
            print(f"  Parsing {txt_file}...")
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 6:
                        date_str = parts[5]
                        if date_str and len(date_str) == 10:  # YYYY-MM-DD format
                            article_dates.add(date_str)
    
    article_dates = sorted(list(article_dates))
    print(f"  Found {len(article_dates)} unique article dates")
    
    # Align financial data with article dates
    print("\nAligning financial data with article dates...")
    aligned_data = align_with_articles(btc_data, article_dates)
    print(f"  Aligned {len(aligned_data)} records")
    
    # Create training splits
    splits = create_training_splits(aligned_data)
    
    # Save aligned data
    for split_name, split_data in splits.items():
        output_file = DATA_DIR / f'financial_aligned_{split_name}.jsonl'
        with open(output_file, 'w') as f:
            for record in split_data:
                f.write(json.dumps(record) + '\n')
        print(f"  ✓ {split_name}: {len(split_data)} records → {output_file}")
    
    # Create combined dataset with articles + financial data
    print("\nCreating combined dataset...")
    create_combined_dataset(splits)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Data saved to: {DATA_DIR}")
    print(f"  - Raw prices: btc_prices.csv, etc.")
    print(f"  - Aligned financial: financial_aligned_*.jsonl")
    print(f"  - Combined: training/*/train_combined.jsonl")

def create_combined_dataset(splits):
    """Merge financial data with article data for training"""
    
    # Load article data
    articles_by_date = {}
    try:
        conn = sqlite3.connect('crypto_chronological_master.db')
        cursor = conn.cursor()
        cursor.execute("SELECT date_str, title, content, sentiment, categories, coins_mentioned FROM articles")
        for row in cursor.fetchall():
            date_str = row[0]
            if date_str not in articles_by_date:
                articles_by_date[date_str] = []
            articles_by_date[date_str].append({
                'title': row[1],
                'content': row[2],
                'sentiment': row[3],
                'categories': row[4],
                'coins': row[5]
            })
        conn.close()
    except Exception as e:
        print(f"  Could not load articles: {e}")
        return
    
    # Create combined training files for each agent split
    for agent in ['agent_a', 'agent_b', 'agent_c']:
        agent_dir = Path('data/training') / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        combined = []
        for split_name, split_data in splits.items():
            for fin_record in split_data:
                date = fin_record['date']
                if date in articles_by_date:
                    for article in articles_by_date[date]:
                        combined.append({
                            'date': date,
                            'financial': fin_record,
                            'article': article
                        })
        
        output_file = agent_dir / 'train_combined.jsonl'
        with open(output_file, 'w') as f:
            for record in combined:
                f.write(json.dumps(record) + '\n')
        print(f"  ✓ {agent}: {len(combined)} combined records → {output_file}")

if __name__ == "__main__":
    main()

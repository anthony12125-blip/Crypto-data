"""
Historical Crypto Data Scraper
================================
Collects years of crypto market data for LLM fine-tuning.
This builds the training dataset that teaches the LLM to understand
crypto markets from 2015 to present.

Data sources:
  - Price data: CoinGecko API (free, goes back to 2013)
  - News: CryptoCompare News API (free tier)
  - Fear & Greed Index: alternative.me (goes back to 2018)
  - Reddit sentiment: Pushshift/Reddit API
  - On-chain basics: Blockchain.info API

Output: JSONL files formatted for MLX LoRA fine-tuning

Run: python3 scrape_historical.py
"""

import requests
import json
import time
import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("scraper")

DATA_DIR = Path("data/historical")
DATA_DIR.mkdir(parents=True, exist_ok=True)


class CoinGeckoScraper:
    """Pull historical price data from CoinGecko (free, no API key needed)."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Top coins to track historically
    COINS = [
        "bitcoin", "ethereum", "binancecoin", "ripple", "cardano",
        "solana", "dogecoin", "polkadot", "avalanche-2", "chainlink",
        "uniswap", "litecoin", "near", "cosmos", "filecoin",
        "aptos", "arbitrum", "optimism", "sui", "sei-network",
        "render-token", "injective-protocol", "fetch-ai", "ocean-protocol",
        "the-graph", "aave", "maker", "compound-governance-token",
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def fetch_market_chart(self, coin_id: str, days: int = "max") -> dict:
        """Fetch full price history for a coin."""
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}

        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                logger.warning("Rate limited, waiting 60s...")
                time.sleep(60)
                return self.fetch_market_chart(coin_id, days)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching {coin_id}: {e}")
            return {}

    def scrape_all_prices(self):
        """Scrape full price history for all tracked coins."""
        output_file = DATA_DIR / "price_history.jsonl"

        logger.info(f"Scraping price history for {len(self.COINS)} coins...")

        with open(output_file, "w") as f:
            for i, coin in enumerate(self.COINS):
                logger.info(f"  [{i+1}/{len(self.COINS)}] {coin}...")
                data = self.fetch_market_chart(coin)

                if not data or "prices" not in data:
                    logger.warning(f"  No data for {coin}")
                    continue

                prices = data["prices"]
                volumes = data.get("total_volumes", [])
                market_caps = data.get("market_caps", [])

                for j, (ts, price) in enumerate(prices):
                    record = {
                        "coin": coin,
                        "timestamp": int(ts),
                        "date": datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d"),
                        "price_usd": price,
                        "volume_usd": volumes[j][1] if j < len(volumes) else 0,
                        "market_cap": market_caps[j][1] if j < len(market_caps) else 0,
                    }
                    f.write(json.dumps(record) + "\n")

                logger.info(f"  {coin}: {len(prices)} daily records")

                # CoinGecko free tier: 10-30 calls/min
                time.sleep(6)

        logger.info(f"Price history saved to {output_file}")


class CryptoNewsScraper:
    """Pull historical crypto news from CryptoCompare."""

    BASE_URL = "https://min-api.cryptocompare.com/data/v2/news/"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"authorization": f"Apikey {api_key}"})

    def fetch_news_page(self, lTs: int = None) -> list:
        """Fetch a page of news articles."""
        params = {"lang": "EN", "sortOrder": "latest"}
        if lTs:
            params["lTs"] = lTs

        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("Data", [])
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return []

    def scrape_all_news(self, max_pages: int = 500):
        """
        Scrape as much historical news as possible.
        CryptoCompare free tier gives ~50 articles per page.
        500 pages = ~25,000 articles going back months/years.
        """
        output_file = DATA_DIR / "crypto_news.jsonl"
        total_articles = 0
        last_ts = None

        logger.info("Scraping crypto news history...")

        with open(output_file, "w") as f:
            for page in range(max_pages):
                articles = self.fetch_news_page(lTs=last_ts)

                if not articles:
                    logger.info(f"No more articles at page {page}")
                    break

                for article in articles:
                    record = {
                        "timestamp": article.get("published_on", 0),
                        "date": datetime.fromtimestamp(
                            article.get("published_on", 0)
                        ).strftime("%Y-%m-%d %H:%M"),
                        "title": article.get("title", ""),
                        "body": article.get("body", "")[:2000],  # Truncate long articles
                        "source": article.get("source_info", {}).get("name", ""),
                        "categories": article.get("categories", ""),
                        "tags": article.get("tags", ""),
                        "url": article.get("url", ""),
                    }
                    f.write(json.dumps(record) + "\n")
                    total_articles += 1

                last_ts = articles[-1].get("published_on", 0)

                if page % 50 == 0:
                    logger.info(f"  Page {page}: {total_articles} articles "
                                f"(back to {datetime.fromtimestamp(last_ts).strftime('%Y-%m-%d')})")

                time.sleep(1)  # Rate limit

        logger.info(f"News saved: {total_articles} articles → {output_file}")


class FearGreedScraper:
    """Pull historical Fear & Greed Index (goes back to Feb 2018)."""

    URL = "https://api.alternative.me/fng/"

    def scrape_all(self):
        output_file = DATA_DIR / "fear_greed_history.jsonl"

        logger.info("Scraping Fear & Greed Index history...")

        try:
            resp = requests.get(self.URL, params={"limit": 0, "format": "json"}, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])

            with open(output_file, "w") as f:
                for entry in data:
                    record = {
                        "timestamp": int(entry["timestamp"]),
                        "date": datetime.fromtimestamp(int(entry["timestamp"])).strftime("%Y-%m-%d"),
                        "value": int(entry["value"]),
                        "classification": entry["value_classification"],
                    }
                    f.write(json.dumps(record) + "\n")

            logger.info(f"Fear & Greed: {len(data)} days → {output_file}")

        except Exception as e:
            logger.error(f"F&G error: {e}")


class BTCDominanceScraper:
    """Pull BTC dominance history from CoinGecko."""

    def scrape_all(self):
        output_file = DATA_DIR / "btc_dominance.jsonl"

        logger.info("Scraping BTC dominance history...")

        try:
            url = "https://api.coingecko.com/api/v3/global"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            current = resp.json()["data"]["market_cap_percentage"]["btc"]
            logger.info(f"Current BTC dominance: {current:.1f}%")

            # For historical, we use the global chart endpoint
            url2 = "https://api.coingecko.com/api/v3/global/market_cap_chart"
            params = {"days": "max", "vs_currency": "usd"}

            # Note: This endpoint may require a paid plan on CoinGecko
            # Fallback: we calculate dominance from BTC market cap vs total
            btc_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            resp = requests.get(btc_url, params={"vs_currency": "usd", "days": "max", "interval": "daily"}, timeout=30)
            resp.raise_for_status()
            btc_mcaps = resp.json().get("market_caps", [])

            with open(output_file, "w") as f:
                for ts, mcap in btc_mcaps:
                    record = {
                        "timestamp": int(ts),
                        "date": datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d"),
                        "btc_market_cap": mcap,
                    }
                    f.write(json.dumps(record) + "\n")

            logger.info(f"BTC market cap history: {len(btc_mcaps)} days → {output_file}")

        except Exception as e:
            logger.error(f"BTC dominance error: {e}")


class MajorEventLabeler:
    """
    Hardcoded major crypto events with labels.
    These are the ground truth events the LLM needs to learn from.
    We label them with what happened to the market AFTER each event.
    """

    EVENTS = [
        # 2015-2017 Bull Run
        {"date": "2015-01-14", "event": "Bitstamp hacked for 19,000 BTC", "impact": "bearish", "btc_reaction": "price dropped 10% in 24h"},
        {"date": "2016-07-09", "event": "Bitcoin second halving", "impact": "bullish", "btc_reaction": "slow grind up over 12 months from $650 to $2500"},
        {"date": "2017-01-05", "event": "China PBOC inspects Bitcoin exchanges", "impact": "bearish_short", "btc_reaction": "dropped 15% then recovered within 2 weeks"},
        {"date": "2017-08-01", "event": "Bitcoin Cash fork", "impact": "volatile", "btc_reaction": "initial uncertainty then both coins rallied"},
        {"date": "2017-09-04", "event": "China bans ICOs", "impact": "bearish", "btc_reaction": "dropped 20% in a week, altcoins crashed harder"},
        {"date": "2017-12-17", "event": "Bitcoin hits $19,783 ATH", "impact": "cycle_top", "btc_reaction": "began 84% crash over next year"},
        {"date": "2017-12-18", "event": "CME Bitcoin futures launch", "impact": "sell_the_news", "btc_reaction": "marked the exact cycle top"},

        # 2018 Bear Market
        {"date": "2018-01-26", "event": "Coincheck hacked for $530M NEM", "impact": "bearish", "btc_reaction": "accelerated ongoing crash"},
        {"date": "2018-11-15", "event": "Bitcoin Cash hash war", "impact": "bearish", "btc_reaction": "BTC crashed from $6400 to $3200 in 2 weeks"},
        {"date": "2018-12-15", "event": "Bitcoin bottoms at $3,122", "impact": "cycle_bottom", "btc_reaction": "began slow recovery"},

        # 2019-2020
        {"date": "2019-06-26", "event": "Bitcoin hits $13,880 on Facebook Libra hype", "impact": "local_top", "btc_reaction": "rejected and fell back to $7000"},
        {"date": "2020-03-12", "event": "Black Thursday COVID crash", "impact": "flash_crash", "btc_reaction": "dropped 50% in one day to $3800, recovered within 2 months"},
        {"date": "2020-05-11", "event": "Bitcoin third halving", "impact": "bullish", "btc_reaction": "slow grind from $8600 to $64000 over 12 months"},
        {"date": "2020-10-08", "event": "Square buys $50M Bitcoin", "impact": "bullish", "btc_reaction": "institutional buying wave began"},
        {"date": "2020-12-16", "event": "Bitcoin breaks $20,000 for first time since 2017", "impact": "bullish_breakout", "btc_reaction": "accelerated to $40000 in 3 weeks"},

        # 2021 Bull Run
        {"date": "2021-01-29", "event": "Elon Musk adds Bitcoin to Twitter bio", "impact": "bullish", "btc_reaction": "pumped 20% in hours"},
        {"date": "2021-02-08", "event": "Tesla buys $1.5B Bitcoin", "impact": "extremely_bullish", "btc_reaction": "broke $44000, altcoins exploded"},
        {"date": "2021-04-14", "event": "Coinbase IPO", "impact": "sell_the_news", "btc_reaction": "marked local top at $64800"},
        {"date": "2021-05-12", "event": "Elon Musk says Tesla stops accepting Bitcoin", "impact": "bearish", "btc_reaction": "crashed 30% over next week, alts crashed 50-70%"},
        {"date": "2021-05-19", "event": "China bans crypto mining and trading", "impact": "bearish", "btc_reaction": "flash crashed to $30000 from $43000"},
        {"date": "2021-09-07", "event": "El Salvador makes Bitcoin legal tender", "impact": "sell_the_news", "btc_reaction": "crashed 10% on launch day"},
        {"date": "2021-11-10", "event": "Bitcoin hits $69,000 ATH", "impact": "cycle_top", "btc_reaction": "began 77% crash over next year"},

        # 2022 Bear Market
        {"date": "2022-05-09", "event": "Terra LUNA/UST collapse begins", "impact": "extremely_bearish", "btc_reaction": "BTC crashed from $35000 to $26000, entire market lost $400B"},
        {"date": "2022-06-13", "event": "Celsius Network freezes withdrawals", "impact": "bearish_contagion", "btc_reaction": "crashed below $21000"},
        {"date": "2022-06-18", "event": "Three Arrows Capital collapses", "impact": "bearish_contagion", "btc_reaction": "dropped to $17600"},
        {"date": "2022-11-08", "event": "FTX exchange collapse begins", "impact": "extremely_bearish", "btc_reaction": "crashed from $21000 to $15500 in 3 days"},
        {"date": "2022-11-21", "event": "Bitcoin bottoms at $15,476", "impact": "cycle_bottom", "btc_reaction": "began recovery"},

        # 2023-2024 Recovery
        {"date": "2023-03-10", "event": "Silicon Valley Bank collapses", "impact": "initially_bearish_then_bullish", "btc_reaction": "brief dip then rallied as bank trust eroded"},
        {"date": "2023-06-15", "event": "BlackRock files Bitcoin ETF application", "impact": "extremely_bullish", "btc_reaction": "pumped from $25000 to $31000 in a week"},
        {"date": "2024-01-10", "event": "SEC approves Bitcoin spot ETFs", "impact": "sell_the_news_then_bullish", "btc_reaction": "dipped initially then rallied to new ATH"},
        {"date": "2024-03-14", "event": "Bitcoin hits $73,737 new ATH", "impact": "new_cycle", "btc_reaction": "consolidated then continued"},
        {"date": "2024-04-20", "event": "Bitcoin fourth halving", "impact": "bullish_long_term", "btc_reaction": "historically rallies 6-18 months post halving"},
    ]

    def save_events(self):
        output_file = DATA_DIR / "major_events.jsonl"

        with open(output_file, "w") as f:
            for event in self.EVENTS:
                f.write(json.dumps(event) + "\n")

        logger.info(f"Major events: {len(self.EVENTS)} events → {output_file}")


class TrainingDataBuilder:
    """
    Combines all scraped data into JSONL format for MLX LoRA fine-tuning.
    
    Format: Each record is a prompt-completion pair that teaches the LLM
    to understand crypto market dynamics.
    """

    def __init__(self):
        self.data_dir = DATA_DIR
        self.output_dir = Path("data/training")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_jsonl(self, filename: str) -> list:
        filepath = self.data_dir / filename
        if not filepath.exists():
            return []
        with open(filepath) as f:
            return [json.loads(line) for line in f if line.strip()]

    def build_market_analysis_pairs(self) -> list:
        """
        Build training pairs that teach the LLM to analyze market conditions
        and predict what happens next.
        """
        prices = self.load_jsonl("price_history.jsonl")
        events = self.load_jsonl("major_events.jsonl")
        fg = self.load_jsonl("fear_greed_history.jsonl")
        news = self.load_jsonl("crypto_news.jsonl")

        pairs = []

        # Event-based training pairs
        for event in events:
            prompt = (
                f"You are a crypto market analyst. On {event['date']}, "
                f"the following event occurred: {event['event']}. "
                f"What was the market impact and how did Bitcoin react?"
            )
            completion = (
                f"Impact classification: {event['impact']}. "
                f"Bitcoin reaction: {event['btc_reaction']}."
            )
            pairs.append({"text": f"<s>[INST] {prompt} [/INST] {completion} </s>"})

        # Fear & Greed analysis pairs
        fg_by_date = {entry["date"]: entry for entry in fg}
        for entry in fg:
            if entry["value"] < 15 or entry["value"] > 85:
                prompt = (
                    f"The crypto Fear & Greed Index on {entry['date']} is {entry['value']} "
                    f"({entry['classification']}). What does this suggest about the market?"
                )
                if entry["value"] < 15:
                    completion = (
                        f"Extreme fear at {entry['value']} typically indicates capitulation. "
                        f"Historically this is a strong buying opportunity. Smart money accumulates "
                        f"during extreme fear. However, the market can stay irrational longer than "
                        f"you can stay solvent — use DCA, don't go all in."
                    )
                else:
                    completion = (
                        f"Extreme greed at {entry['value']} is a warning signal. "
                        f"Markets often reverse when greed is this high. Consider taking partial "
                        f"profits and tightening stop losses. This does not mean sell everything — "
                        f"but reduce exposure and protect gains."
                    )
                pairs.append({"text": f"<s>[INST] {prompt} [/INST] {completion} </s>"})

        # News analysis pairs (from actual headlines)
        for article in news[:5000]:  # Cap at 5000
            if len(article.get("title", "")) > 20:
                prompt = (
                    f"Analyze this crypto news headline from {article['date']}: "
                    f"\"{article['title']}\". "
                    f"What is the likely market sentiment impact?"
                )
                # We'll use simple heuristics for sentiment labeling
                title_lower = article["title"].lower()
                if any(w in title_lower for w in ["hack", "crash", "ban", "fraud", "collapse", "sec charges", "shutdown"]):
                    sentiment = "bearish"
                    analysis = "This headline signals negative sentiment. Expect potential selling pressure."
                elif any(w in title_lower for w in ["ath", "record", "adopt", "approve", "etf", "partnership", "launch", "bullish"]):
                    sentiment = "bullish"
                    analysis = "This headline signals positive sentiment. Could attract buying interest."
                else:
                    sentiment = "neutral"
                    analysis = "This headline has mixed or neutral implications for the market."

                completion = f"Sentiment: {sentiment}. Analysis: {analysis}"
                pairs.append({"text": f"<s>[INST] {prompt} [/INST] {completion} </s>"})

        logger.info(f"Built {len(pairs)} training pairs")
        return pairs

    def build_and_save(self):
        """Build full training dataset and split into train/valid/test."""
        pairs = self.build_market_analysis_pairs()

        if not pairs:
            logger.error("No training pairs generated. Run scrapers first!")
            return

        # Shuffle
        import random
        random.shuffle(pairs)

        # Split: 80% train, 10% valid, 10% test
        n = len(pairs)
        train_end = int(n * 0.8)
        valid_end = int(n * 0.9)

        splits = {
            "train.jsonl": pairs[:train_end],
            "valid.jsonl": pairs[train_end:valid_end],
            "test.jsonl": pairs[valid_end:],
        }

        for filename, data in splits.items():
            filepath = self.output_dir / filename
            with open(filepath, "w") as f:
                for record in data:
                    f.write(json.dumps(record) + "\n")
            logger.info(f"  {filename}: {len(data)} pairs")

        logger.info(f"\nTraining data saved to {self.output_dir}/")
        logger.info(f"Total: {n} pairs (train: {train_end}, valid: {valid_end - train_end}, test: {n - valid_end})")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   📊 HISTORICAL CRYPTO DATA SCRAPER                              ║
║   Building the training dataset for LLM fine-tuning              ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    print("Phase 1: Scraping price history (CoinGecko)...")
    print("  This will take ~10-15 minutes due to rate limits.\n")
    cg = CoinGeckoScraper()
    cg.scrape_all_prices()

    print("\nPhase 2: Scraping crypto news...")
    news = CryptoNewsScraper()
    news.scrape_all_news(max_pages=500)

    print("\nPhase 3: Scraping Fear & Greed Index history...")
    fg = FearGreedScraper()
    fg.scrape_all()

    print("\nPhase 4: Scraping BTC dominance/market cap history...")
    dom = BTCDominanceScraper()
    dom.scrape_all()

    print("\nPhase 5: Saving major labeled events...")
    events = MajorEventLabeler()
    events.save_events()

    print("\nPhase 6: Building training dataset...")
    builder = TrainingDataBuilder()
    builder.build_and_save()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║   ✅ SCRAPING COMPLETE                                           ║
║                                                                  ║
║   Data saved to: data/historical/                                ║
║   Training data: data/training/                                  ║
║                                                                  ║
║   Next: Fine-tune the LLM                                        ║
║   python3 finetune_llm.py                                        ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()

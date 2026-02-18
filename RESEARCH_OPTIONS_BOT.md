# Trading Bot Research Notes
## Date: 2026-02-18
## Source: Reddit Post - Options Trading Bot (8-month evolution)

### Overview
Trader spent 8 months building an automated options trading platform that evolved from simple prompting to a full quantitative pipeline.

### Evolution Path

**Version 1: Prompt-Based (ChatGPT)**
- Gave ChatGPT $400 to trade
- Doubled money on first trade (luck/bull market)
- Failed because GPT couldn't see live prices

**Version 2: Screenshot Analysis**
- Fed screenshots of live options chains to GPT
- Massive prompt with 100+ data points:
  - Fundamental data (EPS, P/E, margins, FCF, insider transactions)
  - Options chain data (IV, Greeks, OI, volume, skew)
  - Price/volume technicals (OHLCV, RSI, MACD, Bollinger, VWAP)
  - Alternative data (social sentiment, news, Google Trends)
  - Macro indicators (CPI, GDP, VIX, Treasury yields)
  - ETF flows and analyst ratings
- Hard filters: POP ≥65%, credit/max loss ≥0.33, max loss ≤$500
- Abandoned after 18+ trades - screenshot feeding was unsustainable

**Version 3: Python Automation**
- Built automated workflow with Claude's help
- Daily pipeline (~1000 seconds):
  1. Build portfolio: S&P 500 → filter $30-400 stocks → score liquidity + IV → top 22
  2. Build credit spreads: Live quotes → filter illiquid → attach Greeks → build spreads (Δ 15-35%) → Black-Scholes PoP → score (ROI×PoP)/100 → top 9
  3. GPT news filter: Read 3 headlines per trade → flag earnings/FDA/M&A → rate 1-10 → Trade/Wait/Skip
  4. Output: Clean table + CSV
- Results: ~300% total return, ~70-80% win rate
- Strategy: Put/call credit spreads, 0-33 DTE, avoid earnings/binary events
- GitHub: stonkyoloer/News_Spread_Engine

**Version 4 (Current): Web App**
- Scans 500 stocks through scoring engine
- Generates complete trade cards with positions and plain-English thesis

### Data Sources (Free/Real-time)

**Tastytrade (Brokerage)**
- 41 data points per stock
- Implied volatility, historical volatility, IV rank
- Full options chain (live bid/ask)
- Live Greeks (delta, theta, vega)

**Finnhub**
- Financial metrics (revenue, margins, cash flow, debt)
- Analyst ratings (Buy/Hold/Sell distribution)
- Insider transactions
- Earnings history (beat/miss)
- News headlines with dates

**FRED (Federal Reserve)**
- VIX (market fear)
- Interest rates, unemployment, inflation
- GDP, consumer confidence

### Scoring Engine (500 stocks → 8 trades)

**Four Categories (0-100 score each):**

1. **Vol-Edge (Pricing Mistake Detection)**
   - Compare implied vs historical volatility
   - Check term structure (short vs long-term options)
   - Technical analysis
   - Edge: When options overpriced → sellers win

2. **Quality (Company Fundamentals)**
   - [Content cut off in PDF]

### Key Insights
- Stopped predicting direction
- Focus on volatility edge (selling overpriced options)
- Risk management through position sizing and Greeks constraints
- Avoid earnings/binary events
- Tight bid-ask spreads only

### Relevance to Iron Dragoon Crypto Project
- Similar scoring/ranking approach applicable
- Data pipeline architecture transferable
- Risk management framework (POP, max loss, diversification)
- Automation and daily scanning methodology
- Consider adding volatility edge analysis to crypto bot

### Notes
- Options credit spreads vs crypto spot/futures different mechanics
- But the "overpriced volatility" concept applies to crypto options
- Scoring engine approach could enhance Iron Dragoon's signal generation

---

## IMPLEMENTATION COMPLETE: scoring_engine.py

**Created:** `Crypto-data/scoring_engine.py` - Four-category scoring system

### Vol-Edge (30% weight)
Crypto adaptation of implied vs historical vol:
- Measures current volatility percentile (7-day vs 90-day history)
- High scores: Extreme vol cooling off OR moderate vol expanding into trend
- Avoid: Low volatility (<20th percentile) = no edge

### Quality (25% weight) 
Crypto Piotroski F-Score adaptation:
- Liquidity health (volume consistency)
- Trend quality (ADX-like directional strength)
- Price stability (avoiding wick-heavy manipulation)
- Volume trend (increasing = accumulation)

### Setup (30% weight)
Technical pattern recognition:
- RSI extremes (25-40 = oversold bounce, 60-75 = momentum)
- Moving average positioning (price > 7/30/90 MA)
- Bollinger Band squeeze detection
- Volume-price divergence confirmation

### Regime (15% weight)
Macro context:
- BTC dominance (alt season vs BTC season)
- Funding rates (negative = bullish)
- BTC correlation (independent movers score higher)
- Market momentum (BTC 7-day trend)

### Signal Strength Classification
- ≥80: STRONG_BUY
- 65-79: BUY  
- 50-64: NEUTRAL
- 35-49: AVOID
- <35: STRONG_AVOID

### Key Adaptations
| Options Concept | Crypto Adaptation |
|-----------------|-------------------|
| Implied vs Historical Vol | Realized vol percentile + trend |
| Piotroski F-Score | Liquidity + trend + stability metrics |
| Credit Spread Criteria | RSI + MA + BB + volume confirmation |
| Macro Regime | BTC dominance + funding + correlation |
| POP ≥60% | Composite score ≥65 |

### Integration
- Call `score_coin()` in Sentry before alerting
- Use `rank_signals()` to prioritize multiple alerts
- Store scores in database for backtesting/optimization
- Filter: Only alert on BUY or STRONG_BUY (composite ≥65)

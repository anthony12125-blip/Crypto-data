# CLAUDE CODE INSTRUCTIONS — Crypto Training Data Formatter

## WHAT YOU'RE DOING
You have 10 zip files containing ~261,000 crypto news articles from CryptoPanic.
Your job: convert ALL of them into MLX LoRA fine-tuning format (JSONL).

## RAW DATA FORMAT
Each zip contains a .txt file. Each line is pipe-delimited:

```
id | timestamp_ms | year | month | ? | date | source | title | body | url | ? | ? | ? | ? | batch_id
```

Example:
```
cp_1522|1640064600000|2021|12||2021-12-20|cryptopanic|Crypto regulation is coming, but Bitcoin traders are still buying the dip|The premium on CME Bitcoin futures dropped to zero, but data shows pro traders are still bullish.|https://cryptopanic.com/news/13721374/...|||||1770705249203
```

Fields we care about:
- Field 1: record ID
- Field 2: timestamp in milliseconds
- Field 6: date (YYYY-MM-DD)
- Field 7: source
- Field 8: title (THE MOST IMPORTANT FIELD)
- Field 9: body/summary
- Field 10: URL

## OUTPUT FORMAT
MLX LoRA expects JSONL with Mistral instruct format:

```json
{"text": "<s>[INST] {prompt} [/INST] {completion} </s>"}
```

## WHAT TO BUILD
Process ALL 10 zip files and generate THREE types of training pairs per article:

### Type 1: Sentiment Analysis
```json
{"text": "<s>[INST] Analyze this crypto news from 2021-12-20: \"Crypto regulation is coming, but Bitcoin traders are still buying the dip\". What is the sentiment and likely market impact? [/INST] Sentiment: slightly_bearish. The headline discusses incoming regulation (bearish catalyst) but notes traders are buying the dip (bullish response). Mixed signal — regulation fears are offset by strong buy-side conviction. Short-term impact: neutral to slightly bearish. If regulation details are moderate, expect recovery. </s>"}
```

### Type 2: Market Context Analysis  
```json
{"text": "<s>[INST] You are a crypto market analyst. On 2021-12-20, this news broke: \"Crypto regulation is coming, but Bitcoin traders are still buying the dip\". The article states: The premium on CME Bitcoin futures dropped to zero, but data shows pro traders are still bullish. What does this tell us about the current market regime and what should a trader do? [/INST] The CME futures premium dropping to zero indicates institutional caution — this is a warning sign. However, pro traders buying the dip shows conviction hasn't broken. The market is in a correction within a bull cycle. A trader should: maintain current positions but tighten stop losses, avoid adding new positions until momentum confirms direction, and watch for BTC dominance shifts that might signal alt rotation. </s>"}
```

### Type 3: Event Classification
```json
{"text": "<s>[INST] Classify this crypto event: \"Crypto regulation is coming, but Bitcoin traders are still buying the dip\" (2021-12-20). Category, severity, and expected duration of impact? [/INST] Category: regulation. Severity: moderate. Expected duration: weeks to months depending on regulatory specifics. This type of headline typically causes 5-15% pullbacks if regulation materializes, but markets have historically adapted to regulatory clarity within 2-4 weeks. </s>"}
```

## SENTIMENT CLASSIFICATION RULES
Use these keyword groups to auto-classify, then generate appropriate completions:

BULLISH: rally, surge, soar, all-time high, ath, breakout, adoption, approve, partnership, launch, institutional, accumulation, upgrade, mainnet, record
BEARISH: crash, plunge, dump, hack, exploit, ban, crackdown, sec charges, lawsuit, fraud, scam, collapse, bankruptcy, liquidat, freeze, halt, suspend
NEUTRAL: report, analysis, update, review, regulation, stablecoin, development, roadmap

## FILTERING RULES
- SKIP lines with titles shorter than 15 characters
- SKIP lines starting with "RT @" that are under 50 chars (social noise)  
- SKIP lines where body is "NULL" or "-" (set body to empty string)
- SKIP lines that don't have at least 10 pipe-delimited fields
- KEEP everything else — even stablecoin news, even minor altcoin news, ALL of it

## OUTPUT FILES — THREE SEPARATE TRAINING SETS

We are training 3 different agents on the SAME data but with DIFFERENT training structures.
This gives each agent a slightly different perspective on the same market history.
All three see the same articles. The dates stay in order within each article. 
But the ORDER the articles are presented to the model during training differs.

### Agent A — Chronological by category
Group articles by category (regulation, security, price_action, defi, etc.) 
and within each category, keep chronological order.
Then interleave the categories. Agent A learns market dynamics category-by-category.

Save to: `data/training/agent_a/`
- `train.jsonl` — 80%
- `valid.jsonl` — 10%  
- `test.jsonl` — 10%

### Agent B — Reverse chronological
Present the most recent data first, oldest data last.
Agent B learns from the present backwards — it prioritizes recent patterns.

Save to: `data/training/agent_b/`
- `train.jsonl` — 80%
- `valid.jsonl` — 10%
- `test.jsonl` — 10%

### Agent C — Random shuffle (but dates within articles preserved)
Fully randomized article order. No chronological or category bias.
Agent C gets a scrambled view — it has to find patterns without temporal crutches.

Save to: `data/training/agent_c/`
- `train.jsonl` — 80%
- `valid.jsonl` — 10%
- `test.jsonl` — 10%

### WHY THREE DIFFERENT ORDERINGS
- Agent A (category-first) will be strong at recognizing category-specific patterns (regulation cycles, hack aftermath patterns)
- Agent B (recent-first) will be strong at weighting recent events more heavily  
- Agent C (random) will be the most generalized, least biased by ordering
- When they vote, their different perspectives catch each other's blind spots
- Same data, three different brains

## IMPORTANT
- Process ALL 10 zip files, not just one
- Generate ALL THREE pair types for each valid article (3x the data)
- Shuffle before splitting
- Log progress: how many records processed, how many pairs generated
- Handle encoding errors gracefully (skip bad lines)
- The completions should be DETAILED and ANALYTICAL, not just one word
- Vary the prompt phrasing so the model learns to respond to different question styles
- Target: ~750,000+ training pairs from 261,000 articles (3 pairs each)
- That's ~750,000 pairs PER AGENT = ~2.25 million total pairs across all three training sets
- Each agent sees the SAME pairs, just in DIFFERENT order

## AFTER FORMATTING
Once the JSONL files are generated, fine-tune THREE separate models:

### Agent A (category-chronological)
```bash
python -m mlx_lm.lora \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --data ./data/training/agent_a \
  --train \
  --adapter-path ./models/llm/agent_a_adapter \
  --batch-size 2 \
  --iters 1000 \
  --learning-rate 2e-5 \
  --lora-layers 16 \
  --save-every 200 \
  --max-seq-length 2048 \
  --grad-checkpoint
```

### Agent B (reverse-chronological)
```bash
python -m mlx_lm.lora \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --data ./data/training/agent_b \
  --train \
  --adapter-path ./models/llm/agent_b_adapter \
  --batch-size 2 \
  --iters 1000 \
  --learning-rate 2e-5 \
  --lora-layers 16 \
  --save-every 200 \
  --max-seq-length 2048 \
  --grad-checkpoint
```

### Agent C (random shuffle)
```bash
python -m mlx_lm.lora \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --data ./data/training/agent_c \
  --train \
  --adapter-path ./models/llm/agent_c_adapter \
  --batch-size 2 \
  --iters 1000 \
  --learning-rate 2e-5 \
  --lora-layers 16 \
  --save-every 200 \
  --max-seq-length 2048 \
  --grad-checkpoint
```

### Then fuse each adapter into standalone models:
```bash
python -m mlx_lm.fuse --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --adapter-path ./models/llm/agent_a_adapter --save-path ./models/llm/crypto_agent_a_fused
python -m mlx_lm.fuse --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --adapter-path ./models/llm/agent_b_adapter --save-path ./models/llm/crypto_agent_b_fused
python -m mlx_lm.fuse --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --adapter-path ./models/llm/agent_c_adapter --save-path ./models/llm/crypto_agent_c_fused
```

Three models. Same knowledge. Different perspectives. They vote.

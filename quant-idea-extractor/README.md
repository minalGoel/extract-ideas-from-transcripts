# Quant Alpha Idea Extractor

Extract quantitative trading ideas, alternative data sources, and strategy signals from Reddit, arXiv, Substack, YouTube, SSRN, and Twitter — unified into a SQLite database, deduplicated with TF-IDF clustering, and exported to CSV/JSON for research prioritisation.

---

## Setup

```bash
# 1. Clone / enter the project
cd quant-idea-extractor

# 2. Create venv
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY and REDDIT_CLIENT_ID / SECRET
```

### Reddit API credentials
Register a free "script" app at <https://www.reddit.com/prefs/apps>. Set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` in `.env`.

---

## Quick Start

```bash
# Collect 50 posts from Reddit (fast, free)
python scripts/run_collect.py --source reddit --limit 50

# Collect arXiv abstracts (last 3 years, all q-fin categories)
python scripts/run_collect.py --source arxiv

# Collect Substack blog posts
python scripts/run_collect.py --source substack

# Collect YouTube transcripts
python scripts/run_collect.py --source youtube --limit 20

# Run Claude extraction on all unprocessed docs ($5 cost cap)
python scripts/run_extract.py --batch-size 20 --max-cost 5.0

# Cluster/deduplicate ideas
python scripts/run_dedup.py

# Export to CSV
python scripts/export.py --format csv --output ideas.csv

# Pipeline stats
python scripts/run_collect.py --stats
```

---

## Full Pipeline

```bash
# Collect from everything
python scripts/run_collect.py --source all --limit 100

# Extract (respects cost cap from config.yaml)
python scripts/run_extract.py

# Dedup
python scripts/run_dedup.py

# Export high-testability ideas
python scripts/export.py \
    --format csv \
    --filter-testability high \
    --filter-confidence high \
    --output high_conviction.csv
```

---

## Export Filters

| Flag | Values |
|---|---|
| `--filter-testability` | `high` \| `medium` \| `low` |
| `--filter-confidence` | `high` \| `medium` \| `low` |
| `--filter-geographic` | `india` \| `us` \| `global` \| `emerging_markets` \| `unspecified` |
| `--filter-type` | `strategy_signal` \| `factor` \| `alt_data_source` \| `market_microstructure` \| `risk_model` \| `execution_technique` \| `infrastructure_tool` \| `other` |

---

## Twitter (Best-Effort)

**Mode A — Nitter scraping** (default, unreliable):
```bash
python scripts/run_collect.py --source twitter
```

**Mode B — Manual import** (recommended):
Export tweets from X archive or a browser extension, then:
```python
# In code / a custom script:
from src.collectors.twitter import TwitterCollector
collector = TwitterCollector(config, mode="import", import_file="tweets.json")
```

---

## Configuration

All tunable parameters live in `config.yaml`. Key knobs:

| Setting | Default | Description |
|---|---|---|
| `extraction.model` | `claude-sonnet-4-0` | Claude model (claude-sonnet-4-20250514) |
| `extraction.max_concurrent_calls` | `5` | Parallel API calls |
| `extraction.max_cost_per_run` | `10.0` | USD hard cap |
| `dedup.similarity_threshold` | `0.6` | Cosine similarity for clustering |
| `reddit.min_score` | `10` | Minimum upvote score |
| `arxiv.years_back` | `3` | How far back to pull papers |

---

## Project Structure

```
quant-idea-extractor/
├── src/
│   ├── collectors/          # One module per platform
│   │   ├── base.py
│   │   ├── reddit.py
│   │   ├── arxiv_collector.py
│   │   ├── substack.py
│   │   ├── youtube.py
│   │   ├── ssrn.py
│   │   └── twitter.py
│   ├── normalizer.py        # Raw → NormalizedDocument
│   ├── extractor.py         # Claude API extraction
│   ├── dedup.py             # TF-IDF + AgglomerativeClustering
│   ├── db.py                # SQLite (aiosqlite)
│   ├── orchestrator.py      # Pipeline coordination
│   └── utils.py             # Rate limiting, retry, logging
├── prompts/
│   └── extraction.txt       # Claude system prompt (edit to iterate)
├── seeds/                   # Channel/subreddit/feed lists (edit freely)
├── scripts/                 # Standalone CLI entry points
├── data/                    # SQLite DB (git-ignored)
├── logs/                    # Log files (git-ignored)
├── config.yaml
└── requirements.txt
```

---

## Cost Estimates

Using `claude-sonnet-4-0` ($3 / 1M input, $15 / 1M output):

| Source | Typical docs | ~Avg tokens/doc | ~Cost per 100 docs |
|---|---|---|---|
| Reddit posts | 200–500 | 2,000 | ~$0.60 |
| arXiv abstracts | 500–2,000 | 600 | ~$0.18 |
| Substack posts | 50–200 | 3,000 | ~$0.90 |
| YouTube transcripts | 100–500 | 4,000 | ~$1.20 |

A full run over ~1,000 documents typically costs **$5–15**.

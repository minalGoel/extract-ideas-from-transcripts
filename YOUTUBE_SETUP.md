# YouTube Transcript → Quant Idea Extraction: Setup Guide

Extract quant trading ideas from 4,392 YouTube transcripts using Claude.

The transcripts are already fetched and cleaned in `youtube_transcripts.parquet`. You just need API credentials and one command.

---

## What you need

- **Python 3.10+**
- **Claude API access** via one of:
  - Azure AI Foundry (recommended if you have it)
  - Anthropic API key
  - AWS Bedrock

---

## Step 1 — Clone & install

```bash
git clone <repo-url>
cd extract-ideas-from-transcripts

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Step 2 — Set your credentials

Pick your backend and export the env vars. You only need ONE of these:

**Azure AI Foundry:**
```bash
export AZURE_ANTHROPIC_ENDPOINT="https://<resource>-<id>-<region>.services.ai.azure.com/anthropic/"
export AZURE_ANTHROPIC_API_KEY="your-key"
export AZURE_ANTHROPIC_DEPLOYMENT="claude-opus-4-6"
```

**Anthropic direct:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**AWS Bedrock:**
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

---

## Step 3 — Test with a single transcript

> **Do this first!** Processes one video, prints the result, writes nothing to disk.

```bash
# Azure (default)
python extract_from_parquet.py --test

# Or explicitly pick a backend
python extract_from_parquet.py --backend azure --test
python extract_from_parquet.py --backend anthropic --test
python extract_from_parquet.py --backend bedrock --test
```

If you see extracted ideas printed to your terminal, you're good.

---

## Step 4 — Full run

```bash
# Default: azure backend, 3 workers, $50 cost cap
python extract_from_parquet.py

# Or pick a backend
python extract_from_parquet.py --backend anthropic
python extract_from_parquet.py --backend bedrock
```

Progress is saved after every video to `extract_ideas_progress.jsonl`. If the run is interrupted (Ctrl+C, crash, rate limit), just re-run the same command — it picks up where it left off.

---

## Outputs

| File | Description |
|---|---|
| `extract_ideas_progress.jsonl` | Checkpoint — one JSON line per video processed |
| `extracted_ideas.csv` | Final output — one row per idea, with video metadata |

---

## Tuning

Edit the constants at the top of `extract_from_parquet.py`:

| Setting | Default | What it does |
|---|---|---|
| `MAX_WORKERS` | `3` | Concurrent API calls (lower = safer for rate limits) |
| `MAX_TRANSCRIPT` | `15,000` | Chars per transcript (trim long ones to save cost) |
| `MAX_COST_USD` | `50.0` | Hard stop (approximate, based on token estimates) |

---

## Cost estimate

Using Opus ($15/1M input, $75/1M output):
- ~4,000 tokens per transcript avg
- ~$0.27 per video
- ~$1,200 for all 4,392 videos

Using Sonnet ($3/1M input, $15/1M output):
- Same token volume
- ~$0.05 per video
- ~$230 for all 4,392 videos

To use Sonnet instead, set the env var for your backend (e.g. `AZURE_ANTHROPIC_DEPLOYMENT=claude-sonnet-4-6`).

---

## Backend details

See [agents.md](agents.md) for full env var reference, model IDs, and troubleshooting for each provider.

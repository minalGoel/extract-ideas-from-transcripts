# Project: YouTube Transcript → Quant Idea Extractor

Extracts quantitative trading ideas from 4,400+ YouTube video transcripts using Claude.

## Key files

| File | Purpose |
|---|---|
| `extract_from_parquet.py` | Main script — reads parquet, calls Claude, writes progress + CSV |
| `llm_backends.py` | Swappable LLM clients: Anthropic, Bedrock, Azure (same interface) |
| `youtube_transcripts.parquet` | 4,392 transcripts (pre-cleaned, ready to process) |
| `quant-idea-extractor/prompts/extraction.txt` | System prompt for idea extraction |
| `temp_metadata_yt_playlist.py` | Utility to list videos in a YouTube playlist |
| `requirements.txt` | Python dependencies |

## How to run

```bash
# Test (1 video, no files written)
python extract_from_parquet.py --test

# Full run (default: azure backend)
python extract_from_parquet.py

# Pick a backend
python extract_from_parquet.py --backend azure      # default
python extract_from_parquet.py --backend anthropic
python extract_from_parquet.py --backend bedrock
```

## Idempotency

Progress is checkpointed to `extract_ideas_progress.jsonl` after every video. Rerunning skips already-processed video_ids. Safe to kill and restart.

## Env vars by backend

- **Azure**: `AZURE_ANTHROPIC_ENDPOINT`, `AZURE_ANTHROPIC_API_KEY`, `AZURE_ANTHROPIC_DEPLOYMENT`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Bedrock**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

## Conventions

- Don't modify `youtube_transcripts.parquet` — it's the clean source data
- `extract_ideas_progress.jsonl` is the checkpoint — don't delete mid-run
- `extracted_ideas.csv` is recompiled from the progress file on every run
- System prompt lives in `quant-idea-extractor/prompts/extraction.txt` — edit to iterate on extraction quality

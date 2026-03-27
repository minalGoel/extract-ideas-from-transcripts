# YouTube Transcript → Quant Idea Extraction: Setup Guide

Extract quant trading ideas from YouTube videos/playlists using Claude API.

---

## What you need

- **Python 3.10+**
- **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com)
- No YouTube API key required (uses `yt-dlp` to pull transcripts for free)

---

## Step 1 — Clone & install

```bash
git clone <repo-url>
cd quant-idea-extractor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Step 2 — Add your API key

```bash
cp .env.example .env
```

Open `.env` and set:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

That's the only key needed for YouTube.

---

## Step 3 — Set your playlist / channel

Edit `seeds/youtube_channels.yaml`. Add a playlist URL or channel URL:

```yaml
channels:
  # Playlist URL (yt-dlp handles both channels and playlists)
  - name: "My Playlist"
    url: "https://youtube.com/playlist?list=PLFj8mcG4JCG9a7wnLdUG-EcjcTJ7lgINM"

  # Or a channel URL
  - name: "QuantPy"
    url: "https://www.youtube.com/@QuantPy/videos"
```

To see what videos are in a playlist before running (no Claude calls, no DB writes):

```bash
python temp_metadata_yt_playlist.py
# Outputs: temp_metadata_yt_playlist.csv with all video titles + links
```

---

## Step 4 — Collect transcripts

```bash
cd quant-idea-extractor

# Collect transcripts from all channels in youtube_channels.yaml
python scripts/run_collect.py --source youtube

# Or limit to first N videos per channel (faster for testing)
python scripts/run_collect.py --source youtube --limit 10
```

This fetches auto-generated or manual subtitles via `yt-dlp` and stores them in a local SQLite DB (`data/quant_ideas.db`). No API calls yet, no cost.

> Videos shorter than 3 minutes or with no available transcript are skipped.

---

## Step 5 — Extract ideas with Claude

```bash
# $5 cost cap (safe default for ~400 videos)
python scripts/run_extract.py --batch-size 20 --max-cost 5.0

# Higher cap for larger runs
python scripts/run_extract.py --batch-size 50 --max-cost 20.0
```

This calls `claude-sonnet-4-0` on each transcript and extracts structured quant ideas.

**Cost estimate:** ~$1.20 per 100 YouTube videos (avg 4k tokens/transcript at $3/1M in, $15/1M out).

---

## Step 6 — Export results

```bash
# All ideas
python scripts/export.py --format csv --output ideas.csv

# High-conviction only
python scripts/export.py \
    --format csv \
    --filter-testability high \
    --filter-confidence high \
    --output high_conviction.csv
```

---

## Tuning

Key settings in `config.yaml`:

| Setting | Default | What it does |
|---|---|---|
| `youtube.max_videos_per_channel` | `100` | Max videos to pull per channel/playlist |
| `youtube.min_duration_seconds` | `180` | Skip videos shorter than this |
| `extraction.max_cost_per_run` | `10.0` | Hard USD cap per extraction run |
| `extraction.max_concurrent_calls` | `5` | Parallel Claude API calls |

---

## Check pipeline stats at any time

```bash
python scripts/run_collect.py --stats
```

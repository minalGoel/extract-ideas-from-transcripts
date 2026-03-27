"""
Extract quant trading ideas from youtube_transcripts.parquet using Claude.

Reads transcripts from the parquet file, calls Claude for each one, and writes
structured ideas to extracted_ideas.csv.

Idempotent: progress is checkpointed to extract_ideas_progress.jsonl after
every video. Re-running the script skips already-processed video_ids, so it
is safe to kill and restart at any time.

──────────────────────────────────────────────────────────────────────────────
QUICK START
──────────────────────────────────────────────────────────────────────────────

  # 1. Set your API key (Anthropic direct):
  export ANTHROPIC_API_KEY=sk-ant-...

  # 2. Run:
  python extract_from_parquet.py

  # 3. To use Bedrock or Azure instead, edit the BACKEND section below.

──────────────────────────────────────────────────────────────────────────────
OUTPUTS
──────────────────────────────────────────────────────────────────────────────
  extract_ideas_progress.jsonl  — checkpoint file (one line per video)
  extracted_ideas.csv           — final flat CSV, one row per idea

──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# BACKEND — swap this one import to change providers
# ──────────────────────────────────────────────────────────────────────────────

from llm_backends import AnthropicBackend  # ← change to BedrockBackend or AzureBackend
BACKEND = AnthropicBackend()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

PARQUET_FILE    = "youtube_transcripts.parquet"
PROGRESS_FILE   = "extract_ideas_progress.jsonl"
OUTPUT_CSV      = "extracted_ideas.csv"
PROMPT_FILE     = Path("quant-idea-extractor/prompts/extraction.txt")

MAX_WORKERS     = 3       # concurrent API calls (lower = safer for rate limits)
MAX_TRANSCRIPT  = 15_000  # chars — trim very long transcripts to save cost
MAX_COST_USD    = 50.0    # hard stop; only tracked for Anthropic direct (approx)

# Rough token/cost estimates for Opus ($15/$75 per 1M in/out)
_PRICE_IN  = 15.0  / 1_000_000
_PRICE_OUT = 75.0  / 1_000_000


# ──────────────────────────────────────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8").strip()
    # Minimal fallback if prompt file is missing
    return (
        "You are a quantitative trading research analyst. "
        "Extract all quantitative trading ideas from the transcript and return "
        "a JSON object with an 'ideas' array. Each idea must have: "
        "idea_type, name, description, mechanism, data_requirements, "
        "testability, asset_class, geographic_relevance, time_horizon, "
        "novelty_assessment, confidence, source_quote, tags."
    )


# ──────────────────────────────────────────────────────────────────────────────
# IDEMPOTENCY — progress file read/write
# ──────────────────────────────────────────────────────────────────────────────

def load_done_ids() -> set[str]:
    """Return set of video_ids already in the progress file."""
    done: set[str] = set()
    if not Path(PROGRESS_FILE).exists():
        return done
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["video_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_progress(video_id: str, status: str, ideas: list, error_msg: str = "") -> None:
    """Append one result line to the progress file (atomic-enough for our use)."""
    record = {
        "video_id":  video_id,
        "status":    status,       # "ok" | "error" | "skip"
        "ideas":     ideas,
        "error_msg": error_msg,
        "ts":        datetime.now(timezone.utc).isoformat(),
    }
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_ideas(raw: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```", "", cleaned).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group())
        return [i for i in data.get("ideas", []) if isinstance(i, dict)]
    except json.JSONDecodeError:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# PER-VIDEO PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_video(row: dict, system_prompt: str) -> tuple[str, str, list, str]:
    """
    Call the LLM for one video. Returns (video_id, status, ideas, error_msg).
    Retries up to 3 times on transient errors with exponential back-off.
    """
    video_id   = row["video_id"]
    title      = row.get("title") or ""
    transcript = (row.get("transcript_text") or "").strip()

    if not transcript:
        return video_id, "skip", [], "no transcript"

    transcript = transcript[:MAX_TRANSCRIPT]
    user_msg   = f"Title: {title}\n\n---\n\n{transcript}"

    last_err = ""
    for attempt in range(3):
        try:
            raw     = BACKEND.complete(system_prompt, user_msg)
            ideas   = parse_ideas(raw)
            return video_id, "ok", ideas, ""
        except Exception as exc:
            last_err = str(exc)
            wait = 2 ** attempt * 5   # 5s, 10s, 20s
            print(f"  [{video_id}] attempt {attempt+1} failed: {exc} — retrying in {wait}s")
            time.sleep(wait)

    return video_id, "error", [], last_err


# ──────────────────────────────────────────────────────────────────────────────
# FINAL CSV COMPILATION
# ──────────────────────────────────────────────────────────────────────────────

IDEA_FIELDS = [
    "idea_type", "name", "description", "mechanism", "data_requirements",
    "testability", "asset_class", "geographic_relevance", "time_horizon",
    "novelty_assessment", "confidence", "source_quote", "tags",
]

def compile_csv(df_meta: pd.DataFrame) -> None:
    """Read progress file and write flat CSV (one row per idea)."""
    meta = df_meta.set_index("video_id")[
        ["title", "channel_name", "video_url", "upload_date"]
    ].to_dict("index")

    rows = []
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["status"] != "ok":
                continue
            vid = record["video_id"]
            m   = meta.get(vid, {})
            for idea in record["ideas"]:
                row = {
                    "video_id":    vid,
                    "title":       m.get("title", ""),
                    "channel":     m.get("channel_name", ""),
                    "video_url":   m.get("video_url", ""),
                    "upload_date": m.get("upload_date", ""),
                }
                for field in IDEA_FIELDS:
                    val = idea.get(field, "")
                    row[field] = json.dumps(val) if isinstance(val, list) else val
                rows.append(row)

    if not rows:
        print("No ideas to export.")
        return

    fieldnames = (
        ["video_id", "title", "channel", "video_url", "upload_date"] + IDEA_FIELDS
    )
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} ideas across {len(set(r['video_id'] for r in rows))} videos → {OUTPUT_CSV}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_parquet(PARQUET_FILE)
    system_prompt = load_system_prompt()

    done_ids  = load_done_ids()
    pending   = df[~df["video_id"].isin(done_ids)].to_dict("records")

    total      = len(df)
    n_done     = len(done_ids)
    n_pending  = len(pending)

    print(f"Parquet: {total} videos total")
    print(f"Already done (progress file): {n_done}")
    print(f"To process: {n_pending}")
    print(f"Backend: {BACKEND.__class__.__name__}  model: {BACKEND.model}")
    print(f"Workers: {MAX_WORKERS}  cost cap: ${MAX_COST_USD}\n")

    if n_pending == 0:
        print("Nothing to do — all videos already processed.")
        compile_csv(df)
        return

    completed   = 0
    errors      = 0
    skipped     = 0
    total_ideas = 0
    est_cost    = 0.0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_video, row, system_prompt): row["video_id"]
            for row in pending
        }

        for future in as_completed(futures):
            vid_id = futures[future]
            try:
                video_id, status, ideas, error_msg = future.result()
            except Exception as exc:
                # Unexpected — shouldn't happen since process_video catches errors
                append_progress(vid_id, "error", [], str(exc))
                errors += 1
                continue

            append_progress(video_id, status, ideas, error_msg)

            if status == "ok":
                completed  += 1
                n_ideas     = len(ideas)
                total_ideas += n_ideas
                # Rough cost estimate (chars / 4 ≈ tokens)
                transcript_chars = len(
                    next((r["transcript_text"] or "" for r in pending if r["video_id"] == video_id), "")
                )
                est_cost += (transcript_chars / 4) * _PRICE_IN + 1000 * _PRICE_OUT
                print(f"  ✓ {video_id}: {n_ideas} ideas  "
                      f"(done {completed}/{n_pending}, est cost ${est_cost:.3f})")

                if est_cost >= MAX_COST_USD:
                    print(f"\nCost cap ${MAX_COST_USD} reached — stopping.")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

            elif status == "skip":
                skipped += 1
            else:
                errors += 1
                print(f"  ✗ {video_id}: {error_msg}")

    print(f"\nDone. {completed} processed, {skipped} skipped, {errors} errors")
    print(f"Total ideas extracted this run: {total_ideas}")
    compile_csv(df)


if __name__ == "__main__":
    main()

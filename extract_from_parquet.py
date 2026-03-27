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

  # 1. Test with Azure (no files written):
  python extract_from_parquet.py --backend azure --test

  # 2. Full run with Azure:
  python extract_from_parquet.py --backend azure

  # 3. Other backends:
  python extract_from_parquet.py --backend anthropic   # needs ANTHROPIC_API_KEY
  python extract_from_parquet.py --backend bedrock      # needs AWS creds

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
# BACKEND — selected via --backend flag (default: azure)
# ──────────────────────────────────────────────────────────────────────────────

from llm_backends import AnthropicBackend, BedrockBackend, AzureBackend

def _pick_backend() -> AnthropicBackend | BedrockBackend | AzureBackend:
    name = "azure"  # default
    for i, arg in enumerate(sys.argv):
        if arg == "--backend" and i + 1 < len(sys.argv):
            name = sys.argv[i + 1].lower()
    backends = {
        "anthropic": AnthropicBackend,
        "bedrock":   BedrockBackend,
        "azure":     AzureBackend,
    }
    cls = backends.get(name)
    if cls is None:
        print(f"Unknown backend: {name!r}. Choose from: {', '.join(backends)}")
        sys.exit(1)
    return cls()

BACKEND = _pick_backend()

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
    raise FileNotFoundError(
        f"System prompt file not found: {PROMPT_FILE}. Please create the file at this path."
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

def run_test() -> None:
    """
    --test mode: process one video and print results to stdout.
    Does NOT write to the progress file — completely side-effect-free.
    """
    df = pd.read_parquet(PARQUET_FILE)
    system_prompt = load_system_prompt()

    # Find first row that actually has a transcript
    test_row = None
    for _, row in df.iterrows():
        if (row.get("transcript_text") or "").strip():
            test_row = row.to_dict()
            break

    if test_row is None:
        print("ERROR: No video with a transcript found in the parquet file.")
        sys.exit(1)

    video_id = test_row["video_id"]
    title    = test_row.get("title") or "(no title)"

    print("=" * 70)
    print("TEST MODE — processing 1 video, no files written")
    print("=" * 70)
    print(f"Backend:  {BACKEND.__class__.__name__}  model: {BACKEND.model}")
    print(f"Video:    {video_id}")
    print(f"Title:    {title}")
    print(f"Transcript length: {len((test_row.get('transcript_text') or '').strip())} chars")
    print("-" * 70)
    print("Calling API...")

    video_id, status, ideas, error_msg = process_video(test_row, system_prompt)

    if status == "error":
        print(f"\nERROR: {error_msg}")
        sys.exit(1)

    if status == "skip":
        print(f"\nSKIPPED: {error_msg}")
        sys.exit(0)

    print(f"\nExtracted {len(ideas)} idea(s) from: {title}\n")
    print(json.dumps({"ideas": ideas}, indent=2))
    print("\n" + "=" * 70)
    print("API key works. You're ready for a full run:")
    print("  python extract_from_parquet.py")
    print("=" * 70)


def main() -> None:
    # ── --test flag: quick single-video check, no side effects ───────────
    if "--test" in sys.argv:
        run_test()
        return

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

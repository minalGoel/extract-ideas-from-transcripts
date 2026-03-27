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
import random
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


def append_progress(video_id: str, status: str, result: dict | None, error_msg: str = "") -> None:
    """Append one result line to the progress file (atomic-enough for our use).

    `result` is the full parsed JSON from the API — stored as-is, no typecasting.
    """
    record = {
        "video_id":  video_id,
        "status":    status,       # "ok" | "error" | "skip"
        "result":    result,       # full API response dict, or None
        "error_msg": error_msg,
        "ts":        datetime.now(timezone.utc).isoformat(),
    }
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_response(raw: str) -> dict | None:
    """Extract the full JSON object from the API response, as-is."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```", "", cleaned).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# PER-VIDEO PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_video(row: dict, system_prompt: str) -> tuple[str, str, dict | None, str]:
    """
    Call the LLM for one video. Returns (video_id, status, result_dict, error_msg).
    `result_dict` is the full parsed JSON from the API — stored as-is.
    Retries up to 3 times on transient errors with exponential back-off.
    """
    video_id   = row["video_id"]
    title      = row.get("title") or ""
    transcript = (row.get("transcript_text") or "").strip()

    if not transcript:
        return video_id, "skip", None, "no transcript"

    transcript = transcript[:MAX_TRANSCRIPT]
    user_msg   = f"""\
                Video: {title}
                video_id: {video_id}

                {transcript}"""

    last_err = ""
    for attempt in range(3):
        try:
            raw    = BACKEND.complete(system_prompt, user_msg)
            result = parse_response(raw)
            if result is None:
                raise ValueError("Could not parse JSON from API response")
            return video_id, "ok", result, ""
        except Exception as exc:
            last_err = str(exc)
            wait = 2 ** attempt * 5   # 5s, 10s, 20s
            print(f"  [{video_id}] attempt {attempt+1} failed: {exc} — retrying in {wait}s")
            time.sleep(wait)

    return video_id, "error", None, last_err


# ──────────────────────────────────────────────────────────────────────────────
# FINAL CSV COMPILATION
# ──────────────────────────────────────────────────────────────────────────────

CLAIM_FIELDS = [
    "cat", "what", "why", "needs", "anchor",
    "asset", "market", "freq", "entities", "numbers", "india",
]

# Top-level fields from the API response (per-video, not per-claim)
TOP_LEVEL_FIELDS = ["quality", "theme", "gap"]

def compile_csv(df_meta: pd.DataFrame) -> None:
    """Read progress file and write flat CSV (one row per claim)."""
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
            vid    = record["video_id"]
            result = record.get("result") or {}
            m      = meta.get(vid, {})

            # Speaker info
            speaker = result.get("speaker") or {}

            for claim in result.get("claims", []):
                row = {
                    "video_id":            vid,
                    "title":               m.get("title", ""),
                    "channel":             m.get("channel_name", ""),
                    "video_url":           m.get("video_url", ""),
                    "upload_date":         m.get("upload_date", ""),
                    "speaker_name":        speaker.get("name", ""),
                    "speaker_affiliation": speaker.get("affiliation", ""),
                    "speaker_type":        speaker.get("type", ""),
                }
                for field in TOP_LEVEL_FIELDS:
                    row[field] = result.get(field, "")
                for field in CLAIM_FIELDS:
                    val = claim.get(field, "")
                    row[field] = json.dumps(val) if isinstance(val, list) else val
                rows.append(row)

    if not rows:
        print("No claims to export.")
        return

    fieldnames = (
        ["video_id", "title", "channel", "video_url", "upload_date",
         "speaker_name", "speaker_affiliation", "speaker_type"]
        + TOP_LEVEL_FIELDS + CLAIM_FIELDS
    )
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} claims across {len(set(r['video_id'] for r in rows))} videos → {OUTPUT_CSV}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_test() -> None:
    """
    --test mode: randomly sample 5 videos and process them.
    Does NOT write to the progress file — completely side-effect-free.
    """
    df = pd.read_parquet(PARQUET_FILE)
    system_prompt = load_system_prompt()

    # Filter to rows that actually have a transcript
    has_transcript = df[df["transcript_text"].fillna("").str.strip().astype(bool)]

    if has_transcript.empty:
        print("ERROR: No video with a transcript found in the parquet file.")
        sys.exit(1)

    n_sample = min(5, len(has_transcript))
    sample   = has_transcript.sample(n=n_sample, random_state=random.randint(0, 2**31))

    print("=" * 70)
    print(f"TEST MODE — processing {n_sample} random video(s), no files written")
    print("=" * 70)
    print(f"Backend: {BACKEND.__class__.__name__}  model: {BACKEND.model}\n")

    ok = 0
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        test_row = row.to_dict()
        video_id = test_row["video_id"]
        title    = test_row.get("title") or "(no title)"
        t_len    = len((test_row.get("transcript_text") or "").strip())

        print(f"[{i}/{n_sample}] {video_id}  {title}")
        print(f"  Transcript: {t_len} chars")
        print(f"  Calling API...")

        video_id, status, result, error_msg = process_video(test_row, system_prompt)

        if status == "error":
            print(f"  ERROR: {error_msg}\n")
            continue
        if status == "skip":
            print(f"  SKIPPED: {error_msg}\n")
            continue

        n_claims = len((result or {}).get("claims", []))
        print(f"  Extracted {n_claims} claim(s)")
        print(json.dumps(result, indent=2))
        print()
        ok += 1

    print("=" * 70)
    print(f"Done: {ok}/{n_sample} succeeded.")
    if ok > 0:
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
                video_id, status, result, error_msg = future.result()
            except Exception as exc:
                # Unexpected — shouldn't happen since process_video catches errors
                append_progress(vid_id, "error", None, str(exc))
                errors += 1
                continue

            append_progress(video_id, status, result, error_msg)

            if status == "ok":
                completed  += 1
                n_claims    = len((result or {}).get("claims", []))
                total_ideas += n_claims
                # Rough cost estimate (chars / 4 ≈ tokens)
                transcript_chars = len(
                    next((r["transcript_text"] or "" for r in pending if r["video_id"] == video_id), "")
                )
                est_cost += (transcript_chars / 4) * _PRICE_IN + 1000 * _PRICE_OUT
                print(f"  ✓ {video_id}: {n_claims} claims  "
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
    print(f"Total claims extracted this run: {total_ideas}")
    compile_csv(df)


if __name__ == "__main__":
    main()

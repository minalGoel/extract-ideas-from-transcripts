#!/usr/bin/env python3
"""
Fetch transcripts for all videos in youtube_video_metadata.csv and store
results in a parquet file.

Design
------
- Idempotent: each video writes one JSON file to a temp folder on success or
  failure.  Re-running skips already-processed video IDs.
- Crash-safe: temp files are written immediately; a separate --merge-only
  pass assembles the parquet from whatever is present.
- Concurrency: asyncio + ThreadPoolExecutor.  Defaults to 3 workers with a
  small random delay per request — enough parallelism to be fast, not enough
  to trigger YouTube rate limits.

Temp file schema (one per video_id)
------------------------------------
All columns from the metadata CSV plus:
  transcript_text       str | null   — joined plain text
  transcript_language   str | null   — e.g. "en"
  transcript_generated  bool | null  — True = auto-generated captions
  transcript_error      str | null   — error message if fetch failed
  fetched_at            str          — ISO-8601 UTC timestamp

Usage
-----
  python youtube_fetch_transcripts.py
  python youtube_fetch_transcripts.py --workers 3 --input youtube_video_metadata.csv
  python youtube_fetch_transcripts.py --merge-only          # merge temp → parquet, skip fetch
  python youtube_fetch_transcripts.py --retry-errors        # re-fetch previously errored videos
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Transcript fetching via youtube-transcript-api
#
# Uses an authenticated requests.Session (loaded from a Netscape cookies.txt)
# so YouTube treats requests as a real user rather than a bot.  Much faster
# than yt-dlp: one list() call + one fetch() call per video vs. a full
# video-page download + player-JS parse.
# ---------------------------------------------------------------------------

import requests
from http.cookiejar import MozillaCookieJar

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)


def _make_session(cookies_file: str | None) -> requests.Session:
    """Build a requests.Session with Chrome headers and optional cookie auth."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    if cookies_file:
        cj = MozillaCookieJar()
        cj.load(cookies_file, ignore_discard=True, ignore_expires=True)
        session.cookies = cj
    return session


# English transcript language codes in priority order.
# Covers plain English, regional variants, and YouTube's auto-translated codes.
_ENGLISH_LANGS = ["en", "en-IN", "en-US", "en-GB", "en-AU", "en-CA",
                  "en-en-IN", "en-en", "en-orig"]


def _fetch_transcript_sync(
    video_id: str,
    languages: list[str],
    cookies_from_browser: str | None = None,   # kept for API compat, unused
    max_429_retries: int = 6,
    backoff_base_seconds: float = 30.0,
    backoff_max_seconds: float = 15 * 60.0,
    cookies_file: str | None = None,
) -> dict:
    """
    Fetch transcript for one video using youtube-transcript-api.

    Strategy:
      1. Try to find a direct English transcript (any regional variant).
      2. Fall back to the first available transcript + translate to English.
      3. Retry on IpBlocked / RequestBlocked with exponential backoff.

    cookies_file: path to a Netscape cookies.txt from a YouTube account.
    All requests go directly to youtube.com — no proxies or external servers.
    """
    session = _make_session(cookies_file)
    # NOTE: per youtube-transcript-api docs, one instance per thread.
    ytt = YouTubeTranscriptApi(http_client=session)

    last_exc: Exception | None = None

    for attempt in range(max_429_retries + 1):
        try:
            tlist = ytt.list(video_id)

            lang_code = "unknown"
            is_generated = True

            # --- try direct English variants first ---
            try:
                t = tlist.find_transcript(_ENGLISH_LANGS)
                fetched = t.fetch()
                lang_code = t.language_code
                is_generated = t.is_generated
            except NoTranscriptFound:
                # --- fall back: first available transcript, translated to English ---
                first = next(iter(tlist), None)
                if first is None:
                    return {
                        "transcript_text": None,
                        "transcript_language": None,
                        "transcript_generated": None,
                        "transcript_error": "NoTranscriptFound",
                    }
                try:
                    t_en = first.translate("en")
                    fetched = t_en.fetch()
                    lang_code = f"{first.language_code}→en"
                    is_generated = first.is_generated
                except Exception:
                    # Translation unavailable — take the raw transcript as-is
                    fetched = first.fetch()
                    lang_code = first.language_code
                    is_generated = first.is_generated

            text = " ".join(s.text for s in fetched if s.text.strip())

            if not text.strip():
                return {
                    "transcript_text": None,
                    "transcript_language": lang_code,
                    "transcript_generated": is_generated,
                    "transcript_error": "EmptyTranscript",
                }

            return {
                "transcript_text": text,
                "transcript_language": lang_code,
                "transcript_generated": is_generated,
                "transcript_error": None,
            }

        except (TranscriptsDisabled, VideoUnavailable) as exc:
            # Permanent failures — don't retry
            return {
                "transcript_text": None,
                "transcript_language": None,
                "transcript_generated": None,
                "transcript_error": f"{type(exc).__name__}: {exc}",
            }

        except (IpBlocked, RequestBlocked, Exception) as exc:
            last_exc = exc
            msg = str(exc) or type(exc).__name__
            is_rate_limited = (
                isinstance(exc, (IpBlocked, RequestBlocked))
                or "429" in msg
                or "Too Many Requests" in msg
            )
            if not is_rate_limited or attempt >= max_429_retries:
                break

            sleep_s = min(backoff_max_seconds, backoff_base_seconds * (2 ** attempt))
            sleep_s *= random.uniform(0.8, 1.2)
            print(
                f"  [blocked] {video_id} cooldown {sleep_s:.0f}s "
                f"(attempt {attempt + 1}/{max_429_retries}  {type(exc).__name__})",
                flush=True,
            )
            time.sleep(sleep_s)

    return {
        "transcript_text": None,
        "transcript_language": None,
        "transcript_generated": None,
        "transcript_error": f"ytt: {type(last_exc).__name__}: {last_exc}",
    }


# ---------------------------------------------------------------------------
# Temp file helpers
# ---------------------------------------------------------------------------

def _temp_path(temp_dir: Path, video_id: str) -> Path:
    return temp_dir / f"{video_id}.json"


def _write_temp(temp_dir: Path, record: dict) -> None:
    path = _temp_path(temp_dir, record["video_id"])
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def _already_done(temp_dir: Path, video_id: str, retry_errors: bool) -> bool:
    p = _temp_path(temp_dir, video_id)
    if not p.exists():
        return False
    if retry_errors:
        try:
            data = json.loads(p.read_text())
            return data.get("transcript_error") is None  # only skip successes
        except Exception:
            return False
    return True


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------

async def _process_video(
    executor: ThreadPoolExecutor,
    sem: asyncio.Semaphore,
    row: dict,
    temp_dir: Path,
    languages: list[str],
    counter: list[int],   # [done, total, errors] — mutated in-place
    delay_range: tuple[float, float],
    cookies_from_browser: str | None,
    max_429_retries: int,
    backoff_base_seconds: float,
    backoff_max_seconds: float,
    cookies_file: str | None = None,
) -> None:
    async with sem:
        await asyncio.sleep(random.uniform(*delay_range))
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(
            executor,
            _fetch_transcript_sync,
            row["video_id"],
            languages,
            cookies_from_browser,
            max_429_retries,
            backoff_base_seconds,
            backoff_max_seconds,
            cookies_file,
        )

    record = {**row, **transcript, "fetched_at": datetime.now(timezone.utc).isoformat()}
    _write_temp(temp_dir, record)

    counter[0] += 1
    if transcript["transcript_error"]:
        counter[2] += 1
        status = f"SKIP ({transcript['transcript_error'][:40]})"
    else:
        words = len((transcript["transcript_text"] or "").split())
        status = f"OK   {words} words"

    print(
        f"  [{counter[0]:>4}/{counter[1]}] {status:50s} {row['video_id']}  {row['title'][:50]}"
    )


# ---------------------------------------------------------------------------
# Merge temp → parquet
# ---------------------------------------------------------------------------

def merge_to_parquet(temp_dir: Path, output_path: Path) -> int:
    import pandas as pd

    files = sorted(temp_dir.glob("*.json"))
    if not files:
        print("No temp files found — nothing to merge.")
        return 0

    print(f"Merging {len(files)} temp files → {output_path} …")
    records = []
    for f in files:
        try:
            records.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"  [WARN] Could not read {f.name}: {exc}")

    df = pd.DataFrame(records)

    # Tidy up column types
    for col in ("duration_seconds", "view_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Written: {output_path}  ({len(df)} rows, {len(df.columns)} columns)")

    # Summary
    if "transcript_error" in df.columns:
        n_ok  = df["transcript_error"].isna().sum()
        n_err = df["transcript_error"].notna().sum()
        print(f"  Transcripts: {n_ok} ok, {n_err} failed/unavailable")

    return len(df)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def read_metadata_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video_id") or "").strip()
            if vid:
                rows.append(dict(row))
    return rows


async def run(
    input_path: Path,
    temp_dir: Path,
    output_path: Path,
    workers: int,
    languages: list[str],
    retry_errors: bool,
    delay_range: tuple[float, float],
    cookies_from_browser: str | None,
    max_429_retries: int,
    backoff_base_seconds: float,
    backoff_max_seconds: float,
    cookies_file: str | None = None,
) -> None:
    all_rows = read_metadata_csv(input_path)
    print(f"Loaded {len(all_rows)} videos from {input_path}")

    temp_dir.mkdir(parents=True, exist_ok=True)

    # Filter to videos not yet processed
    todo = [r for r in all_rows if not _already_done(temp_dir, r["video_id"], retry_errors)]
    already_done = len(all_rows) - len(todo)
    print(
        f"Already processed: {already_done}  |  To fetch: {len(todo)}  "
        f"(workers={workers}, delay={delay_range[0]:.1f}–{delay_range[1]:.1f}s)"
    )

    if todo:
        sem     = asyncio.Semaphore(workers)
        counter = [0, len(todo), 0]   # [done, total, errors]
        t0 = time.monotonic()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = [
                _process_video(
                    executor, sem, row, temp_dir, languages,
                    counter, delay_range, cookies_from_browser,
                    max_429_retries, backoff_base_seconds, backoff_max_seconds,
                    cookies_file,
                )
                for row in todo
            ]
            await asyncio.gather(*tasks)

        elapsed = time.monotonic() - t0
        print(
            f"\nFetch complete: {counter[0]} processed, {counter[2]} errors "
            f"in {elapsed:.0f}s ({elapsed/max(counter[0],1):.1f}s/video)"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch YouTube transcripts and store as parquet."
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("youtube_video_metadata.csv"),
        help="Input metadata CSV (default: youtube_video_metadata.csv)",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("youtube_transcripts.parquet"),
        help="Output parquet file (default: youtube_transcripts.parquet)",
    )
    p.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("temp_transcripts"),
        help="Temp folder for per-video JSON files (default: temp_transcripts/)",
    )
    p.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Concurrent transcript fetches (default: 1 — safest without cookies)",
    )
    p.add_argument(
        "--languages",
        nargs="+",
        default=["en", "hi"],
        help="Preferred transcript languages in order (default: en hi)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        metavar="SECS",
        help="Base random delay between requests in seconds (default: 2.0); "
             "actual delay is uniform(delay, delay*2)",
    )
    p.add_argument(
        "--max-429-retries",
        type=int,
        default=6,
        metavar="N",
        help="How many times to retry a video when YouTube returns HTTP 429 "
             "(default: 6). Retries use exponential cooldown.",
    )
    p.add_argument(
        "--backoff-base-seconds",
        type=float,
        default=30.0,
        metavar="SECS",
        help="Initial cooldown after a 429 before retrying (default: 30).",
    )
    p.add_argument(
        "--backoff-max-seconds",
        type=float,
        default=900.0,
        metavar="SECS",
        help="Maximum cooldown cap after repeated 429s (default: 900 = 15 min).",
    )
    p.add_argument(
        "--cookies-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a Netscape cookies.txt file from a throwaway YouTube account. "
             "Helps avoid 429 rate limits. Export only youtube.com cookies — "
             "no personal Google account data needed.",
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip fetching; just merge existing temp files into parquet and exit",
    )
    p.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-fetch videos whose previous attempt recorded an error",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete the temp folder after merging (default: delete)",
    )
    args = p.parse_args()

    if not args.merge_only and not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    delay_range = (args.delay, args.delay * 2)

    if not args.merge_only:
        asyncio.run(
            run(
                input_path=args.input,
                temp_dir=args.temp_dir,
                output_path=args.output,
                workers=args.workers,
                languages=args.languages,
                retry_errors=args.retry_errors,
                delay_range=delay_range,
                cookies_from_browser=None,
                max_429_retries=args.max_429_retries,
                backoff_base_seconds=args.backoff_base_seconds,
                backoff_max_seconds=args.backoff_max_seconds,
                cookies_file=args.cookies_file,
            )
        )

    # Always merge after fetch (unless we bailed early)
    merge_to_parquet(args.temp_dir, args.output)

    if not args.keep_temp:
        print(f"Deleting temp folder: {args.temp_dir}")
        shutil.rmtree(args.temp_dir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()

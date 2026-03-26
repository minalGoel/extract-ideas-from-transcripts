#!/usr/bin/env python3
"""
Read a CSV of YouTube links (video | playlist | channel) and write a flat
metadata CSV — one row per resolved video.  No transcripts are fetched.

yt-dlp's "flat" extraction is used so only the channel/playlist index page
is fetched (not each individual video page) — much faster for bulk runs.

Output columns
--------------
video_id, title, channel_name, channel_url, video_url, upload_date,
duration_seconds, view_count, description_snippet, source_type, source_url

Usage
-----
python youtube_csv_to_metadata.py stock_market_videos.csv
python youtube_csv_to_metadata.py stock_market_videos.csv -o all_videos.csv --max-per-source 200
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterator

import yt_dlp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_url(url: str) -> str:
    """Add https:// if the URL is missing a scheme."""
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _channel_videos_url(url: str) -> str:
    """
    Force a channel URL to its /videos tab so yt-dlp fetches all uploads
    rather than the Home tab (which returns only a handful of featured videos).

    Handles all channel URL patterns:
      /@handle            →  /@handle/videos
      /@handle/           →  /@handle/videos
      /c/name             →  /c/name/videos
      /channel/UCxxx      →  /channel/UCxxx/videos
      /user/name          →  /user/name/videos
    Playlist and single-video URLs are returned unchanged.
    """
    # Strip trailing slash, then check the path
    url = url.rstrip("/")
    channel_patterns = (
        r"youtube\.com/@[^/?#]+$",
        r"youtube\.com/c/[^/?#]+$",
        r"youtube\.com/channel/[^/?#]+$",
        r"youtube\.com/user/[^/?#]+$",
    )
    if any(re.search(p, url) for p in channel_patterns):
        return url + "/videos"
    return url


def _flat_extract(url: str, max_entries: int | None = None) -> list[dict]:
    """
    Use yt-dlp flat extraction to list entries from a channel, playlist,
    or single video URL.  Returns a list of raw yt-dlp info dicts.
    """
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "socket_timeout": 30,
        "retries": 3,
        "ignoreerrors": True,
    }
    if max_entries:
        opts["playlistend"] = max_entries

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        print(f"  [ERROR] yt-dlp failed for {url}: {exc}", file=sys.stderr)
        return []

    if not info:
        return []

    # Single video (not a playlist/channel)
    if info.get("_type") not in ("playlist", "multi_video"):
        return [info]

    entries = [e for e in (info.get("entries") or []) if e]
    return entries


def _parse_entry(entry: dict, source_type: str, source_url: str) -> dict | None:
    """Convert a raw yt-dlp entry into our output row dict."""
    video_id = entry.get("id") or entry.get("webpage_url_basename") or ""
    if not video_id:
        return None

    # Reconstruct canonical URL
    if re.fullmatch(r"[A-Za-z0-9_\-]{11}", video_id):
        video_url = f"https://www.youtube.com/watch?v={video_id}"
    else:
        video_url = entry.get("url") or entry.get("webpage_url") or ""
        # If it looks like a bare ID embedded in a relative URL, fix it
        if video_url.startswith("/watch"):
            video_url = "https://www.youtube.com" + video_url

    title = (entry.get("title") or "").strip() or video_id

    channel_name = (
        entry.get("channel")
        or entry.get("uploader")
        or entry.get("channel_id")
        or ""
    ).strip()

    channel_url = (
        entry.get("channel_url")
        or entry.get("uploader_url")
        or ""
    ).strip()

    upload_date = entry.get("upload_date") or ""  # YYYYMMDD or empty
    if upload_date and len(upload_date) == 8:
        # Reformat to YYYY-MM-DD for readability
        upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"

    duration = entry.get("duration")  # seconds (int) or None

    view_count = entry.get("view_count")

    # Description: flat extraction often doesn't include it; keep snippet if present
    desc = (entry.get("description") or "").replace("\n", " ").strip()
    desc_snippet = desc[:200] if desc else ""

    return {
        "video_id":          video_id,
        "title":             title,
        "channel_name":      channel_name,
        "channel_url":       channel_url,
        "video_url":         video_url,
        "upload_date":       upload_date,
        "duration_seconds":  duration if duration is not None else "",
        "view_count":        view_count if view_count is not None else "",
        "description_snippet": desc_snippet,
        "source_type":       source_type,
        "source_url":        source_url,
    }


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def iter_csv_rows(csv_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (row_type, url) pairs from the input CSV."""
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
        type_col = cols.get("type")
        link_col = cols.get("link")
        if not type_col or not link_col:
            raise ValueError("CSV must have 'type' and 'link' columns")

        for row in reader:
            row_type = (row.get(type_col) or "").strip().lower()
            link = (row.get(link_col) or "").strip()
            if row_type and link:
                yield row_type, _normalise_url(link)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect_metadata(
    csv_path: Path,
    max_per_source: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[str] = set()
    total_sources = 0

    for row_type, url in iter_csv_rows(csv_path):
        total_sources += 1
        fetch_url = _channel_videos_url(url) if row_type == "channel" else url
        if fetch_url != url:
            print(f"[{row_type.upper():8s}] {url}")
            print(f"           → fetching: {fetch_url}")
        else:
            print(f"[{row_type.upper():8s}] {url}")

        entries = _flat_extract(fetch_url, max_entries=max_per_source)
        source_count = 0

        for entry in entries:
            parsed = _parse_entry(entry, source_type=row_type, source_url=url)
            if not parsed:
                continue
            vid = parsed["video_id"]
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            rows.append(parsed)
            source_count += 1

        print(f"           → {source_count} videos resolved  (running total: {len(rows)})")

    print(f"\nProcessed {total_sources} sources → {len(rows)} unique videos")
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        print("No rows to write.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written: {output_path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Expand a YouTube CSV (video/playlist/channel) into a flat video metadata CSV."
    )
    p.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "stock_market_videos.csv",
        help="Input CSV with 'type' and 'link' columns (default: stock_market_videos.csv)",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(__file__).parent / "youtube_video_metadata.csv",
        help="Output CSV path (default: youtube_video_metadata.csv)",
    )
    p.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        metavar="N",
        help="Max videos to resolve per channel/playlist (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv_file.exists():
        print(f"Input file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    rows = collect_metadata(args.csv_file, max_per_source=args.max_per_source)
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()

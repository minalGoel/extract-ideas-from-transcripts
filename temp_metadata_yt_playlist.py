"""
Fetch video metadata from a single hardcoded YouTube playlist and write
to temp_metadata_yt_playlist.csv (same schema as youtube_video_metadata.csv).

Run:  python temp_metadata_yt_playlist.py
"""

import csv
import json
import subprocess

PLAYLIST_URL = "https://youtube.com/playlist?list=PLFj8mcG4JCG9a7wnLdUG-EcjcTJ7lgINM&si=V54conugAIPkww94"
OUTPUT_CSV   = "temp_metadata_yt_playlist.csv"

FIELDNAMES = [
    "video_id", "title", "channel_name", "channel_url", "video_url",
    "upload_date", "duration_seconds", "view_count",
    "description_snippet", "source_type", "source_url",
]


def fetch_playlist_entries(playlist_url: str) -> list[dict]:
    import shutil, sys
    ytdlp = shutil.which("yt-dlp") or str(
        __import__("pathlib").Path(sys.executable).parent / "yt-dlp"
    )
    cmd = [
        ytdlp,
        "--flat-playlist",
        "--print-json",
        "--no-warnings",
        "--quiet",
        playlist_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    entries = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def main():
    print(f"Fetching playlist: {PLAYLIST_URL}")
    entries = fetch_playlist_entries(PLAYLIST_URL)
    print(f"Found {len(entries)} videos")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for e in entries:
            video_id = e.get("id", "")
            channel_id = e.get("channel_id") or e.get("uploader_id", "")
            channel_url = (
                f"https://www.youtube.com/channel/{channel_id}" if channel_id else ""
            )
            upload_date = e.get("upload_date", "") or ""
            # yt-dlp returns YYYYMMDD — keep as-is for consistency with existing CSV
            writer.writerow({
                "video_id":           video_id,
                "title":              e.get("title", ""),
                "channel_name":       e.get("channel") or e.get("uploader", ""),
                "channel_url":        channel_url,
                "video_url":          f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
                "upload_date":        upload_date,
                "duration_seconds":   e.get("duration", ""),
                "view_count":         e.get("view_count", ""),
                "description_snippet": (e.get("description") or "")[:200],
                "source_type":        "playlist",
                "source_url":         PLAYLIST_URL,
            })

    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

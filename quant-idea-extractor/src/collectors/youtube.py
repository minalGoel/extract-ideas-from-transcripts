"""
YouTube collector — uses yt-dlp to pull auto-generated or manual subtitles.

yt-dlp must be installed (it's in requirements.txt).
Channel URLs are read from seeds/youtube_channels.yaml.
"""

import asyncio
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger

from ..normalizer import NormalizedDocument, normalize
from .base import BaseCollector

_SEEDS = Path("seeds/youtube_channels.yaml")


class YouTubeCollector(BaseCollector):
    def __init__(self, config: dict) -> None:
        self.config = config
        yc = config.get("collection", {}).get("youtube", {})
        self.max_videos   = yc.get("max_videos_per_channel", 100)
        self.min_duration = yc.get("min_duration_seconds", 180)

    def _load_channels(self) -> list[dict]:
        if _SEEDS.exists():
            with open(_SEEDS) as f:
                return yaml.safe_load(f).get("channels", [])
        return []

    # ------------------------------------------------------------------
    # yt-dlp helpers (run in a thread pool — they're synchronous/blocking)
    # ------------------------------------------------------------------

    def _list_channel_videos(self, channel_url: str) -> list[dict]:
        """Return metadata for up to max_videos videos (no download)."""
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--playlist-end", str(self.max_videos),
            "--print-json",
            "--no-warnings",
            "--quiet",
            channel_url,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            entries = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return entries
        except Exception as exc:
            logger.error(f"YouTube: yt-dlp list failed for {channel_url}: {exc}")
            return []

    def _get_transcript(self, video_id: str) -> str:
        """Download subtitles for one video and return plain text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "yt-dlp",
                "--write-auto-subs",
                "--write-subs",
                "--skip-download",
                "--sub-format", "json3",
                "--sub-lang", "en",
                "--no-warnings",
                "--quiet",
                "-o", os.path.join(tmpdir, "%(id)s"),
                f"https://www.youtube.com/watch?v={video_id}",
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=60)
            except Exception as exc:
                logger.debug(f"YouTube: subtitle download failed for {video_id}: {exc}")
                return ""

            for fname in os.listdir(tmpdir):
                if fname.endswith(".json3"):
                    fpath = os.path.join(tmpdir, fname)
                    try:
                        with open(fpath) as f:
                            data = json.load(f)
                        return self._parse_json3(data)
                    except Exception as exc:
                        logger.debug(f"YouTube: json3 parse failed {fname}: {exc}")
            return ""

    @staticmethod
    def _parse_json3(data: dict) -> str:
        parts = []
        for event in data.get("events", []):
            for seg in event.get("segs", []):
                t = seg.get("utf8", "").strip()
                if t and t != "\n":
                    parts.append(t)
        return " ".join(parts)

    # ------------------------------------------------------------------

    async def collect(self, limit: int | None = None) -> list[NormalizedDocument]:
        channels = self._load_channels()
        docs: list[NormalizedDocument] = []
        loop = asyncio.get_event_loop()
        max_vids = min(limit or self.max_videos, self.max_videos)

        for ch in channels:
            url  = ch.get("url", "")
            name = ch.get("name", url)
            logger.info(f"YouTube: collecting {name}")

            try:
                entries = await loop.run_in_executor(
                    None, self._list_channel_videos, url
                )

                count = 0
                for entry in entries[:max_vids]:
                    duration = entry.get("duration") or 0
                    if duration < self.min_duration:
                        continue

                    video_id = entry.get("id", "")
                    if not video_id:
                        continue

                    transcript = await loop.run_in_executor(
                        None, self._get_transcript, video_id
                    )
                    if not transcript:
                        logger.debug(f"YouTube: no transcript for {video_id}")
                        continue

                    raw_date = entry.get("upload_date", "")
                    try:
                        date = datetime.strptime(raw_date, "%Y%m%d").replace(
                            tzinfo=timezone.utc
                        ).isoformat()
                    except (ValueError, TypeError):
                        date = ""

                    doc = normalize(
                        source_id=video_id,
                        source_platform="youtube",
                        title=entry.get("title", ""),
                        author=entry.get("channel") or name,
                        date=date,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        raw_text=transcript,
                        metadata={
                            "channel_name": name,
                            "duration": duration,
                            "view_count": entry.get("view_count"),
                        },
                    )
                    if doc:
                        docs.append(doc)
                        count += 1

                logger.info(f"YouTube: {name} → {count} videos")

            except Exception as exc:
                logger.error(f"YouTube: channel {name} failed: {exc}")

        return docs

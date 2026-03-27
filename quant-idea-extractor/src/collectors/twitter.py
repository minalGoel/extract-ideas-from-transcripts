"""
Twitter/X collector — best-effort, entirely optional.

Mode A: Nitter scraping (public instances, frequently break).
Mode B: Manual import from a JSON or CSV file (X archive export or
        browser-extension export).

The pipeline will NOT fail if this collector is skipped or errors out.
Seeds are in seeds/twitter_handles.yaml (for future use when a reliable
free method exists).
"""

import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml
from bs4 import BeautifulSoup
from loguru import logger

from ..normalizer import NormalizedDocument, normalize, make_source_id
from .base import BaseCollector

_SEEDS = Path("seeds/twitter_handles.yaml")

_NITTER_INSTANCES = [
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.net",
    "https://nitter.1d4.us",
]

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


class TwitterCollector(BaseCollector):
    """
    Parameters
    ----------
    mode : "nitter" | "import"
    import_file : path to JSON or CSV when mode="import"
    """

    def __init__(
        self,
        config: dict,
        mode: str = "nitter",
        import_file: str | None = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self.import_file = import_file

    # ------------------------------------------------------------------
    # Nitter scraping
    # ------------------------------------------------------------------

    async def _nitter_handle(
        self,
        client: httpx.AsyncClient,
        handle: str,
    ) -> list[NormalizedDocument]:
        docs: list[NormalizedDocument] = []

        for instance in _NITTER_INSTANCES:
            url = f"{instance}/{handle}"
            try:
                resp = await client.get(
                    url,
                    timeout=15,
                    follow_redirects=True,
                    headers={"User-Agent": _UA},
                )
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                for div in soup.find_all("div", class_="tweet-content"):
                    text = div.get_text(strip=True)
                    if not text:
                        continue

                    # Try to find timestamp
                    ts_tag = div.find_parent("div", class_="tweet-body")
                    date_str = ""
                    if ts_tag:
                        t = ts_tag.find("span", class_="tweet-date")
                        if t and t.get("title"):
                            date_str = t["title"]

                    doc = normalize(
                        source_id=make_source_id(text[:120]),
                        source_platform="twitter",
                        title=f"Tweet by @{handle}",
                        author=f"@{handle}",
                        date=date_str,
                        url=f"https://twitter.com/{handle}",
                        raw_text=text,
                        metadata={"handle": handle, "nitter_instance": instance},
                    )
                    if doc:
                        docs.append(doc)

                if docs:
                    break   # got data from this instance

            except Exception as exc:
                logger.debug(f"Twitter/Nitter: {instance} failed for @{handle}: {exc}")

        return docs

    # ------------------------------------------------------------------
    # Manual import
    # ------------------------------------------------------------------

    def _import_json(self, path: Path) -> list[NormalizedDocument]:
        docs: list[NormalizedDocument] = []
        try:
            with open(path) as f:
                tweets = json.load(f)
            if isinstance(tweets, dict):
                tweets = tweets.get("data", tweets.get("tweets", []))
            for tw in tweets:
                text   = tw.get("full_text") or tw.get("text", "")
                author = (tw.get("user") or {}).get("screen_name", "unknown")
                date   = tw.get("created_at", "")
                tid    = str(tw.get("id") or make_source_id(text[:100]))

                doc = normalize(
                    source_id=tid,
                    source_platform="twitter",
                    title=f"Tweet by @{author}",
                    author=f"@{author}",
                    date=date,
                    url=f"https://twitter.com/{author}/status/{tid}",
                    raw_text=text,
                    metadata={"source": "manual_import_json"},
                )
                if doc:
                    docs.append(doc)
        except Exception as exc:
            logger.error(f"Twitter: JSON import failed: {exc}")
        return docs

    def _import_csv(self, path: Path) -> list[NormalizedDocument]:
        docs: list[NormalizedDocument] = []
        try:
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    text   = row.get("full_text") or row.get("text", "")
                    author = row.get("username") or row.get("author") or row.get("screen_name", "unknown")
                    date   = row.get("created_at", "")
                    tid    = row.get("id") or row.get("tweet_id") or make_source_id(text[:100])

                    doc = normalize(
                        source_id=str(tid),
                        source_platform="twitter",
                        title=f"Tweet by @{author}",
                        author=f"@{author}",
                        date=date,
                        url=f"https://twitter.com/{author}",
                        raw_text=text,
                        metadata={"source": "manual_import_csv"},
                    )
                    if doc:
                        docs.append(doc)
        except Exception as exc:
            logger.error(f"Twitter: CSV import failed: {exc}")
        return docs

    # ------------------------------------------------------------------

    async def collect(self, limit: int | None = None) -> list[NormalizedDocument]:
        docs: list[NormalizedDocument] = []

        if self.mode == "import" and self.import_file:
            path = Path(self.import_file)
            if not path.exists():
                logger.error(f"Twitter: import file not found: {self.import_file}")
                return []
            if path.suffix.lower() == ".json":
                docs = self._import_json(path)
            elif path.suffix.lower() == ".csv":
                docs = self._import_csv(path)
            else:
                logger.error("Twitter: import file must be .json or .csv")
                return []

        elif self.mode == "nitter":
            if not _SEEDS.exists():
                logger.info("Twitter: no handles file, skipping.")
                return []
            with open(_SEEDS) as f:
                handles = yaml.safe_load(f).get("handles", [])

            async with httpx.AsyncClient() as client:
                for handle in handles[:20]:
                    logger.info(f"Twitter/Nitter: trying @{handle}")
                    new_docs = await self._nitter_handle(client, handle)
                    docs.extend(new_docs)
                    await asyncio.sleep(2)

        else:
            logger.warning("Twitter: no valid mode configured, skipping.")
            return []

        if limit:
            docs = docs[:limit]

        logger.info(f"Twitter: {len(docs)} items collected")
        return docs

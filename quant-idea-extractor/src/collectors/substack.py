"""
Substack / blog collector — parses RSS feeds and fetches full post text.

Uses feedparser for RSS, httpx for HTTP, and trafilatura for boilerplate
removal (falls back to BeautifulSoup if trafilatura is unavailable).
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import httpx
import yaml
from loguru import logger

from ..normalizer import NormalizedDocument, normalize, make_source_id
from .base import BaseCollector

_SEEDS = Path("seeds/substack_feeds.yaml")
_UA = "quant-idea-extractor/1.0 (RSS reader)"


class SubstackCollector(BaseCollector):
    def __init__(self, config: dict) -> None:
        self.config = config
        sc = config.get("collection", {}).get("substack", {})
        self.max_posts = sc.get("max_posts_per_feed", 50)
        self._timeout = httpx.Timeout(30.0)

    # ------------------------------------------------------------------

    def _load_feeds(self) -> list[dict[str, str]]:
        if _SEEDS.exists():
            with open(_SEEDS) as f:
                return yaml.safe_load(f).get("feeds", [])
        return []

    async def _full_text(
        self, client: httpx.AsyncClient, url: str
    ) -> str:
        """Fetch and extract main article text from a URL."""
        try:
            resp = await client.get(
                url, timeout=self._timeout, follow_redirects=True
            )
            resp.raise_for_status()

            try:
                import trafilatura
                text = trafilatura.extract(
                    resp.text,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                )
                return text or ""
            except ImportError:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    tag.decompose()
                return soup.get_text(separator="\n")
        except Exception as exc:
            logger.debug(f"full_text fetch failed for {url}: {exc}")
            return ""

    async def _collect_one_feed(
        self,
        client: httpx.AsyncClient,
        feed_info: dict[str, str],
    ) -> list[NormalizedDocument]:
        feed_url  = feed_info.get("url", "")
        blog_name = feed_info.get("name", feed_url)
        docs: list[NormalizedDocument] = []

        try:
            resp = await client.get(
                feed_url, timeout=self._timeout, follow_redirects=True
            )
            feed: Any = feedparser.parse(resp.text)
        except Exception as exc:
            logger.warning(f"Substack: feed fetch failed {feed_url}: {exc}")
            return docs

        for entry in feed.entries[: self.max_posts]:
            try:
                article_url = entry.get("link", "")

                # RSS content first, then fetch full text if short
                raw = ""
                if hasattr(entry, "content"):
                    raw = entry.content[0].value
                elif hasattr(entry, "summary"):
                    raw = entry.summary

                if len(raw) < 600 and article_url:
                    full = await self._full_text(client, article_url)
                    if full:
                        raw = full
                    await asyncio.sleep(1.0)   # polite delay

                pub_date = ""
                if getattr(entry, "published_parsed", None):
                    pub_date = datetime(
                        *entry.published_parsed[:6], tzinfo=timezone.utc
                    ).isoformat()

                sid = make_source_id(article_url or entry.get("id", raw[:100]))

                doc = normalize(
                    source_id=sid,
                    source_platform="substack",
                    title=entry.get("title", ""),
                    author=entry.get("author", blog_name),
                    date=pub_date,
                    url=article_url,
                    raw_text=raw,
                    metadata={"blog_name": blog_name, "feed_url": feed_url},
                )
                if doc:
                    docs.append(doc)

            except Exception as exc:
                logger.debug(f"Substack: entry error from {blog_name}: {exc}")

        logger.info(f"Substack: {blog_name} → {len(docs)} posts")
        return docs

    # ------------------------------------------------------------------

    async def collect(self, limit: int | None = None) -> list[NormalizedDocument]:
        feeds = self._load_feeds()
        docs: list[NormalizedDocument] = []

        async with httpx.AsyncClient(headers={"User-Agent": _UA}) as client:
            tasks = [self._collect_one_feed(client, f) for f in feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, list):
                docs.extend(r)
            elif isinstance(r, Exception):
                logger.warning(f"Substack: feed task raised {r}")

        if limit:
            docs = docs[:limit]

        logger.info(f"Substack: total {len(docs)} posts")
        return docs

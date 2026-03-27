"""
SSRN collector — web scraping with httpx + BeautifulSoup.
Respects robots.txt spirit: delays between requests, graceful failure.

No authentication required.
"""

import asyncio
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from ..normalizer import NormalizedDocument, normalize, make_source_id
from .base import BaseCollector

_DEFAULT_KEYWORDS = [
    "alternative data",
    "factor model",
    "statistical arbitrage",
    "market microstructure",
    "quantitative trading",
    "machine learning finance",
    "alpha signal",
    "systematic trading",
    "momentum strategy",
    "high frequency trading",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class SSRNCollector(BaseCollector):
    def __init__(self, config: dict) -> None:
        self.config = config
        sc = config.get("collection", {}).get("ssrn", {})
        self.delay      = sc.get("request_delay_seconds", 3)
        self.max_results = sc.get("max_results_per_keyword", 100)

    # ------------------------------------------------------------------

    def _parse_results(self, html: str, keyword: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        papers = []

        # SSRN search results container varies — try multiple selectors
        containers = (
            soup.find_all("div", class_=re.compile(r"paper-abstract"))
            or soup.find_all("div", class_=re.compile(r"articles"))
            or soup.find_all("article")
        )

        for el in containers:
            try:
                title_tag = (
                    el.find("a", class_=re.compile(r"title"))
                    or el.find("h3", class_=re.compile(r"title"))
                    or el.find("a", href=re.compile(r"abstract"))
                )
                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                url   = title_tag.get("href", "")
                if url and not url.startswith("http"):
                    url = f"https://papers.ssrn.com{url}"

                abstract_tag = el.find(
                    "div", class_=re.compile(r"abstract|summary")
                )
                abstract = abstract_tag.get_text(strip=True) if abstract_tag else ""

                author_tag = el.find(
                    attrs={"class": re.compile(r"author|byline")}
                )
                author = author_tag.get_text(strip=True) if author_tag else ""

                date_tag = el.find(attrs={"class": re.compile(r"date|posted")})
                date = date_tag.get_text(strip=True) if date_tag else ""

                # Extract SSRN ID from URL
                m = re.search(r"(?:abstract_id|abstract)=?(\d{5,})", url)
                ssrn_id = m.group(1) if m else make_source_id(url)

                if title and (abstract or url):
                    papers.append({
                        "ssrn_id": ssrn_id,
                        "title":   title,
                        "author":  author,
                        "abstract": abstract,
                        "date":    date,
                        "url":     url,
                        "keyword": keyword,
                    })
            except Exception as exc:
                logger.debug(f"SSRN: result parse error: {exc}")

        return papers

    async def _search_keyword(
        self,
        client: httpx.AsyncClient,
        keyword: str,
    ) -> list[dict]:
        papers = []
        page = 0

        while len(papers) < self.max_results:
            url = (
                "https://papers.ssrn.com/sol3/search.taf"
                f"?txtKeywords={keyword.replace(' ', '+')}"
                "&sort=Date_Posted_first"
                f"&start={page * 20}"
            )
            try:
                await asyncio.sleep(self.delay)
                resp = await client.get(url, headers=_HEADERS, timeout=30)

                if resp.status_code == 403:
                    logger.warning(f"SSRN: blocked for '{keyword}' — stopping this keyword")
                    break

                resp.raise_for_status()
                batch = self._parse_results(resp.text, keyword)
                if not batch:
                    break

                papers.extend(batch)
                page += 1

                if len(batch) < 20:
                    break   # last page

            except httpx.HTTPStatusError as exc:
                logger.warning(f"SSRN: HTTP {exc.response.status_code} for '{keyword}'")
                break
            except httpx.RequestError as exc:
                logger.warning(f"SSRN: request error for '{keyword}': {exc}")
                break
            except Exception as exc:
                logger.error(f"SSRN: unexpected error for '{keyword}': {exc}")
                break

        return papers[: self.max_results]

    # ------------------------------------------------------------------

    async def collect(self, limit: int | None = None) -> list[NormalizedDocument]:
        docs: list[NormalizedDocument] = []
        seen_ids: set[str] = set()
        max_per_kw = min(limit or self.max_results, self.max_results)

        async with httpx.AsyncClient(follow_redirects=True) as client:
            for kw in _DEFAULT_KEYWORDS:
                logger.info(f"SSRN: searching '{kw}'")
                try:
                    papers = await self._search_keyword(client, kw)
                    new_count = 0

                    for p in papers:
                        sid = p["ssrn_id"]
                        if sid in seen_ids:
                            continue
                        seen_ids.add(sid)

                        text = (
                            f"# {p['title']}\n\n"
                            f"**Authors:** {p['author']}\n\n"
                            f"**Abstract:**\n{p['abstract']}"
                        )
                        doc = normalize(
                            source_id=sid,
                            source_platform="ssrn",
                            title=p["title"],
                            author=p["author"],
                            date=p["date"],
                            url=p["url"],
                            raw_text=text,
                            metadata={"search_keyword": kw},
                        )
                        if doc:
                            docs.append(doc)
                            new_count += 1

                    logger.info(f"SSRN: '{kw}' → {new_count} new papers")

                except Exception as exc:
                    logger.error(f"SSRN: keyword '{kw}' failed: {exc}")

        logger.info(f"SSRN: total {len(docs)} unique papers")
        return docs

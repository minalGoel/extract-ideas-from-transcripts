"""
arXiv collector — uses the `arxiv` Python package (no auth required).
Only the abstract is used for extraction (no PDF downloads).
"""

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

import arxiv
import yaml
from loguru import logger

from ..normalizer import NormalizedDocument, normalize
from .base import BaseCollector

_SEEDS = Path("seeds/arxiv_categories.yaml")
_DEFAULT_CATS = [
    "q-fin.TR",   # Trading and Market Microstructure
    "q-fin.ST",   # Statistical Finance
    "q-fin.PM",   # Portfolio Management
    "q-fin.CP",   # Computational Finance
    "q-fin.RM",   # Risk Management
]


class ArxivCollector(BaseCollector):
    def __init__(self, config: dict) -> None:
        self.config = config
        ac = config.get("collection", {}).get("arxiv", {})
        self.max_papers = ac.get("max_papers_per_category", 500)
        self.years_back = ac.get("years_back", 3)
        self._client = arxiv.Client(page_size=100, delay_seconds=3.0)

    def _load_categories(self) -> list[str]:
        if _SEEDS.exists():
            with open(_SEEDS) as f:
                return yaml.safe_load(f).get("categories", _DEFAULT_CATS)
        return _DEFAULT_CATS

    async def collect(self, limit: int | None = None) -> list[NormalizedDocument]:
        categories = self._load_categories()
        docs: list[NormalizedDocument] = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=365 * self.years_back)
        max_per_cat = min(limit or self.max_papers, self.max_papers)

        for cat in categories:
            logger.info(f"arXiv: collecting {cat}")
            try:
                search = arxiv.Search(
                    query=f"cat:{cat}",
                    max_results=max_per_cat,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                def _fetch():
                    results = []
                    for r in self._client.results(search):
                        pub = r.published
                        if pub.tzinfo is None:
                            pub = pub.replace(tzinfo=timezone.utc)
                        if pub < cutoff:
                            break
                        results.append(r)
                    return results

                results = await asyncio.get_event_loop().run_in_executor(None, _fetch)
                count = 0

                for r in results:
                    authors = ", ".join(a.name for a in r.authors[:5])
                    if len(r.authors) > 5:
                        authors += " et al."

                    text = (
                        f"# {r.title}\n\n"
                        f"**Authors:** {authors}\n\n"
                        f"**Abstract:**\n{r.summary}"
                    )

                    doc = normalize(
                        source_id=r.get_short_id(),
                        source_platform="arxiv",
                        title=r.title,
                        author=authors,
                        date=r.published.isoformat(),
                        url=r.entry_id,
                        raw_text=text,
                        metadata={
                            "categories": r.categories,
                            "pdf_url": r.pdf_url,
                            "arxiv_id": r.get_short_id(),
                        },
                    )
                    if doc:
                        docs.append(doc)
                        count += 1

                logger.info(f"arXiv: {cat} → {count} papers")

            except Exception as exc:
                logger.error(f"arXiv: {cat} failed: {exc}")

        return docs

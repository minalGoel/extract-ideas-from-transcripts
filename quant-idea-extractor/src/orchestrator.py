"""
Main pipeline orchestrator.

Coordinates: collect → extract → dedup → stats
Each stage can be run standalone (see scripts/) or called from here.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import yaml
from loguru import logger

from .db import get_stats, get_unextracted_documents, init_db, insert_document
from .dedup import run_dedup
from .extractor import IdeaExtractor
from .utils import setup_logging


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Stage: collect
# ---------------------------------------------------------------------------

async def run_collect(
    source: str = "all",
    limit: int | None = None,
    config: dict | None = None,
    dry_run: bool = False,
) -> None:
    if config is None:
        config = load_config()

    db_path = config.get("database", {}).get("path", "data/quant_ideas.db")
    await init_db(db_path)

    collectors = _build_collectors(source, config)
    if not collectors:
        logger.error(f"Unknown or unsupported source: {source!r}")
        return

    async with aiosqlite.connect(db_path) as db:
        for name, collector in collectors.items():
            logger.info(f"Collecting from {name} …")
            try:
                if dry_run:
                    await collector.dry_run(limit=5)
                else:
                    docs = await collector.collect(limit=limit)
                    new_count = 0
                    for doc in docs:
                        row_id = await insert_document(db, doc)
                        if row_id:
                            new_count += 1
                    logger.info(
                        f"{name}: {len(docs)} fetched, {new_count} new in DB"
                    )
            except Exception as exc:
                logger.error(f"Collection failed for {name}: {exc}")


def _build_collectors(source: str, config: dict) -> dict:
    collectors: dict = {}

    def _try_add(key: str, cls, *args, **kwargs):
        if source in (key, "all"):
            try:
                collectors[key] = cls(*args, **kwargs)
            except Exception as exc:
                logger.warning(f"{key} collector init failed (skipping): {exc}")

    from .collectors.arxiv_collector import ArxivCollector
    from .collectors.reddit import RedditCollector
    from .collectors.ssrn import SSRNCollector
    from .collectors.substack import SubstackCollector
    from .collectors.twitter import TwitterCollector
    from .collectors.youtube import YouTubeCollector

    _try_add("reddit",   RedditCollector,  config)
    _try_add("arxiv",    ArxivCollector,   config)
    _try_add("substack", SubstackCollector, config)
    _try_add("youtube",  YouTubeCollector,  config)
    _try_add("ssrn",     SSRNCollector,    config)
    _try_add("twitter",  TwitterCollector, config)

    return collectors


# ---------------------------------------------------------------------------
# Stage: extract
# ---------------------------------------------------------------------------

async def run_extract(
    batch_size: int = 20,
    max_cost: float = 5.0,
    config: dict | None = None,
) -> None:
    if config is None:
        config = load_config()

    config.setdefault("extraction", {})["max_cost_per_run"] = max_cost
    db_path = config.get("database", {}).get("path", "data/quant_ideas.db")

    extractor = IdeaExtractor(config)
    total_processed = 0

    async with aiosqlite.connect(db_path) as db:
        while True:
            docs = await get_unextracted_documents(db, limit=batch_size)
            if not docs:
                logger.info("No more unextracted documents.")
                break

            if extractor.total_cost >= max_cost:
                logger.info(f"Cost cap ${max_cost:.2f} reached. Stopping.")
                break

            processed, cost = await extractor.extract_batch(db, docs)
            total_processed += processed
            logger.info(
                f"Batch done: {processed}/{len(docs)} docs | "
                f"running cost ${cost:.4f}"
            )

    logger.info(
        f"Extraction complete. "
        f"{total_processed} docs processed | "
        f"{extractor.total_ideas} ideas extracted | "
        f"${extractor.total_cost:.4f} spent"
    )


# ---------------------------------------------------------------------------
# Stage: dedup
# ---------------------------------------------------------------------------

async def run_dedup_pipeline(config: dict | None = None) -> None:
    if config is None:
        config = load_config()
    db_path = config.get("database", {}).get("path", "data/quant_ideas.db")

    async with aiosqlite.connect(db_path) as db:
        n_clusters = await run_dedup(db, config)

    logger.info(f"Dedup complete. {n_clusters} clusters created.")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

async def print_stats(config: dict | None = None) -> None:
    if config is None:
        config = load_config()
    db_path = config.get("database", {}).get("path", "data/quant_ideas.db")

    async with aiosqlite.connect(db_path) as db:
        stats = await get_stats(db)

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print("\n[bold cyan]Quant Idea Extractor — Pipeline Stats[/bold cyan]\n")

        doc_table = Table(title="Documents by Source", show_lines=True)
        doc_table.add_column("Source", style="cyan")
        doc_table.add_column("Count", justify="right")
        for src, cnt in sorted(stats.get("docs_by_source", {}).items()):
            doc_table.add_row(src, str(cnt))
        doc_table.add_row(
            "[bold]Total[/bold]", str(stats.get("total_docs", 0))
        )
        doc_table.add_row(
            "[bold]Extracted[/bold]", str(stats.get("extracted_docs", 0))
        )
        console.print(doc_table)

        idea_table = Table(title="Ideas by Type", show_lines=True)
        idea_table.add_column("Type", style="green")
        idea_table.add_column("Count", justify="right")
        for t, cnt in sorted(stats.get("ideas_by_type", {}).items()):
            idea_table.add_row(t or "(unset)", str(cnt))
        idea_table.add_row(
            "[bold]Total[/bold]", str(stats.get("total_ideas", 0))
        )
        console.print(idea_table)

        console.print(
            f"\n[bold yellow]Total API cost:[/bold yellow] "
            f"${stats.get('total_cost_usd', 0.0):.6f}\n"
        )

    except ImportError:
        # Fallback without Rich
        print("\n=== Pipeline Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

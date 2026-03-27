#!/usr/bin/env python3
"""
CLI: cluster and deduplicate extracted ideas.

Examples
--------
python scripts/run_dedup.py
python scripts/run_dedup.py --threshold 0.7
"""

import asyncio
import sys
from pathlib import Path

import aiosqlite
import typer
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from src.orchestrator import load_config
from src.dedup import run_dedup
from src.utils import setup_logging

app = typer.Typer(add_completion=False)


@app.command()
def main(
    threshold: float = typer.Option(
        None,
        "--threshold", "-t",
        help="Cosine similarity threshold (0–1). Overrides config.yaml value.",
    ),
    config_path: str = typer.Option("config.yaml", "--config", "-c"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    setup_logging(log_level)
    config = load_config(config_path)

    if threshold is not None:
        config.setdefault("dedup", {})["similarity_threshold"] = threshold

    db_path = config.get("database", {}).get("path", "data/quant_ideas.db")

    async def _run():
        async with aiosqlite.connect(db_path) as db:
            n = await run_dedup(db, config)
        print(f"Dedup complete: {n} clusters assigned.")

    asyncio.run(_run())


if __name__ == "__main__":
    app()

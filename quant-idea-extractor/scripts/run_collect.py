#!/usr/bin/env python3
"""
CLI: collect documents from one or all sources.

Examples
--------
python scripts/run_collect.py --source all --limit 100
python scripts/run_collect.py --source reddit --limit 50
python scripts/run_collect.py --source arxiv
python scripts/run_collect.py --stats
python scripts/run_collect.py --source substack --dry-run
"""

import asyncio
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from src.orchestrator import load_config, print_stats, run_collect
from src.utils import setup_logging

app = typer.Typer(add_completion=False)


@app.command()
def main(
    source: str = typer.Option(
        "all",
        "--source", "-s",
        help="Source: youtube | reddit | arxiv | ssrn | substack | twitter | all",
    ),
    limit: int = typer.Option(
        None,
        "--limit", "-l",
        help="Max items to collect per source (default: from config.yaml)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview what would be collected without storing to DB",
    ),
    stats: bool = typer.Option(
        False,
        "--stats",
        help="Show pipeline statistics and exit",
    ),
    config_path: str = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to config.yaml",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level: DEBUG | INFO | WARNING",
    ),
) -> None:
    setup_logging(log_level)
    config = load_config(config_path)

    if stats:
        asyncio.run(print_stats(config))
        return

    asyncio.run(run_collect(source=source, limit=limit, config=config, dry_run=dry_run))


if __name__ == "__main__":
    app()

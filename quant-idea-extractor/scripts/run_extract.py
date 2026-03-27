#!/usr/bin/env python3
"""
CLI: run Claude API extraction on unprocessed documents.

Examples
--------
python scripts/run_extract.py --batch-size 20 --max-cost 5.0
python scripts/run_extract.py --batch-size 50 --max-cost 10.0
"""

import asyncio
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from src.orchestrator import load_config, run_extract
from src.utils import setup_logging

app = typer.Typer(add_completion=False)


@app.command()
def main(
    batch_size: int = typer.Option(
        20,
        "--batch-size", "-b",
        help="Number of documents per concurrent batch",
    ),
    max_cost: float = typer.Option(
        5.0,
        "--max-cost",
        help="Hard cost cap in USD for this run",
    ),
    config_path: str = typer.Option(
        "config.yaml",
        "--config", "-c",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    setup_logging(log_level)
    config = load_config(config_path)
    asyncio.run(run_extract(batch_size=batch_size, max_cost=max_cost, config=config))


if __name__ == "__main__":
    app()

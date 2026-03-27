#!/usr/bin/env python3
"""
CLI: export extracted ideas to CSV or JSON.

Examples
--------
python scripts/export.py --format csv --output ideas.csv
python scripts/export.py --format json --output ideas.json
python scripts/export.py --format csv --filter-testability high --output high_testability.csv
python scripts/export.py --format csv --filter-confidence high --filter-geographic india
"""

import asyncio
import csv
import json
import sys
from pathlib import Path

import aiosqlite
import typer
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from src.orchestrator import load_config
from src.utils import setup_logging

app = typer.Typer(add_completion=False)


async def _load_ideas(
    db_path: str,
    testability: str | None,
    confidence: str | None,
    geographic: str | None,
    idea_type: str | None,
) -> list[dict]:
    clauses = []
    params  = []

    if testability:
        clauses.append("i.testability = ?")
        params.append(testability)
    if confidence:
        clauses.append("i.confidence = ?")
        params.append(confidence)
    if geographic:
        clauses.append("i.geographic_relevance = ?")
        params.append(geographic)
    if idea_type:
        clauses.append("i.idea_type = ?")
        params.append(idea_type)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT
            i.id,
            i.idea_type,
            i.name,
            i.description,
            i.mechanism,
            i.data_requirements,
            i.testability,
            i.asset_class,
            i.geographic_relevance,
            i.time_horizon,
            i.novelty_assessment,
            i.confidence,
            i.source_quote,
            i.tags_json,
            i.cluster_id,
            i.created_at,
            d.source_platform,
            d.title  AS doc_title,
            d.author AS doc_author,
            d.date   AS doc_date,
            d.url    AS doc_url
        FROM ideas i
        JOIN documents d ON d.id = i.document_id
        {where}
        ORDER BY i.confidence DESC, i.testability DESC, i.id
    """
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def _flatten(row: dict) -> dict:
    """Parse JSON tags field for flat export."""
    try:
        row["tags"] = ", ".join(json.loads(row.get("tags_json") or "[]"))
    except (json.JSONDecodeError, TypeError):
        row["tags"] = ""
    row.pop("tags_json", None)
    return row


@app.command()
def main(
    format: str = typer.Option(
        "csv",
        "--format", "-f",
        help="Output format: csv | json",
    ),
    output: str = typer.Option(
        "ideas.csv",
        "--output", "-o",
        help="Output file path",
    ),
    filter_testability: str = typer.Option(
        None,
        "--filter-testability",
        help="Filter by testability: high | medium | low",
    ),
    filter_confidence: str = typer.Option(
        None,
        "--filter-confidence",
        help="Filter by confidence: high | medium | low",
    ),
    filter_geographic: str = typer.Option(
        None,
        "--filter-geographic",
        help="Filter by geographic_relevance: india | us | global | ...",
    ),
    filter_type: str = typer.Option(
        None,
        "--filter-type",
        help="Filter by idea_type: strategy_signal | factor | alt_data_source | ...",
    ),
    config_path: str = typer.Option("config.yaml", "--config", "-c"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    setup_logging(log_level)
    config   = load_config(config_path)
    db_path  = config.get("database", {}).get("path", "data/quant_ideas.db")

    ideas = asyncio.run(
        _load_ideas(
            db_path,
            testability=filter_testability,
            confidence=filter_confidence,
            geographic=filter_geographic,
            idea_type=filter_type,
        )
    )

    if not ideas:
        print("No ideas matched the filter criteria.")
        return

    ideas = [_flatten(row) for row in ideas]

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "json":
        with open(out, "w", encoding="utf-8") as f:
            json.dump(ideas, f, indent=2, ensure_ascii=False, default=str)
        print(f"Exported {len(ideas)} ideas → {out}")

    else:  # CSV
        if not ideas:
            return
        fieldnames = list(ideas[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ideas)
        print(f"Exported {len(ideas)} ideas → {out}")


if __name__ == "__main__":
    app()

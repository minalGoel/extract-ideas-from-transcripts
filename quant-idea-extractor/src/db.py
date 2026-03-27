import json
from pathlib import Path

import aiosqlite
from loguru import logger

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id             TEXT    NOT NULL,
    source_platform       TEXT    NOT NULL,
    title                 TEXT,
    author                TEXT,
    date                  TEXT,
    url                   TEXT,
    text                  TEXT,
    metadata_json         TEXT,
    char_count            INTEGER,
    collected_at          TEXT,
    extracted             INTEGER DEFAULT 0,
    extraction_tokens_in  INTEGER,
    extraction_tokens_out INTEGER,
    created_at            TEXT    DEFAULT (datetime('now')),
    UNIQUE(source_platform, source_id)
)
"""

_CREATE_IDEAS = """
CREATE TABLE IF NOT EXISTS ideas (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id          INTEGER NOT NULL,
    idea_type            TEXT,
    name                 TEXT,
    description          TEXT,
    mechanism            TEXT,
    data_requirements    TEXT,
    testability          TEXT,
    asset_class          TEXT,
    geographic_relevance TEXT,
    time_horizon         TEXT,
    novelty_assessment   TEXT,
    confidence           TEXT,
    source_quote         TEXT,
    tags_json            TEXT,
    cluster_id           INTEGER,
    created_at           TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (document_id) REFERENCES documents(id)
)
"""

_CREATE_EXTRACTION_LOG = """
CREATE TABLE IF NOT EXISTS extraction_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id    INTEGER,
    timestamp      TEXT    DEFAULT (datetime('now')),
    model_used     TEXT,
    tokens_in      INTEGER,
    tokens_out     INTEGER,
    cost_estimate  REAL,
    success        INTEGER,
    error_message  TEXT
)
"""

_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_platform_id ON documents(source_platform, source_id)",
    "CREATE INDEX IF NOT EXISTS idx_docs_extracted    ON documents(extracted)",
    "CREATE INDEX IF NOT EXISTS idx_ideas_type        ON ideas(idea_type)",
    "CREATE INDEX IF NOT EXISTS idx_ideas_testability ON ideas(testability)",
    "CREATE INDEX IF NOT EXISTS idx_ideas_confidence  ON ideas(confidence)",
    "CREATE INDEX IF NOT EXISTS idx_ideas_cluster     ON ideas(cluster_id)",
]


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

async def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_CREATE_DOCUMENTS)
        await db.execute(_CREATE_IDEAS)
        await db.execute(_CREATE_EXTRACTION_LOG)
        for stmt in _INDEXES:
            await db.execute(stmt)
        await db.commit()
    logger.info(f"Database ready at {db_path}")


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

async def insert_document(db: aiosqlite.Connection, doc) -> int | None:
    """Insert a NormalizedDocument. Returns new row id, or None if duplicate."""
    try:
        cursor = await db.execute(
            """
            INSERT INTO documents
                (source_id, source_platform, title, author, date, url, text,
                 metadata_json, char_count, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_platform, source_id) DO NOTHING
            """,
            (
                doc.source_id,
                doc.source_platform,
                doc.title,
                doc.author,
                doc.date,
                doc.url,
                doc.text,
                json.dumps(doc.metadata),
                doc.char_count,
                doc.collected_at,
            ),
        )
        await db.commit()
        return cursor.lastrowid if cursor.lastrowid else None
    except Exception as exc:
        logger.error(f"Error inserting document {doc.source_id}: {exc}")
        return None


async def get_unextracted_documents(
    db: aiosqlite.Connection, limit: int = 100
) -> list[dict]:
    async with db.execute(
        "SELECT * FROM documents WHERE extracted = 0 ORDER BY created_at LIMIT ?",
        (limit,),
    ) as cur:
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


async def mark_document_extracted(
    db: aiosqlite.Connection,
    doc_id: int,
    tokens_in: int,
    tokens_out: int,
) -> None:
    await db.execute(
        """
        UPDATE documents
        SET extracted = 1, extraction_tokens_in = ?, extraction_tokens_out = ?
        WHERE id = ?
        """,
        (tokens_in, tokens_out, doc_id),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Ideas
# ---------------------------------------------------------------------------

async def insert_ideas(
    db: aiosqlite.Connection, document_id: int, ideas: list[dict]
) -> None:
    for idea in ideas:
        await db.execute(
            """
            INSERT INTO ideas
                (document_id, idea_type, name, description, mechanism,
                 data_requirements, testability, asset_class,
                 geographic_relevance, time_horizon, novelty_assessment,
                 confidence, source_quote, tags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                idea.get("idea_type"),
                idea.get("name"),
                idea.get("description"),
                idea.get("mechanism"),
                idea.get("data_requirements"),
                idea.get("testability"),
                idea.get("asset_class"),
                idea.get("geographic_relevance"),
                idea.get("time_horizon"),
                idea.get("novelty_assessment"),
                idea.get("confidence"),
                idea.get("source_quote"),
                json.dumps(idea.get("tags", [])),
            ),
        )
    await db.commit()


async def get_all_ideas(db: aiosqlite.Connection) -> list[dict]:
    async with db.execute("SELECT * FROM ideas ORDER BY id") as cur:
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Extraction log
# ---------------------------------------------------------------------------

async def log_extraction(
    db: aiosqlite.Connection,
    doc_id: int,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    success: bool,
    error: str | None = None,
) -> None:
    await db.execute(
        """
        INSERT INTO extraction_log
            (document_id, model_used, tokens_in, tokens_out, cost_estimate,
             success, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, model, tokens_in, tokens_out, cost, int(success), error),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

async def get_stats(db: aiosqlite.Connection) -> dict:
    stats: dict = {}

    async with db.execute(
        "SELECT source_platform, COUNT(*) FROM documents GROUP BY source_platform"
    ) as cur:
        stats["docs_by_source"] = dict(await cur.fetchall())

    async with db.execute(
        "SELECT COUNT(*) FROM documents WHERE extracted = 1"
    ) as cur:
        stats["extracted_docs"] = (await cur.fetchone())[0]

    async with db.execute(
        "SELECT COUNT(*) FROM documents"
    ) as cur:
        stats["total_docs"] = (await cur.fetchone())[0]

    async with db.execute(
        "SELECT COUNT(*) FROM ideas"
    ) as cur:
        stats["total_ideas"] = (await cur.fetchone())[0]

    async with db.execute(
        "SELECT idea_type, COUNT(*) FROM ideas GROUP BY idea_type"
    ) as cur:
        stats["ideas_by_type"] = dict(await cur.fetchall())

    async with db.execute(
        "SELECT SUM(cost_estimate) FROM extraction_log WHERE success = 1"
    ) as cur:
        row = await cur.fetchone()
        stats["total_cost_usd"] = round(row[0] or 0.0, 6)

    return stats

"""
Idea deduplication and clustering.

Algorithm:
1. Load all extracted ideas from the DB.
2. Group by idea_type (reduces false-positive cross-type matches).
3. Within each group, compute TF-IDF on name + description + mechanism.
4. Run AgglomerativeClustering (precomputed cosine distance, complete linkage).
5. Write cluster_id back to each idea row.

No Claude API calls needed — this is purely local ML.
"""

import asyncio

import aiosqlite
import numpy as np
from loguru import logger


async def run_dedup(db: aiosqlite.Connection, config: dict) -> int:
    """
    Cluster ideas and update cluster_id in the DB.
    Returns total number of clusters assigned.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        logger.error(f"scikit-learn not available: {exc}. Run: pip install scikit-learn")
        return 0

    dc = config.get("dedup", {})
    threshold = float(dc.get("similarity_threshold", 0.6))

    # Load all ideas
    async with db.execute("SELECT id, idea_type, name, description, mechanism FROM ideas") as cur:
        rows = await cur.fetchall()

    if len(rows) < 2:
        logger.info("Dedup: fewer than 2 ideas — nothing to cluster")
        return 0

    # Build lookup structures
    idea_ids    = [r[0] for r in rows]
    idea_types  = [r[1] or "other" for r in rows]
    texts       = [
        " ".join(filter(None, [r[2], r[3], r[4]]))
        for r in rows
    ]

    unique_types = sorted(set(idea_types))
    cluster_offset = 0
    total_updates  = 0

    for itype in unique_types:
        indices = [i for i, t in enumerate(idea_types) if t == itype]

        if len(indices) == 1:
            # Singleton — give it its own cluster
            await db.execute(
                "UPDATE ideas SET cluster_id = ? WHERE id = ?",
                (cluster_offset, idea_ids[indices[0]]),
            )
            cluster_offset += 1
            total_updates  += 1
            continue

        type_texts = [texts[i] for i in indices]

        # TF-IDF
        vec = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )
        try:
            mat = vec.fit_transform(type_texts)
        except Exception as exc:
            logger.warning(f"Dedup: TF-IDF failed for type '{itype}': {exc}")
            for idx in indices:
                await db.execute(
                    "UPDATE ideas SET cluster_id = ? WHERE id = ?",
                    (cluster_offset, idea_ids[idx]),
                )
                cluster_offset += 1
                total_updates  += 1
            continue

        # Cosine similarity → distance matrix
        sim = cosine_similarity(mat)
        dist = np.clip(1.0 - sim, 0.0, 1.0)

        try:
            agg = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1.0 - threshold,
                metric="precomputed",
                linkage="complete",
            )
            labels = agg.fit_predict(dist)
        except Exception as exc:
            logger.warning(f"Dedup: clustering failed for '{itype}': {exc}")
            labels = list(range(len(indices)))   # each idea its own cluster

        for pos, idx in enumerate(indices):
            cid = cluster_offset + int(labels[pos])
            await db.execute(
                "UPDATE ideas SET cluster_id = ? WHERE id = ?",
                (cid, idea_ids[idx]),
            )
            total_updates += 1

        cluster_offset += int(max(labels)) + 1

    await db.commit()
    logger.info(
        f"Dedup: {len(rows)} ideas → {cluster_offset} clusters "
        f"({total_updates} rows updated)"
    )
    return cluster_offset

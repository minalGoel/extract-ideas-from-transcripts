#!/usr/bin/env python3
"""
Deduplicate and consolidate extracted quant-trading ideas using local sentence
embeddings and two-pass clustering. No API calls — runs entirely on-device.

Pass 1 — Dedup (cosine ≥ 0.75): merge near-identical rephrases of the same claim.
Pass 2 — Topic grouping (cosine ≥ 0.55): cluster related ideas into reviewable
          topic groups so you can scan ~1-3k topics instead of ~25k claims.

Hardware target: MacBook Air M2 16GB, <30 min runtime.

Outputs:
  deduplicated_ideas.csv   — one row per unique idea (after Pass 1)
  topic_clusters.csv       — one row per topic group (after Pass 2), sorted by
                             how many speakers/videos mention that topic
  dedup_stats.json         — timing and summary statistics
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROGRESS_FILE = "extract_ideas_progress.jsonl"
OUTPUT_DEDUP_CSV = "deduplicated_ideas.csv"
OUTPUT_TOPICS_CSV = "topic_clusters.csv"
OUTPUT_JSONL = "deduplicated_ideas.jsonl"
STATS_FILE = "dedup_stats.json"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Pass 1: merge near-identical rephrases
DEDUP_THRESHOLD = 0.75
# Pass 2: group related ideas into topics
TOPIC_THRESHOLD = 0.55

QUALITY_RANK = {"high": 3, "medium": 2, "low": 1, "noise": 0}

# Normalize messy category labels from LLM output
CATEGORY_MAP = {
    "STRATEGY": "STRATEGY",
    "MARKET_STRUCTURE": "MARKET_STRUCTURE",
    "RISK": "RISK",
    "ALT_DATA": "ALT_DATA",
    "TOOL": "TOOL",
    "MICROSTRUCTURE": "MICROSTRUCTURE",
    "MACRO": "MACRO",
    "MACROECONOMICS": "MACRO",
    "MACRO_STRUCTURE": "MACRO",
    "MACROSTRUCTURE": "MACRO",
    "MACROE": "MACRO",
    "MACROstructure": "MACRO",
    "MACRO?": "MACRO",
    "MACROECONOMY": "MACRO",
    "MACROURE": "MACRO",
    "MACRODATA": "MACRO",
    "MACROECON": "MACRO",
    "FX": "STRATEGY",
    "COMMODITIES": "STRATEGY",
    "SUPPLY_CHAIN": "ALT_DATA",
    "FUNDAMENTAL": "STRATEGY",
    "REGULATION": "MARKET_STRUCTURE",
    "TECHNOLOGY": "TOOL",
    "METRIC": "STRATEGY",
}

DROP_CATEGORIES = {"SKIP", "EDUCATOR", "EDUCATION"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_claims(progress_file: str) -> list[dict]:
    """Load all non-noise claims from the progress JSONL."""
    claims = []
    seen_videos = set()
    with open(progress_file) as f:
        for line in f:
            rec = json.loads(line)
            if rec["status"] != "ok":
                continue
            vid = rec["video_id"]
            if vid in seen_videos:
                continue
            seen_videos.add(vid)

            result = rec["result"]
            quality = result.get("quality", "noise")
            if quality == "noise":
                continue

            speaker = result.get("speaker", {})
            theme = result.get("theme", "")
            gap = result.get("gap", "")

            for claim in result.get("claims", []):
                raw_cat = claim.get("cat", "UNKNOWN")
                cat = CATEGORY_MAP.get(raw_cat, raw_cat)
                if cat in DROP_CATEGORIES:
                    continue

                claims.append({
                    "video_id": vid,
                    "quality": quality,
                    "quality_rank": QUALITY_RANK.get(quality, 0),
                    "speaker_name": speaker.get("name"),
                    "speaker_affiliation": speaker.get("affiliation"),
                    "speaker_type": speaker.get("type"),
                    "theme": theme,
                    "gap": gap,
                    "cat": cat,
                    "what": claim.get("what", ""),
                    "why": claim.get("why", ""),
                    "needs": claim.get("needs", ""),
                    "anchor": claim.get("anchor", ""),
                    "asset": claim.get("asset", ""),
                    "market": claim.get("market", ""),
                    "freq": claim.get("freq", ""),
                    "entities": claim.get("entities", []),
                    "numbers": claim.get("numbers"),
                    "india": claim.get("india", ""),
                })

    return claims


def build_embed_text(claim: dict) -> str:
    """Combine what + why + category for richer semantic signal."""
    what = claim["what"].strip()
    why = claim["why"].strip() if claim["why"] else ""
    cat = claim["cat"]
    # Prefix with category to bias clustering within-category
    prefix = f"[{cat}] "
    if why:
        return f"{prefix}{what} — {why}"
    return f"{prefix}{what}"


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------
def agglom_cluster(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """
    Agglomerative clustering on L2-normalized embeddings.
    threshold: cosine similarity cutoff (pairs >= threshold merge).
    Returns label array.
    """
    n = len(embeddings)
    if n <= 1:
        return np.array([0] * n)

    # Compute cosine distance matrix using matrix multiply (fast for normalized vecs)
    sim = embeddings @ embeddings.T
    np.clip(sim, -1, 1, out=sim)
    dist = 1.0 - sim

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - threshold,
        metric="precomputed",
        linkage="average",
    )
    return clustering.fit_predict(dist)


def pick_representative(
    member_indices: list[int],
    claims: list[dict],
    embeddings: np.ndarray,
) -> int:
    """Pick the best representative from a cluster: highest quality, closest to centroid."""
    if len(member_indices) == 1:
        return member_indices[0]

    emb = embeddings[member_indices]
    centroid = emb.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid /= norm

    best_idx = member_indices[0]
    best_score = -999.0
    for j, gidx in enumerate(member_indices):
        q = claims[gidx]["quality_rank"]
        cos = float(emb[j] @ centroid)
        score = q * 10 + cos
        if score > best_score:
            best_score = score
            best_idx = gidx
    return best_idx


def cluster_avg_sim(member_indices: list[int], embeddings: np.ndarray) -> float:
    """Average pairwise cosine similarity within a cluster."""
    if len(member_indices) <= 1:
        return 1.0
    emb = embeddings[member_indices]
    sim = emb @ emb.T
    n = len(member_indices)
    triu = np.triu_indices(n, k=1)
    return float(sim[triu].mean())


def collect_unique(key: str, indices: list[int], claims: list[dict]) -> list[str]:
    """Collect unique non-None values for a field across cluster members."""
    vals = set()
    for idx in indices:
        v = claims[idx].get(key)
        if v:
            vals.add(v)
    return sorted(vals)


# ---------------------------------------------------------------------------
# Pass 1: Dedup
# ---------------------------------------------------------------------------
def pass1_dedup(
    claims: list[dict],
    embeddings: np.ndarray,
    threshold: float,
) -> list[dict]:
    """
    Merge near-identical claims (global, cross-category).
    Returns list of dedup cluster records.
    """
    print(f"\n{'─'*60}")
    print(f"  PASS 1 — Dedup (threshold={threshold})")
    print(f"{'─'*60}")

    t0 = time.time()
    labels = agglom_cluster(embeddings, threshold)
    n_clusters = labels.max() + 1
    print(f"  Agglomerative clustering: {len(claims)} → {n_clusters} clusters [{time.time()-t0:.1f}s]")

    clusters = []
    for cid in range(n_clusters):
        members = list(np.where(labels == cid)[0])
        rep_idx = pick_representative(members, claims, embeddings)
        clusters.append({
            "representative_idx": rep_idx,
            "representative": claims[rep_idx],
            "cluster_size": len(members),
            "member_indices": members,
            "avg_similarity": cluster_avg_sim(members, embeddings),
        })

    # Stats
    multi = [c for c in clusters if c["cluster_size"] > 1]
    merged_count = sum(c["cluster_size"] for c in multi)
    print(f"  Singletons: {len(clusters) - len(multi)}")
    print(f"  Multi-member clusters: {len(multi)} (merged {merged_count} claims)")
    print(f"  Dedup ratio: {1 - len(clusters)/len(claims):.1%}")

    return clusters


# ---------------------------------------------------------------------------
# Pass 2: Topic grouping
# ---------------------------------------------------------------------------
def pass2_topics(
    dedup_clusters: list[dict],
    claims: list[dict],
    embeddings: np.ndarray,
    threshold: float,
) -> list[dict]:
    """
    Group deduplicated ideas into broader topic clusters.
    Uses only the representative embeddings from Pass 1.
    """
    print(f"\n{'─'*60}")
    print(f"  PASS 2 — Topic grouping (threshold={threshold})")
    print(f"{'─'*60}")

    # Get representative embeddings
    rep_indices = [c["representative_idx"] for c in dedup_clusters]
    rep_emb = embeddings[rep_indices]  # (n_dedup, 768)

    t0 = time.time()
    labels = agglom_cluster(rep_emb, threshold)
    n_topics = labels.max() + 1
    print(f"  Clustering {len(dedup_clusters)} ideas → {n_topics} topics [{time.time()-t0:.1f}s]")

    # Build topic groups
    topics = []
    for tid in range(n_topics):
        # Which dedup clusters belong to this topic?
        dedup_member_mask = (labels == tid)
        dedup_members = list(np.where(dedup_member_mask)[0])

        # Collect ALL original claim indices across all dedup clusters in this topic
        all_claim_indices = []
        for di in dedup_members:
            all_claim_indices.extend(dedup_clusters[di]["member_indices"])

        # Pick the best representative across all dedup clusters in this topic
        dedup_rep_emb = rep_emb[dedup_members]
        centroid = dedup_rep_emb.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        best_dedup_idx = dedup_members[0]
        best_score = -999.0
        for j, di in enumerate(dedup_members):
            q = dedup_clusters[di]["representative"]["quality_rank"]
            sz = dedup_clusters[di]["cluster_size"]
            cos = float(dedup_rep_emb[j] @ centroid)
            # Favor quality, then cluster size (well-corroborated), then centrality
            score = q * 100 + sz * 10 + cos
            if score > best_score:
                best_score = score
                best_idx = di
                best_dedup_idx = di

        rep_claim = dedup_clusters[best_dedup_idx]["representative"]

        # Collect metadata
        speakers = collect_unique("speaker_name", all_claim_indices, claims)
        videos = collect_unique("video_id", all_claim_indices, claims)
        categories = list(set(claims[i]["cat"] for i in all_claim_indices))

        # Collect the top ideas in this topic (representatives of each dedup cluster)
        topic_ideas = []
        for di in dedup_members:
            dc = dedup_clusters[di]
            topic_ideas.append({
                "what": dc["representative"]["what"],
                "why": dc["representative"]["why"],
                "quality": dc["representative"]["quality"],
                "cluster_size": dc["cluster_size"],
            })
        # Sort by quality then cluster size
        topic_ideas.sort(key=lambda x: (-QUALITY_RANK.get(x["quality"], 0), -x["cluster_size"]))

        topics.append({
            "topic_id": tid,
            "representative": rep_claim,
            "num_ideas": len(dedup_members),
            "num_claims": len(all_claim_indices),
            "num_speakers": len(speakers),
            "num_videos": len(videos),
            "categories": categories,
            "speakers": speakers,
            "video_ids": videos,
            "ideas": topic_ideas,
            "claim_indices": all_claim_indices,
        })

    # Sort by num_speakers desc, then num_claims desc (most-discussed first)
    topics.sort(key=lambda t: (-t["num_speakers"], -t["num_claims"]))

    # Reassign topic IDs after sorting
    for i, t in enumerate(topics):
        t["topic_id"] = i + 1

    multi = [t for t in topics if t["num_ideas"] > 1]
    print(f"  Single-idea topics: {len(topics) - len(multi)}")
    print(f"  Multi-idea topics: {len(multi)}")
    print(f"  Consolidation ratio: {1 - len(topics)/len(dedup_clusters):.1%}")

    return topics


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_dedup_csv(clusters: list[dict], claims: list[dict], output_path: str):
    """Write one row per unique idea (Pass 1 output)."""
    # Sort: category, quality desc, cluster_size desc
    clusters_sorted = sorted(clusters, key=lambda c: (
        c["representative"]["cat"],
        -c["representative"]["quality_rank"],
        -c["cluster_size"],
    ))

    rows = []
    for i, cluster in enumerate(clusters_sorted):
        rep = cluster["representative"]
        videos = collect_unique("video_id", cluster["member_indices"], claims)
        speakers = collect_unique("speaker_name", cluster["member_indices"], claims)
        rows.append({
            "idea_id": i + 1,
            "cat": rep["cat"],
            "quality": rep["quality"],
            "what": rep["what"],
            "why": rep["why"],
            "needs": rep["needs"],
            "asset": rep["asset"],
            "market": rep["market"],
            "freq": rep["freq"],
            "india": rep["india"],
            "entities": json.dumps(rep["entities"]) if rep["entities"] else "",
            "numbers": json.dumps(rep["numbers"]) if rep["numbers"] else "",
            "anchor": rep["anchor"],
            "speaker_name": rep["speaker_name"] or "",
            "speaker_affiliation": rep["speaker_affiliation"] or "",
            "speaker_type": rep["speaker_type"] or "",
            "cluster_size": cluster["cluster_size"],
            "avg_similarity": round(cluster["avg_similarity"], 3),
            "num_speakers": len(speakers),
            "speakers": json.dumps(speakers) if len(speakers) > 1 else (speakers[0] if speakers else ""),
            "num_source_videos": len(videos),
            "source_video_ids": json.dumps(videos),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Wrote {len(df)} rows to {output_path}")
    return df


def export_topics_csv(topics: list[dict], output_path: str):
    """Write one row per topic group (Pass 2 output)."""
    rows = []
    for t in topics:
        rep = t["representative"]
        # Build a short summary of the top 5 ideas
        top_ideas = t["ideas"][:5]
        ideas_summary = " | ".join(
            f"[{idea['quality'][0].upper()}] {idea['what'][:100]}"
            for idea in top_ideas
        )

        rows.append({
            "topic_id": t["topic_id"],
            "representative_idea": rep["what"],
            "representative_why": rep["why"],
            "primary_cat": rep["cat"],
            "all_categories": json.dumps(t["categories"]),
            "quality": rep["quality"],
            "num_unique_ideas": t["num_ideas"],
            "num_total_claims": t["num_claims"],
            "num_speakers": t["num_speakers"],
            "num_videos": t["num_videos"],
            "top_ideas": ideas_summary,
            "speakers": json.dumps(t["speakers"][:10]),  # cap at 10
            "asset": rep["asset"],
            "market": rep["market"],
            "freq": rep["freq"],
            "india": rep["india"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Wrote {len(df)} rows to {output_path}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate and consolidate extracted quant ideas (two-pass)"
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=DEDUP_THRESHOLD,
        help=f"Pass 1 cosine threshold for dedup (default: {DEDUP_THRESHOLD})"
    )
    parser.add_argument(
        "--topic-threshold", type=float, default=TOPIC_THRESHOLD,
        help=f"Pass 2 cosine threshold for topic grouping (default: {TOPIC_THRESHOLD})"
    )
    parser.add_argument(
        "--progress-file", default=PROGRESS_FILE,
        help=f"Input JSONL file (default: {PROGRESS_FILE})"
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device: cpu, mps, or cuda (default: auto-detect)"
    )
    parser.add_argument(
        "--skip-pass2", action="store_true",
        help="Only run Pass 1 (dedup), skip topic grouping"
    )
    args = parser.parse_args()

    t_start = time.time()

    # --- Step 1: Load ---
    print(f"[1/6] Loading claims from {args.progress_file}...")
    claims = load_claims(args.progress_file)
    print(f"  Loaded {len(claims)} claims")

    cat_counts = defaultdict(int)
    quality_counts = defaultdict(int)
    for c in claims:
        cat_counts[c["cat"]] += 1
        quality_counts[c["quality"]] += 1
    print("  Categories:", dict(sorted(cat_counts.items(), key=lambda x: -x[1])))
    print("  Quality:", dict(sorted(quality_counts.items(), key=lambda x: -x[1])))

    # --- Step 2: Load model ---
    print(f"\n[2/6] Loading embedding model: {MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer

    device = args.device
    if device is None:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"  Using device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    t_model = time.time()
    print(f"  Model loaded in {t_model - t_start:.1f}s")

    # --- Step 3: Embed ---
    print(f"\n[3/6] Embedding {len(claims)} claims...")
    texts = [build_embed_text(c) for c in claims]
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    t_embed = time.time()
    print(f"  Embedded in {t_embed - t_model:.1f}s  ({embeddings.shape})")

    # Free model memory before clustering (need RAM for distance matrices)
    del model
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    # --- Step 4: Pass 1 — Dedup ---
    print(f"\n[4/6] Pass 1 — Dedup...")
    dedup_clusters = pass1_dedup(claims, embeddings, args.dedup_threshold)
    t_pass1 = time.time()

    # --- Step 5: Pass 2 — Topic grouping ---
    if not args.skip_pass2:
        print(f"\n[5/6] Pass 2 — Topic grouping...")
        topics = pass2_topics(dedup_clusters, claims, embeddings, args.topic_threshold)
        t_pass2 = time.time()
    else:
        topics = None
        t_pass2 = time.time()

    # --- Step 6: Export ---
    print(f"\n[6/6] Exporting results...")
    dedup_df = export_dedup_csv(dedup_clusters, claims, OUTPUT_DEDUP_CSV)

    # JSONL with full detail
    with open(OUTPUT_JSONL, "w") as f:
        for i, cluster in enumerate(sorted(dedup_clusters, key=lambda c: (
            c["representative"]["cat"],
            -c["representative"]["quality_rank"],
            -c["cluster_size"],
        ))):
            rec = {
                "idea_id": i + 1,
                "representative": cluster["representative"],
                "cluster_size": cluster["cluster_size"],
                "avg_similarity": round(cluster["avg_similarity"], 3),
                "source_videos": collect_unique("video_id", cluster["member_indices"], claims),
                "speakers": collect_unique("speaker_name", cluster["member_indices"], claims),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"  Wrote {len(dedup_clusters)} records to {OUTPUT_JSONL}")

    if topics:
        topics_df = export_topics_csv(topics, OUTPUT_TOPICS_CSV)

    # --- Stats ---
    t_end = time.time()
    stats = {
        "input_claims": len(claims),
        "pass1_dedup_threshold": args.dedup_threshold,
        "pass1_unique_ideas": len(dedup_clusters),
        "pass1_dedup_ratio": round(1 - len(dedup_clusters) / len(claims), 3),
        "model": MODEL_NAME,
        "device": device,
    }
    if topics:
        stats["pass2_topic_threshold"] = args.topic_threshold
        stats["pass2_num_topics"] = len(topics)
        stats["pass2_consolidation_ratio"] = round(1 - len(topics) / len(dedup_clusters), 3)

        # Top 30 topics by speaker count
        stats["top_topics"] = [
            {
                "topic_id": t["topic_id"],
                "representative": t["representative"]["what"],
                "num_ideas": t["num_ideas"],
                "num_claims": t["num_claims"],
                "num_speakers": t["num_speakers"],
                "categories": t["categories"],
            }
            for t in topics[:30]
        ]

    stats["largest_dedup_clusters"] = sorted(
        [{"what": c["representative"]["what"], "size": c["cluster_size"], "cat": c["representative"]["cat"]}
         for c in dedup_clusters if c["cluster_size"] >= 3],
        key=lambda x: -x["size"],
    )[:20]

    stats["timing"] = {
        "model_load_s": round(t_model - t_start, 1),
        "embedding_s": round(t_embed - t_model, 1),
        "pass1_s": round(t_pass1 - t_embed, 1),
        "pass2_s": round(t_pass2 - t_pass1, 1) if not args.skip_pass2 else 0,
        "total_s": round(t_end - t_start, 1),
    }

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote stats to {STATS_FILE}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  INPUT:  {len(claims)} claims from {len(set(c['video_id'] for c in claims))} videos")
    print(f"  PASS 1: {len(dedup_clusters)} unique ideas (dedup {stats['pass1_dedup_ratio']:.1%})")
    if topics:
        print(f"  PASS 2: {len(topics)} topic clusters (consolidated {stats['pass2_consolidation_ratio']:.1%})")
        multi_topics = [t for t in topics if t["num_ideas"] > 1]
        print(f"          {len(multi_topics)} multi-idea topics, {len(topics) - len(multi_topics)} singletons")
    print(f"  TIME:   {stats['timing']['total_s']:.1f}s total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

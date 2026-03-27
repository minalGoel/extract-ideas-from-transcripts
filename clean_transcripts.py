#!/usr/bin/env python3
"""
clean_transcripts.py — Remove noise from auto-generated YouTube transcripts.

Reads temp_transcripts/*.json, cleans the transcript_text field, writes
cleaned copies to cleaned_transcripts/ (same JSON schema, smaller text).

Cleaning passes:
  1. Filler words (um, uh, you know, basically, etc.)
  2. YouTube boilerplate (subscribe, bell icon, sponsor reads)
  3. Caption artifacts ([Music], [Applause], [Laughter])
  4. Consecutive duplicate words (the the → the)
  5. Collapse excess whitespace

Safe for extraction — removes ONLY noise, never substantive content.
The verbatim_anchor field in extractions will be cleaner as a result.

Usage:
    python clean_transcripts.py                # clean all transcripts
    python clean_transcripts.py --stats-only   # just report savings, don't write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT       = Path(__file__).parent
TEMP_DIR   = ROOT / "temp_transcripts"
CLEAN_DIR  = ROOT / "cleaned_transcripts"


# ---------------------------------------------------------------------------
# Cleaning patterns
# ---------------------------------------------------------------------------

# Pure filler words — always safe to strip
_FILLER_WORDS = re.compile(
    r"\b(?:uh|um|hmm|hm|ah|er|eh)[,.]?\s+",
    re.IGNORECASE,
)

# Filler phrases — safe to strip in spoken transcripts
_FILLER_PHRASES = re.compile(
    r"\b(?:"
    r"you know[,.]?\s*"
    r"|i mean[,.]?\s*"
    r"|kind of\s+"
    r"|sort of\s+"
    r"|a little bit\s+"
    r"|at the end of the day[,.]?\s*"
    r"|basically[,.]?\s*"
    r"|okay so\s+"
    r"|so basically\s+"
    r"|right\?\s*"
    r")",
    re.IGNORECASE,
)

# YouTube boilerplate (subscribe / sponsor / links)
_YT_NOISE = re.compile(
    r"(?:"
    r"(?:please\s+)?(?:like\s+(?:and|&)\s+)?subscribe(?:\s+to\s+(?:my|the|our)\s+channel)?"
    r"|hit\s+the\s+(?:bell|notification|like)\s*(?:icon|button)?"
    r"|smash\s+that\s+(?:like|subscribe)\s*(?:button)?"
    r"|(?:link|links?)\s+(?:in|below)\s+(?:the\s+)?description"
    r"|comment\s+(?:down\s+)?below"
    r"|don'?t\s+forget\s+to\s+(?:like|subscribe|share)"
    r"|sponsored\s+by\b[^.]*?[.]"
    r"|brought\s+to\s+you\s+by\b[^.]*?[.]"
    r")",
    re.IGNORECASE,
)

# Caption artifacts
_CAPTION_ARTIFACTS = re.compile(
    r"\[(?:music|applause|laughter|silence|inaudible|foreign)\]",
    re.IGNORECASE,
)

# Excess whitespace
_MULTI_SPACE = re.compile(r"  +")


def clean_transcript(text: str) -> str:
    """Apply all cleaning passes to a transcript string."""
    if not text:
        return text

    # 1. Filler words
    text = _FILLER_WORDS.sub(" ", text)

    # 2. Filler phrases
    text = _FILLER_PHRASES.sub(" ", text)

    # 3. YouTube noise
    text = _YT_NOISE.sub(" ", text)

    # 4. Caption artifacts
    text = _CAPTION_ARTIFACTS.sub(" ", text)

    # 5. Collapse consecutive duplicate words ("the the" → "the")
    words = text.split()
    if words:
        deduped = [words[0]]
        for i in range(1, len(words)):
            if words[i].lower() != words[i - 1].lower():
                deduped.append(words[i])
        text = " ".join(deduped)

    # 6. Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Clean auto-generated YouTube transcripts.")
    ap.add_argument("--stats-only", action="store_true",
                    help="Report reduction stats without writing output files.")
    args = ap.parse_args()

    files = sorted(TEMP_DIR.glob("*.json"))
    if not files:
        print(f"No transcripts found in {TEMP_DIR}")
        sys.exit(1)

    if not args.stats_only:
        CLEAN_DIR.mkdir(exist_ok=True)

    total_before = 0
    total_after  = 0
    cleaned_count = 0
    skipped      = 0

    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue

        text = d.get("transcript_text") or ""
        if not text:
            skipped += 1
            if not args.stats_only:
                # Copy as-is (error / empty transcripts)
                (CLEAN_DIR / f.name).write_text(
                    json.dumps(d, ensure_ascii=False), encoding="utf-8"
                )
            continue

        before_words = len(text.split())
        cleaned_text = clean_transcript(text)
        after_words  = len(cleaned_text.split())

        total_before += before_words
        total_after  += after_words
        cleaned_count += 1

        if not args.stats_only:
            d["transcript_text"] = cleaned_text
            d["_cleaning"] = {
                "words_before": before_words,
                "words_after":  after_words,
                "reduction_pct": round((1 - after_words / before_words) * 100, 1)
                                 if before_words else 0,
            }
            (CLEAN_DIR / f.name).write_text(
                json.dumps(d, ensure_ascii=False), encoding="utf-8"
            )

    # Report
    reduction_pct = (1 - total_after / total_before) * 100 if total_before else 0
    tokens_saved  = int((total_before - total_after) * 1.33)

    print(f"Files processed : {cleaned_count:,}  (skipped {skipped:,} empty/errored)")
    print(f"Words before    : {total_before:,}")
    print(f"Words after     : {total_after:,}")
    print(f"Reduction       : {reduction_pct:.1f}%  ({total_before - total_after:,} words)")
    print(f"Est. tokens saved: {tokens_saved:,}")
    print()
    avg_before = total_before // max(cleaned_count, 1)
    avg_after  = total_after  // max(cleaned_count, 1)
    print(f"Per video avg   : {avg_before:,} → {avg_after:,} words")

    if not args.stats_only:
        print(f"\nCleaned files written to: {CLEAN_DIR}/")
        print("Update extract_ideas.py to read from cleaned_transcripts/ instead of temp_transcripts/")


if __name__ == "__main__":
    main()

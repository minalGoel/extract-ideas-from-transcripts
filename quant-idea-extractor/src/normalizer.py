"""
Converts raw platform-specific data into a unified NormalizedDocument.
Every collector outputs raw dicts; this module is the single place where
cleaning, truncation and schema enforcement happen.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

MAX_TEXT_CHARS = 15_000
MIN_TEXT_CHARS = 200


@dataclass
class NormalizedDocument:
    source_id: str
    source_platform: str   # youtube | reddit | arxiv | ssrn | substack | twitter
    title: str
    author: str
    date: str              # ISO-8601 UTC
    url: str
    text: str              # cleaned content as plain text / markdown
    metadata: dict = field(default_factory=dict)
    char_count: int = 0
    collected_at: str = ""

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        if not self.collected_at:
            self.collected_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, rep in [
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "),
        ("&#x27;", "'"), ("&#x2F;", "/"),
    ]:
        text = text.replace(ent, rep)
    return text


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_youtube_captions(text: str) -> str:
    """Remove auto-caption noise: timing markers, repeated consecutive words."""
    text = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", text)
    # Collapse runs of the same word (common auto-caption artifact)
    text = re.sub(r"\b(\w+)(\s+\1){2,}", r"\1", text, flags=re.IGNORECASE)
    return text


def _truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Keep intro + outro, drop middle with a note."""
    if len(text) <= max_chars:
        return text
    intro = int(max_chars * 0.50)
    outro = int(max_chars * 0.35)
    return (
        text[:intro]
        + "\n\n[... content truncated for token budget ...]\n\n"
        + text[-outro:]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(
    source_id: str,
    source_platform: str,
    title: str,
    author: str,
    date: str,
    url: str,
    raw_text: str,
    metadata: dict | None = None,
) -> "NormalizedDocument | None":
    """
    Build a NormalizedDocument from raw inputs.
    Returns None if the cleaned text is too short to be useful.
    """
    text = raw_text or ""
    text = _strip_html(text)

    if source_platform == "youtube":
        text = _clean_youtube_captions(text)

    text = _normalize_whitespace(text)

    if len(text) < MIN_TEXT_CHARS:
        return None

    text = _truncate(text)

    return NormalizedDocument(
        source_id=source_id,
        source_platform=source_platform,
        title=(title or "").strip(),
        author=(author or "").strip(),
        date=(date or "").strip(),
        url=(url or "").strip(),
        text=text,
        metadata=metadata or {},
    )


def make_source_id(value: str) -> str:
    """Stable 16-char hex id from an arbitrary string (URL, content hash, etc.)."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]

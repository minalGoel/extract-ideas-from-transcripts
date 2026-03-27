"""
Reddit collector — uses PRAW (free OAuth2 app).

Required env vars:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import praw
import yaml
from loguru import logger

from ..normalizer import NormalizedDocument, normalize
from ..utils import RateLimiter
from .base import BaseCollector

_SEEDS = Path("seeds/subreddits.yaml")
_DEFAULT_SUBS = [
    "quant",
    "algotrading",
    "systematictrading",
    "QuantFinance",
    "options",
    "IndianStreetBets",
    "IndianStockMarket",
]


class RedditCollector(BaseCollector):
    def __init__(self, config: dict) -> None:
        self.config = config
        rc = config.get("collection", {}).get("reddit", {})
        self.min_score   = rc.get("min_score", 10)
        self.time_filter = rc.get("time_filter", "year")
        self.max_comments = rc.get("max_comments_per_post", 10)
        self._rl = RateLimiter(55)   # stay under Reddit's 60 req/min

        self._reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ.get(
                "REDDIT_USER_AGENT", "quant-idea-extractor/1.0"
            ),
        )

    # ------------------------------------------------------------------

    def _load_subreddits(self) -> list[str]:
        if _SEEDS.exists():
            with open(_SEEDS) as f:
                return yaml.safe_load(f).get("subreddits", _DEFAULT_SUBS)
        return _DEFAULT_SUBS

    def _build_text(self, submission) -> str:
        parts = [f"# {submission.title}\n"]
        if submission.selftext and submission.selftext not in ("[removed]", "[deleted]"):
            parts.append(submission.selftext)

        submission.comments.replace_more(limit=0)
        top_comments = sorted(
            [c for c in submission.comments.list()
             if hasattr(c, "score") and isinstance(c.score, int) and c.score > 0],
            key=lambda c: c.score,
            reverse=True,
        )[: self.max_comments]

        if top_comments:
            parts.append("\n## Top Comments\n")
            for c in top_comments:
                body = (c.body or "").strip()
                if body:
                    parts.append(f"**[score: {c.score}]** {body}\n")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------

    async def collect(self, limit: int | None = None) -> list[NormalizedDocument]:
        subs = self._load_subreddits()
        docs: list[NormalizedDocument] = []

        for sub_name in subs:
            logger.info(f"Reddit: collecting r/{sub_name}")
            try:
                sub = self._reddit.subreddit(sub_name)
                posts_seen = 0
                max_posts = limit or 200

                for submission in sub.top(
                    time_filter=self.time_filter, limit=max_posts
                ):
                    if submission.score < self.min_score:
                        continue
                    await self._rl.acquire()

                    try:
                        text = await asyncio.get_event_loop().run_in_executor(
                            None, self._build_text, submission
                        )
                    except Exception as exc:
                        logger.debug(f"Reddit: comment fetch failed for {submission.id}: {exc}")
                        text = f"# {submission.title}\n\n{submission.selftext or ''}"

                    doc = normalize(
                        source_id=submission.id,
                        source_platform="reddit",
                        title=submission.title,
                        author=str(submission.author) if submission.author else "[deleted]",
                        date=datetime.fromtimestamp(
                            submission.created_utc, tz=timezone.utc
                        ).isoformat(),
                        url=f"https://reddit.com{submission.permalink}",
                        raw_text=text,
                        metadata={
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "subreddit": sub_name,
                        },
                    )
                    if doc:
                        docs.append(doc)
                        posts_seen += 1

                logger.info(f"Reddit: r/{sub_name} → {posts_seen} posts")

            except Exception as exc:
                logger.error(f"Reddit: r/{sub_name} failed: {exc}")

        return docs

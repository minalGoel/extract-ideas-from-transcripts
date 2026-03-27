"""
Shared utilities: logging setup, async rate-limiting, retry decorator,
and cost estimation helpers.
"""

import asyncio
import sys
import time
from functools import wraps
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> — "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/pipeline.log",
        rotation="10 MB",
        retention="1 week",
        level="DEBUG",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Async rate limiter (token bucket, per-minute)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, calls_per_minute: int) -> None:
        self._interval = 60.0 / max(calls_per_minute, 1)
        self._last: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------

def retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """Decorator: retry an async function with exponential back-off."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"{func.__name__}: attempt {attempt + 1}/{max_retries} "
                            f"failed ({exc}). Retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__}: all {max_retries} attempts failed."
                        )
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Cost estimation (claude-sonnet-4-0 / claude-sonnet-4-20250514)
# ---------------------------------------------------------------------------

# $3 / 1M input tokens,  $15 / 1M output tokens
_INPUT_PRICE  = 3.0  / 1_000_000
_OUTPUT_PRICE = 15.0 / 1_000_000


def estimate_cost(tokens_in: int, tokens_out: int) -> float:
    return tokens_in * _INPUT_PRICE + tokens_out * _OUTPUT_PRICE

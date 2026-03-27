"""
Claude API extraction layer.

For each NormalizedDocument stored in the DB, calls Claude and extracts
a structured list of quant trading ideas.  Runs up to max_concurrent_calls
in parallel using asyncio.Semaphore.
"""

import asyncio
import json
import re
from pathlib import Path

import anthropic
from loguru import logger

from .db import insert_ideas, log_extraction, mark_document_extracted
from .utils import estimate_cost, retry_async

_PROMPT_FILE = Path("prompts/extraction.txt")

# Model IDs per the SDK model catalog
_DEFAULT_MODEL = "claude-sonnet-4-0"   # alias for claude-sonnet-4-20250514

# Per-token pricing for claude-sonnet-4-0 / claude-sonnet-4-20250514
_INPUT_PRICE  = 3.0  / 1_000_000
_OUTPUT_PRICE = 15.0 / 1_000_000


class IdeaExtractor:
    def __init__(self, config: dict) -> None:
        self.config = config
        ec = config.get("extraction", {})
        self.model          = ec.get("model", _DEFAULT_MODEL)
        self.max_concurrent = ec.get("max_concurrent_calls", 5)
        self.max_cost       = ec.get("max_cost_per_run", 10.0)

        self._sem   = asyncio.Semaphore(self.max_concurrent)
        self._client = anthropic.AsyncAnthropic()
        self.total_cost  = 0.0
        self.total_ideas = 0
        self._system = self._load_prompt()

    # ------------------------------------------------------------------

    def _load_prompt(self) -> str:
        if _PROMPT_FILE.exists():
            return _PROMPT_FILE.read_text(encoding="utf-8").strip()
        logger.warning("prompts/extraction.txt not found — using built-in fallback")
        return _FALLBACK_SYSTEM

    @staticmethod
    def _parse_ideas(raw: str) -> list[dict]:
        """
        Extract the JSON payload from Claude's response text.
        Claude may wrap the JSON in a markdown code block.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```", "", cleaned).strip()

        # Find the outermost {...} block
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return []

        try:
            data = json.loads(m.group())
            ideas = data.get("ideas", [])
            return [i for i in ideas if isinstance(i, dict)]
        except json.JSONDecodeError as exc:
            logger.debug(f"JSON parse error: {exc}")
            return []

    # ------------------------------------------------------------------

    @retry_async(max_retries=3, base_delay=2.0, exceptions=(anthropic.RateLimitError,))
    async def _call_claude(
        self, text: str, title: str
    ) -> tuple[list[dict], int, int]:
        """Returns (ideas, tokens_in, tokens_out)."""
        user_msg = f"Title: {title}\n\n---\n\n{text}"

        async with self._sem:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._system,
                messages=[{"role": "user", "content": user_msg}],
            )

        tokens_in  = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        raw_text   = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        ideas = self._parse_ideas(raw_text)
        return ideas, tokens_in, tokens_out

    # ------------------------------------------------------------------

    async def extract_document(
        self, db, doc: dict
    ) -> bool:
        """
        Extract ideas from one document row.
        Returns True (processed), False (cost cap hit — stop the run).
        """
        if self.total_cost >= self.max_cost:
            logger.warning(
                f"Cost cap ${self.max_cost:.2f} reached. "
                "Stopping further extraction."
            )
            return False

        doc_id = doc["id"]
        try:
            ideas, tokens_in, tokens_out = await self._call_claude(
                doc.get("text", ""), doc.get("title", "")
            )
            cost = tokens_in * _INPUT_PRICE + tokens_out * _OUTPUT_PRICE
            self.total_cost  += cost
            self.total_ideas += len(ideas)

            if ideas:
                await insert_ideas(db, doc_id, ideas)

            await mark_document_extracted(db, doc_id, tokens_in, tokens_out)
            await log_extraction(
                db, doc_id, self.model, tokens_in, tokens_out, cost, True
            )

            logger.info(
                f"doc {doc_id} [{doc.get('source_platform')}]: "
                f"{len(ideas)} ideas | "
                f"{tokens_in}+{tokens_out} tok | "
                f"${cost:.5f} | running ${self.total_cost:.4f}"
            )
            return True

        except anthropic.BadRequestError as exc:
            # Content policy or invalid request — log and skip
            logger.warning(f"doc {doc_id}: BadRequest ({exc})")
            await log_extraction(db, doc_id, self.model, 0, 0, 0.0, False, str(exc))
            await mark_document_extracted(db, doc_id, 0, 0)
            return True

        except Exception as exc:
            logger.error(f"doc {doc_id}: extraction error: {exc}")
            await log_extraction(db, doc_id, self.model, 0, 0, 0.0, False, str(exc))
            return True   # don't block the run on transient errors

    async def extract_batch(
        self, db, docs: list[dict]
    ) -> tuple[int, float]:
        """Process a list of doc rows concurrently. Returns (n_processed, cost)."""
        tasks = [self.extract_document(db, doc) for doc in docs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        n = sum(1 for r in results if r is True)
        return n, self.total_cost


# ---------------------------------------------------------------------------
# Fallback system prompt (used only if prompts/extraction.txt is missing)
# ---------------------------------------------------------------------------

_FALLBACK_SYSTEM = """\
You are a quantitative trading research analyst.
Extract all quantitative trading ideas from the provided content and return them
as a JSON object with an "ideas" array following the schema in the user prompt.
"""

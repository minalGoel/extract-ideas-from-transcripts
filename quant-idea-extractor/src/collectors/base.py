from abc import ABC, abstractmethod

from ..normalizer import NormalizedDocument


class BaseCollector(ABC):
    """All collectors must implement collect() returning NormalizedDocuments."""

    @abstractmethod
    async def collect(self, limit: int | None = None) -> list[NormalizedDocument]:
        ...

    async def dry_run(self, limit: int = 5) -> list[NormalizedDocument]:
        """Preview what would be collected — prints metadata, does NOT store."""
        docs = await self.collect(limit=limit)
        for doc in docs:
            print(
                f"[{doc.source_platform}] {doc.title[:80]!r} "
                f"by {doc.author!r}  ({doc.char_count} chars)"
            )
        return docs

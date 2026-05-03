"""Retrieval pipeline for the ArchRAG system.

Wraps :class:`~archrag.vector_store.VectorStore` with higher-level logic:
source deduplication, context assembly, and context window budgeting.
The output of :meth:`Retriever.retrieve` is a structured
:class:`RetrievalResult` ready to be injected into an LLM prompt.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import RetrievalConfig
from .vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured output of a single retrieval operation.

    Attributes:
        query: The original user query string.
        chunks: Ordered list of retrieved and deduplicated chunks.
        context: Assembled context string ready for prompt injection.
        sources: Deduplicated list of source file paths that contributed
            at least one chunk to ``context``.
        total_words: Sum of word counts across all chunks in ``context``.
    """

    query: str
    chunks: list[SearchResult]
    context: str
    sources: list[str]
    total_words: int

    @property
    def is_empty(self) -> bool:
        """Returns ``True`` if no chunks passed the similarity threshold."""
        return len(self.chunks) == 0


class Retriever:
    """Orchestrates vector search, deduplication, and context assembly.

    Args:
        store: A populated :class:`~archrag.vector_store.VectorStore`.
        config: Retrieval parameters. Defaults to ``RetrievalConfig()``.

    Example::

        retriever = Retriever(store)
        result = retriever.retrieve("How do I remove orphaned packages?")
        print(result.context)
    """

    def __init__(
        self,
        store: VectorStore,
        config: RetrievalConfig = RetrievalConfig(),
    ) -> None:
        self._store = store
        self._cfg = config

    def retrieve(self, query: str) -> RetrievalResult:
        """Execute the full retrieval pipeline for a single query.

        Steps:
            1. Search the vector store for ``top_k`` similar chunks.
            2. Deduplicate by source (max ``max_chunks_per_source`` per file).
            3. Assemble chunks into a context string within the word budget.

        Args:
            query: Natural language question from the user.

        Returns:
            A :class:`RetrievalResult` containing the context string and
            supporting metadata. If no chunks exceed the similarity threshold,
            ``result.is_empty`` will be ``True``.
        """
        raw_results = self._store.search(
            query=query,
            top_k=self._cfg.top_k,
            min_similarity=self._cfg.min_similarity,
        )

        deduplicated = self._deduplicate(raw_results)
        context, used_chunks = self._assemble_context(deduplicated)

        unique_sources = list(dict.fromkeys(c.source for c in used_chunks))
        total_words = sum(len(c.text.split()) for c in used_chunks)

        logger.info(
            "Retrieved %d chunks (%d words) from %d sources for query: '%s'",
            len(used_chunks),
            total_words,
            len(unique_sources),
            query[:80],
        )

        return RetrievalResult(
            query=query,
            chunks=used_chunks,
            context=context,
            sources=unique_sources,
            total_words=total_words,
        )

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Limit contributions from any single source file.

        Without deduplication, a highly relevant document can monopolise
        the context window, crowding out other useful sources.

        Args:
            results: Similarity-sorted search results.

        Returns:
            Filtered list with at most ``max_chunks_per_source`` entries
            per unique source path.
        """
        source_counts: dict[str, int] = {}
        filtered: list[SearchResult] = []

        for result in results:
            count = source_counts.get(result.source, 0)
            if count < self._cfg.max_chunks_per_source:
                filtered.append(result)
                source_counts[result.source] = count + 1
            else:
                logger.debug(
                    "Deduplication: skipping extra chunk from %s",
                    Path(result.source).name,
                )

        return filtered

    def _assemble_context(
        self, chunks: list[SearchResult]
    ) -> tuple[str, list[SearchResult]]:
        """Build the context string injected into the LLM prompt.

        Concatenates chunks with clear source headers and a separator.
        Stops adding chunks once the word count reaches ``max_context_words``
        to stay within the LLM's context window budget.

        Args:
            chunks: Deduplicated, similarity-sorted search results.

        Returns:
            A tuple of ``(context_string, included_chunks)`` where
            ``included_chunks`` is the subset of ``chunks`` that fit within
            the word budget.
        """
        parts: list[str] = []
        word_count = 0
        included: list[SearchResult] = []

        for i, chunk in enumerate(chunks):
            source_label = Path(chunk.source).stem
            header = (
                f"[Source {i + 1}: {source_label} | "
                f"similarity: {chunk.similarity:.2f}]"
            )
            chunk_words = len(chunk.text.split())
            remaining_budget = self._cfg.max_context_words - word_count

            if remaining_budget <= 0:
                logger.debug("Context budget exhausted at chunk %d", i)
                break

            if chunk_words > remaining_budget:
                truncated = " ".join(chunk.text.split()[:remaining_budget])
                parts.append(f"{header}\n{truncated} [truncated]")
                word_count += remaining_budget
                included.append(chunk)
                break

            parts.append(f"{header}\n{chunk.text}")
            word_count += chunk_words
            included.append(chunk)

        context = "\n\n---\n\n".join(parts)
        return context, included

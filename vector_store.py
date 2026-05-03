"""Vector store management for the ArchRAG pipeline.

Handles embedding generation via ``sentence-transformers`` and persistence
via ChromaDB. Exposes a clean interface for indexing chunks and querying
by semantic similarity.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig
from .ingestor import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single result returned by :meth:`VectorStore.search`.

    Attributes:
        chunk_id: Identifier of the matched chunk.
        text: Raw text content of the chunk.
        source: Path to the originating document.
        chunk_index: Position of the chunk within its document.
        similarity: Cosine similarity score in the range ``[0, 1]``.
    """

    chunk_id: str
    text: str
    source: str
    chunk_index: int
    similarity: float


class VectorStore:
    """Manages embedding generation and ChromaDB persistence.

    The store is lazy — the embedding model is not loaded until the first
    call to :meth:`index` or :meth:`search`, avoiding import-time overhead.

    Args:
        config: Embedding and storage parameters. Defaults to
            ``EmbeddingConfig()``.

    Example::

        store = VectorStore()
        store.index(chunks)
        results = store.search("how do I remove orphaned packages", top_k=5)
    """

    def __init__(self, config: EmbeddingConfig = EmbeddingConfig()) -> None:
        self._cfg = config
        self._model: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-loaded embedding model.

        Downloads the model on first access (~80 MB) then caches it in
        memory for the lifetime of the ``VectorStore`` instance.
        """
        if self._model is None:
            logger.info("Loading embedding model: %s", self._cfg.model_name)
            self._model = SentenceTransformer(self._cfg.model_name)
        return self._model

    @property
    def collection(self) -> chromadb.Collection:
        """Lazy-loaded ChromaDB collection.

        Creates the persistent client and collection on first access.
        Uses cosine similarity space so that L2-normalised dot products
        and cosine distances are equivalent.
        """
        if self._collection is None:
            self._cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._cfg.chroma_dir))
            self._collection = self._client.get_or_create_collection(
                name=self._cfg.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug(
                "Opened ChromaDB collection '%s' at %s (%d vectors)",
                self._cfg.collection_name,
                self._cfg.chroma_dir,
                self._collection.count(),
            )
        return self._collection

    def index(self, chunks: list[Chunk], force: bool = False) -> None:
        """Embed a list of chunks and persist them to ChromaDB.

        Skips re-embedding if the collection is already populated and
        ``force`` is ``False``.

        Args:
            chunks: Chunks produced by :class:`~archrag.ingestor.Ingestor`.
            force: When ``True``, drops the existing collection and
                rebuilds from scratch.

        Raises:
            ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        existing = self.collection.count()

        if existing > 0 and not force:
            logger.info(
                "Collection already contains %d vectors — skipping re-index. "
                "Pass force=True to rebuild.",
                existing,
            )
            return

        if force and existing > 0:
            logger.info("Force re-index requested — dropping %d existing vectors.", existing)
            self._client.delete_collection(self._cfg.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._cfg.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        logger.info("Embedding %d chunks in batches of %d …", len(chunks), self._cfg.batch_size)

        for batch_start in range(0, len(chunks), self._cfg.batch_size):
            batch = chunks[batch_start : batch_start + self._cfg.batch_size]

            embeddings = self.model.encode(
                [c.text for c in batch],
                batch_size=self._cfg.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self._cfg.normalize,
            ).tolist()

            self.collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                        "total_chunks": c.total_chunks,
                        "word_count": c.word_count,
                    }
                    for c in batch
                ],
            )

            indexed_so_far = min(batch_start + self._cfg.batch_size, len(chunks))
            logger.debug("Indexed %d / %d chunks", indexed_so_far, len(chunks))

        logger.info(
            "Indexing complete — %d vectors stored in %s",
            self.collection.count(),
            self._cfg.chroma_dir,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.25,
    ) -> list[SearchResult]:
        """Retrieve the most semantically similar chunks for a query.

        Embeds ``query`` with the same model used during indexing, queries
        ChromaDB, and filters results below ``min_similarity``.

        Args:
            query: Natural language question or search string.
            top_k: Maximum number of results to return before filtering.
            min_similarity: Cosine similarity floor. Results below this
                value are discarded.

        Returns:
            List of :class:`SearchResult` objects ordered by descending
            similarity. May be shorter than ``top_k`` after filtering.

        Raises:
            RuntimeError: If the vector store is empty.
        """
        if self.collection.count() == 0:
            raise RuntimeError(
                "Vector store is empty. Call VectorStore.index() before searching."
            )

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=self._cfg.normalize,
        )[0].tolist()

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            similarity = 1.0 - dist
            if similarity < min_similarity:
                logger.debug(
                    "Discarding chunk (similarity %.3f < threshold %.3f)",
                    similarity,
                    min_similarity,
                )
                continue

            results.append(
                SearchResult(
                    chunk_id=f"{meta['source']}_chunk_{meta['chunk_index']}",
                    text=doc,
                    source=meta["source"],
                    chunk_index=meta["chunk_index"],
                    similarity=similarity,
                )
            )

        results.sort(key=lambda r: r.similarity, reverse=True)
        logger.debug("search('%s') → %d results above threshold", query[:60], len(results))
        return results

    @property
    def is_populated(self) -> bool:
        """Returns ``True`` if the vector store contains at least one vector."""
        return self.collection.count() > 0

    @property
    def vector_count(self) -> int:
        """Total number of vectors currently stored."""
        return self.collection.count()

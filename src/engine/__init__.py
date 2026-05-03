import logging
from typing import Iterator

from .base import BaseEngine
from .cloud_llm import CloudLLMEngine
from .local_llm import LocalLLMEngine

logger = logging.getLogger(__name__)


class HybridEngine(BaseEngine):
    name = "hybrid"

    def __init__(self, config) -> None:
        self._cfg = config
        self.local = LocalLLMEngine(config.local_llm)
        self.cloud = CloudLLMEngine(config.cloud_llm)
        self._preferred = config.preferred_backend

    @property
    def active_backend(self) -> str:
        return self._preferred

    def set_backend(self, backend: str) -> None:
        if backend not in ("local", "cloud"):
            raise ValueError(f"Invalid backend '{backend}'. Choose 'local' or 'cloud'.")
        self._preferred = backend
        logger.info("Preferred backend set to '%s'.", backend)

    def is_available(self) -> bool:
        return self.local.is_available() or self.cloud.is_available()

    def generate(self, query: str, context: str) -> Iterator[str]:
        primary = self.cloud if self._preferred == "cloud" else self.local
        fallback = self.local if self._preferred == "cloud" else self.cloud

        if primary.is_available():
            logger.info("Generating with %s backend.", primary.name)
            try:
                yield from primary.generate(query, context)
                return
            except Exception as exc:
                logger.error("%s backend failed: %s", primary.name, exc)

        if self._cfg.fallback_to_local and fallback.is_available():
            logger.warning("Falling back to %s.", fallback.name)
            yield f"\n\n⚠️ *{primary.name} unavailable — switched to {fallback.name}.*\n\n"
            yield from fallback.generate(query, context)
            return

        raise RuntimeError(
            "No inference backend is available. "
            "Check your GROQ_API_KEY is exported and you have internet."
        )


__all__ = ["BaseEngine", "LocalLLMEngine", "CloudLLMEngine", "HybridEngine"]
"""Engine sub-package for ArchRAG.

Exports the concrete engine classes and the :class:`HybridEngine` which
wraps them with automatic fallback logic.
"""

import logging
from typing import Iterator

from .base import BaseEngine
from .cloud_llm import CloudLLMEngine
from .local_llm import LocalLLMEngine
from ..config import AppConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class HybridEngine(BaseEngine):
    """Inference engine with automatic local/cloud fallback.

    On each ``generate`` call, the engine attempts to use the preferred
    backend. If that backend is unavailable and ``fallback_to_local`` is
    enabled, it transparently switches to the local backend and emits a
    warning rather than raising an exception.

    Args:
        config: Top-level application config. Reads ``preferred_backend``
            and ``fallback_to_local`` from this object.

    Example::

        engine = HybridEngine()
        engine.set_backend("cloud")
        for token in engine.generate(query, context):
            print(token, end="", flush=True)
    """

    name = "hybrid"

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config
        self.local = LocalLLMEngine(config.local_llm)
        self.cloud = CloudLLMEngine(config.cloud_llm)
        self._preferred: str = config.preferred_backend

    @property
    def active_backend(self) -> str:
        """The currently preferred backend: ``"local"`` or ``"cloud"``."""
        return self._preferred

    def set_backend(self, backend: str) -> None:
        """Switch the preferred backend.

        Args:
            backend: Either ``"local"`` or ``"cloud"``.

        Raises:
            ValueError: If ``backend`` is not a valid option.
        """
        if backend not in ("local", "cloud"):
            raise ValueError(f"Invalid backend '{backend}'. Choose 'local' or 'cloud'.")
        self._preferred = backend
        logger.info("Preferred backend set to '%s'.", backend)

    def is_available(self) -> bool:
        """Return ``True`` if at least one backend is available."""
        return self.local.is_available() or self.cloud.is_available()

    def generate(self, query: str, context: str) -> Iterator[str]:
        """Stream tokens using the preferred backend, falling back if needed.

        If the preferred backend fails and ``config.fallback_to_local`` is
        ``True``, the method yields a notice token informing the user of
        the switch before continuing with the fallback backend.

        Args:
            query: User's natural language question.
            context: Assembled retrieval context.

        Yields:
            String tokens from whichever backend ultimately handles the request.

        Raises:
            RuntimeError: If both backends are unavailable.
        """
        primary = self.cloud if self._preferred == "cloud" else self.local
        fallback = self.local if self._preferred == "cloud" else self.cloud

        if primary.is_available():
            logger.info("Generating with %s backend.", primary.name)
            try:
                yield from primary.generate(query, context)
                return
            except Exception as exc:
                logger.error(
                    "%s backend failed during generation: %s", primary.name, exc
                )

        if self._cfg.fallback_to_local and fallback.is_available():
            logger.warning(
                "%s backend failed — falling back to %s.",
                primary.name,
                fallback.name,
            )
            yield f"\n\n⚠️ *{primary.name} backend unavailable — switched to {fallback.name}.*\n\n"
            yield from fallback.generate(query, context)
            return

        raise RuntimeError(
            "No inference backend is available. "
            "Start Ollama (ollama serve) or set a cloud API key."
        )


__all__ = ["BaseEngine", "LocalLLMEngine", "CloudLLMEngine", "HybridEngine"]

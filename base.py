"""Abstract base class for ArchRAG inference engines.

All concrete engine implementations must inherit from :class:`BaseEngine`
and implement :meth:`generate`. This contract guarantees that the
application layer (``main.py``) can swap backends without any changes to
its own code.
"""

import logging
from abc import ABC, abstractmethod
from typing import Iterator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a knowledgeable assistant specialising in Arch Linux, Hyprland, \
and Linux system administration.

Answer questions based ONLY on the provided context passages. \
Each passage is tagged with its source file and similarity score.

Rules:
- If the context does not contain sufficient information, respond with: \
"I don't have enough information in my knowledge base to answer this confidently."
- Be specific and practical — include exact commands, file paths, and \
config snippets when they appear in the context.
- Never fabricate commands or file paths that are absent from the context.
- If multiple sources support an answer, cite both.
- Keep answers concise but complete.\
"""


def build_prompt(query: str, context: str) -> str:
    """Assemble the user-turn prompt sent to the LLM.

    Places retrieved context before the question so the model reads
    available evidence before formulating its answer.

    Args:
        query: The user's natural language question.
        context: Assembled retrieval context from :class:`~archrag.retriever.Retriever`.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Here are the relevant passages from the knowledge base:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Based on the above passages, please answer this question:\n{query}"
    )


class BaseEngine(ABC):
    """Contract that all LLM inference backends must satisfy.

    Concrete subclasses implement :meth:`generate` as a Python generator
    that ``yield``s string tokens. Streaming via generators means the
    Gradio UI can display tokens as they arrive without blocking.

    Attributes:
        name: Human-readable backend identifier used in logging and UI labels.
    """

    name: str = "base"

    @abstractmethod
    def generate(self, query: str, context: str) -> Iterator[str]:
        """Stream an answer token by token.

        Args:
            query: The user's natural language question.
            context: Assembled retrieval context to inject into the prompt.

        Yields:
            String tokens as they are produced by the underlying model.
            The caller accumulates tokens to build the complete response.

        Raises:
            NotImplementedError: If a subclass does not override this method.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the backend is reachable and ready.

        Used by the hybrid engine to decide whether to fall back to the
        local backend when the preferred backend is unavailable.

        Returns:
            Availability boolean.
        """
        raise NotImplementedError

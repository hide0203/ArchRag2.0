"""Local inference engine backed by Ollama.

Communicates with the Ollama REST API over localhost. Streams tokens by
consuming the newline-delimited JSON response from ``/api/generate``.
Requires Ollama to be running (``ollama serve``) and the target model to
be pulled (``ollama pull mistral``).
"""

import json
import logging
from typing import Iterator

import requests

from ..config import LocalLLMConfig
from .base import BaseEngine, build_prompt, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LocalLLMEngine(BaseEngine):
    """Ollama-backed streaming inference engine.

    Uses the Ollama ``/api/generate`` endpoint with ``stream=true``.
    Each line of the response is a JSON object containing a ``response``
    token field and a ``done`` boolean.

    Args:
        config: Ollama connection and sampling parameters.

    Example::

        engine = LocalLLMEngine()
        if engine.is_available():
            for token in engine.generate(query, context):
                print(token, end="", flush=True)
    """

    name = "local"

    def __init__(self, config: LocalLLMConfig = LocalLLMConfig()) -> None:
        self._cfg = config

    def is_available(self) -> bool:
        """Return ``True`` if the Ollama server is reachable.

        Performs a lightweight ``GET /api/tags`` request with a short
        timeout to avoid blocking the application startup.

        Returns:
            ``True`` when Ollama responds with HTTP 200.
        """
        try:
            response = requests.get(
                f"{self._cfg.base_url}/api/tags",
                timeout=3,
            )
            available = response.status_code == 200
            if not available:
                logger.warning(
                    "Ollama returned unexpected status %d", response.status_code
                )
            return available
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Ollama is not reachable at %s. Start it with: ollama serve",
                self._cfg.base_url,
            )
            return False

    def list_models(self) -> list[str]:
        """Return the names of models currently pulled in Ollama.

        Returns:
            List of model name strings (e.g. ``["mistral:latest"]``).
            Returns an empty list if Ollama is not reachable.
        """
        try:
            response = requests.get(
                f"{self._cfg.base_url}/api/tags",
                timeout=5,
            )
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.error("Failed to list Ollama models: %s", exc)
            return []

    def ensure_model(self) -> bool:
        """Pull the configured model if it is not already present.

        Returns:
            ``True`` if the model is ready to use after this call.
        """
        models = self.list_models()
        model_base = self._cfg.model.split(":")[0]

        if any(model_base in m for m in models):
            logger.info("Model '%s' is already available locally.", self._cfg.model)
            return True

        logger.info(
            "Model '%s' not found locally — pulling from Ollama registry …",
            self._cfg.model,
        )
        return self._pull_model()

    def generate(self, query: str, context: str) -> Iterator[str]:
        """Stream a response from the local Ollama model.

        Sends a POST request to ``/api/generate`` with ``stream=true`` and
        yields individual tokens as they arrive.

        Args:
            query: The user's natural language question.
            context: Assembled retrieval context.

        Yields:
            Individual string tokens from the model's output.

        Raises:
            requests.exceptions.RequestException: On network or HTTP errors.
        """
        prompt = build_prompt(query, context)
        payload = {
            "model": self._cfg.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": True,
            "options": {
                "temperature": self._cfg.temperature,
                "top_p": self._cfg.top_p,
                "num_ctx": self._cfg.num_ctx,
            },
        }

        logger.debug(
            "Sending request to Ollama: model=%s, prompt_words=%d",
            self._cfg.model,
            len(prompt.split()),
        )

        with requests.post(
            f"{self._cfg.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self._cfg.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done", False):
                    logger.debug("Ollama stream completed.")
                    break

    def _pull_model(self) -> bool:
        """Download the configured model from the Ollama registry.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            response = requests.post(
                f"{self._cfg.base_url}/api/pull",
                json={"name": self._cfg.model},
                stream=True,
                timeout=600,
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get("status") == "success":
                        logger.info("Model '%s' pulled successfully.", self._cfg.model)
                        return True
            return True
        except Exception as exc:
            logger.error("Failed to pull model '%s': %s", self._cfg.model, exc)
            return False

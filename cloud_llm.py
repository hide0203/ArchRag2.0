"""Cloud inference engine supporting Anthropic, OpenAI, and Groq.

Routes requests to the configured provider using that provider's native
streaming API. Each provider implementation is a private method that
yields tokens, keeping the public ``generate`` interface uniform.

Supported providers
-------------------
``anthropic``
    Uses the Anthropic Messages API (``POST /v1/messages``) with
    ``stream=true``. Requires ``ANTHROPIC_API_KEY``.

``openai``
    Uses the OpenAI Chat Completions API (``POST /v1/chat/completions``)
    with ``stream=true``. Requires ``OPENAI_API_KEY``. Also compatible
    with any OpenAI-compatible endpoint (e.g. Together AI, Anyscale).

``groq``
    Uses the Groq Chat Completions API, which is OpenAI-compatible.
    Requires ``GROQ_API_KEY``. Groq is recommended for fast, free-tier
    cloud inference when the local GPU is unavailable.
"""

import logging
import os
from typing import Iterator

import requests

from ..config import CloudLLMConfig
from .base import BaseEngine, build_prompt, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_PROVIDER_BASE_URLS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "groq": "https://api.groq.com/openai",
}


class CloudLLMEngine(BaseEngine):
    """Multi-provider cloud streaming inference engine.

    Selects the appropriate provider implementation at runtime based on
    ``config.provider``. API keys are read from environment variables —
    never hard-coded.

    Args:
        config: Provider, model, and sampling parameters.

    Example::

        engine = CloudLLMEngine()
        if engine.is_available():
            for token in engine.generate(query, context):
                print(token, end="", flush=True)
    """

    name = "cloud"

    def __init__(self, config: CloudLLMConfig = CloudLLMConfig()) -> None:
        self._cfg = config
        self._base_url = (
            config.base_url or _PROVIDER_BASE_URLS.get(config.provider, "")
        )

    @property
    def _api_key(self) -> str | None:
        """Read the API key from the configured environment variable."""
        return os.environ.get(self._cfg.api_key_env)

    def is_available(self) -> bool:
        """Return ``True`` if the provider API key is set and the endpoint responds.

        Performs a lightweight connectivity check rather than a full
        inference request to avoid unnecessary token charges.

        Returns:
            ``True`` when the key is present and the provider base URL
            is reachable.
        """
        if not self._api_key:
            logger.warning(
                "Cloud backend unavailable: environment variable '%s' is not set.",
                self._cfg.api_key_env,
            )
            return False

        try:
            response = requests.head(self._base_url, timeout=4)
            return response.status_code < 500
        except requests.exceptions.RequestException as exc:
            logger.warning("Cloud provider connectivity check failed: %s", exc)
            return False

    def generate(self, query: str, context: str) -> Iterator[str]:
        """Stream a response from the configured cloud provider.

        Dispatches to the provider-specific implementation based on
        ``config.provider``.

        Args:
            query: The user's natural language question.
            context: Assembled retrieval context.

        Yields:
            Individual string tokens from the model's output.

        Raises:
            ValueError: If ``config.provider`` is not supported.
            requests.exceptions.RequestException: On network or HTTP errors.
        """
        provider = self._cfg.provider.lower()

        if provider == "anthropic":
            yield from self._generate_anthropic(query, context)
        elif provider in ("openai", "groq"):
            yield from self._generate_openai_compatible(query, context)
        else:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Choose from: anthropic, openai, groq."
            )

    def _generate_anthropic(self, query: str, context: str) -> Iterator[str]:
        """Stream tokens from the Anthropic Messages API.

        Parses the server-sent events (SSE) stream, extracting
        ``content_block_delta`` events that carry text tokens.

        Args:
            query: User question.
            context: Retrieved context string.

        Yields:
            Text token strings.
        """
        prompt = build_prompt(query, context)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self._cfg.model,
            "max_tokens": self._cfg.max_tokens,
            "system": SYSTEM_PROMPT,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }

        logger.debug(
            "Sending request to Anthropic: model=%s", self._cfg.model
        )

        with requests.post(
            f"{self._base_url}/v1/messages",
            headers=headers,
            json=payload,
            stream=True,
            timeout=self._cfg.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                if not decoded.startswith("data:"):
                    continue
                data_str = decoded[len("data:"):].strip()
                if data_str in ("", "[DONE]"):
                    continue
                try:
                    import json
                    data = json.loads(data_str)
                    if data.get("type") == "content_block_delta":
                        token = data.get("delta", {}).get("text", "")
                        if token:
                            yield token
                    elif data.get("type") == "message_stop":
                        logger.debug("Anthropic stream completed.")
                        break
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.debug("Skipping unparseable SSE line: %s", exc)

    def _generate_openai_compatible(self, query: str, context: str) -> Iterator[str]:
        """Stream tokens from an OpenAI-compatible chat completions endpoint.

        Compatible with OpenAI, Groq, Together AI, and any provider that
        implements the OpenAI Chat Completions streaming format.

        Args:
            query: User question.
            context: Retrieved context string.

        Yields:
            Text token strings.
        """
        import json

        prompt = build_prompt(query, context)

        if self._cfg.provider == "groq":
            base = "https://api.groq.com/openai"
            key_header = "Authorization"
            key_value = f"Bearer {self._api_key}"
        else:
            base = self._base_url
            key_header = "Authorization"
            key_value = f"Bearer {self._api_key}"

        headers = {
            key_header: key_value,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._cfg.model,
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }

        logger.debug(
            "Sending request to %s: model=%s", self._cfg.provider, self._cfg.model
        )

        with requests.post(
            f"{base}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=self._cfg.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                if not decoded.startswith("data:"):
                    continue
                data_str = decoded[len("data:"):].strip()
                if data_str == "[DONE]":
                    logger.debug("%s stream completed.", self._cfg.provider)
                    break
                try:
                    data = json.loads(data_str)
                    token = (
                        data.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if token:
                        yield token
                except (json.JSONDecodeError, IndexError, KeyError) as exc:
                    logger.debug("Skipping unparseable SSE line: %s", exc)

"""Provider-neutral LLM interface for the research layer (ROADMAP M9).

The workflow code depends only on :class:`CompletionProvider`; concrete
providers are:

- :class:`MockProvider` — deterministic, offline, free. The default for
  tests and demos; scripted responses or a stable digest-echo fallback.
- :class:`AnthropicProvider` — real calls through the official
  ``anthropic`` SDK (optional ``research`` extra). The API key is read
  from ``ANTHROPIC_API_KEY`` at call-site construction and is never
  persisted by this package; keys must never be committed to the
  repository (ARCHITECTURE §8).

Every provider declares ``is_paid``; workflows refuse to call a paid
provider without an explicit human approval record — the paid-services
gate lives above this module, not inside it.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol, runtime_checkable

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"


@runtime_checkable
class CompletionProvider(Protocol):
    """The only LLM surface the research layer knows about."""

    @property
    def name(self) -> str:
        """Stable provider identifier, used in entity provenance strings."""
        ...

    @property
    def is_paid(self) -> bool:
        """True if calls cost money (triggers the paid-services gate)."""
        ...

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        """Return the model's text completion for ``prompt``."""
        ...


class MockProvider:
    """Deterministic offline provider for tests and demos.

    With ``responses``, replies are consumed in order (raising when
    exhausted — a test that over-calls should fail loudly). Without, the
    reply is a stable digest-echo of the prompt: identical prompts always
    produce identical output, so downstream artifacts are reproducible.
    """

    def __init__(self, responses: tuple[str, ...] | None = None) -> None:
        """Create the provider, optionally with scripted responses."""
        self._responses = list(responses) if responses is not None else None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "mock"

    @property
    def is_paid(self) -> bool:
        """MockProvider is free."""
        return False

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        """Scripted response, or a deterministic digest-echo of the prompt.

        Raises:
            RuntimeError: If scripted responses are exhausted.
        """
        if self._responses is not None:
            if not self._responses:
                raise RuntimeError("MockProvider has no scripted responses left.")
            return self._responses.pop(0)
        digest = hashlib.sha256((system + "\n" + prompt).encode()).hexdigest()[:8]
        return f"[mock:{digest}] {prompt[:200]}"


class AnthropicProvider:
    """Claude-backed provider via the official ``anthropic`` SDK.

    The key comes from ``ANTHROPIC_API_KEY`` (or the ``api_key``
    argument, for tests only) and is held in memory — never written to
    the store, logs, or reports.
    """

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, api_key: str | None = None) -> None:
        """Configure the provider.

        Raises:
            ValueError: If no API key is available — the error names the
                environment variable rather than prompting for a secret.
        """
        self._model = model
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "AnthropicProvider requires an API key: set the ANTHROPIC_API_KEY "
                "environment variable (keys are never stored by rlcore)."
            )
        self._api_key = key

    @property
    def name(self) -> str:
        """Provider identifier including the model."""
        return f"anthropic:{self._model}"

    @property
    def is_paid(self) -> bool:
        """Anthropic API calls are a paid service."""
        return True

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        """One Messages API call; returns the concatenated text blocks.

        Raises:
            ImportError: If the ``anthropic`` SDK is not installed
                (install the ``research`` extra).
        """
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "AnthropicProvider requires the anthropic SDK; install the "
                "'research' extra (uv sync --extra research)."
            ) from exc
        client = anthropic.Anthropic(api_key=self._api_key)
        if system:
            message = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            message = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        return "".join(block.text for block in message.content if block.type == "text")

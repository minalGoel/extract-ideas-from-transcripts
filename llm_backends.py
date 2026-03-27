"""
Swappable LLM backends — all expose the same interface:

    backend.complete(system: str, user: str) -> str

Pick one and pass it to the extractor. Configure via env vars or kwargs.

──────────────────────────────────────────────────────────────────────────────
BACKEND 1 — Direct Anthropic API key
──────────────────────────────────────────────────────────────────────────────
Required env var:
    ANTHROPIC_API_KEY=sk-ant-...

Usage:
    from llm_backends import AnthropicBackend
    backend = AnthropicBackend()                          # reads from env
    backend = AnthropicBackend(api_key="sk-ant-...",      # or pass directly
                               model="claude-opus-4-6")

──────────────────────────────────────────────────────────────────────────────
BACKEND 2 — AWS Bedrock
──────────────────────────────────────────────────────────────────────────────
Required env vars (or IAM role attached to the machine):
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    AWS_DEFAULT_REGION=us-east-1   (or whichever region has Claude enabled)

Usage:
    from llm_backends import BedrockBackend
    backend = BedrockBackend()
    backend = BedrockBackend(
        model="anthropic.claude-opus-4-5-20251101-v1:0",
        region="us-west-2",
    )

Bedrock model IDs (check AWS console for latest):
    anthropic.claude-opus-4-5-20251101-v1:0
    anthropic.claude-sonnet-4-5-20251101-v1:0

──────────────────────────────────────────────────────────────────────────────
BACKEND 3 — Azure AI Foundry
──────────────────────────────────────────────────────────────────────────────
Required env vars:
    AZURE_API_KEY=...
    AZURE_ENDPOINT=https://<your-resource>.services.ai.azure.com/models
    (optionally) AZURE_MODEL=claude-opus-4-5  (name of your deployed model)

Usage:
    from llm_backends import AzureBackend
    backend = AzureBackend()
    backend = AzureBackend(
        endpoint="https://...",
        api_key="...",
        model="claude-opus-4-5",
    )
"""

from __future__ import annotations

import os


# ──────────────────────────────────────────────────────────────────────────────

class AnthropicBackend:
    """Direct Anthropic API — the simplest option."""

    DEFAULT_MODEL = "claude-opus-4-6"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        import anthropic
        self.model      = model or os.getenv("ANTHROPIC_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self._client    = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


# ──────────────────────────────────────────────────────────────────────────────

class BedrockBackend:
    """AWS Bedrock — uses boto3 credentials (key pair or IAM role)."""

    DEFAULT_MODEL  = "anthropic.claude-opus-4-5-20251101-v1:0"
    DEFAULT_REGION = "us-east-1"

    def __init__(
        self,
        model: str | None = None,
        region: str | None = None,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        import anthropic
        self.model      = model or os.getenv("BEDROCK_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self._client    = anthropic.AnthropicBedrock(
            aws_access_key=aws_access_key or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=aws_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=region or os.getenv("AWS_DEFAULT_REGION", self.DEFAULT_REGION),
        )

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


# ──────────────────────────────────────────────────────────────────────────────

class AzureBackend:
    """
    Azure AI Foundry — Claude deployed on Azure.

    Azure AI Foundry exposes a Messages-compatible endpoint; we point the
    Anthropic SDK at it via base_url.  The api_key is sent both as the SDK
    auth header and as the "api-key" header that Azure requires.
    """

    DEFAULT_MODEL = "claude-opus-4-5"

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        import anthropic
        _endpoint = endpoint or os.environ["AZURE_ENDPOINT"]
        _key      = api_key  or os.environ["AZURE_API_KEY"]
        self.model      = model or os.getenv("AZURE_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self._client    = anthropic.Anthropic(
            api_key=_key,
            base_url=_endpoint,
            default_headers={"api-key": _key},
        )

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

import asyncio

import httpx
import pytest

from rag_agent.integrations.ollama.client import OllamaClient
from rag_agent.integrations.ollama.exceptions import OllamaTimeoutError, OllamaUnavailableError


class ConnectErrorAsyncClient:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "ConnectErrorAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    async def post(self, url: str, json: dict[str, object]) -> None:
        request = httpx.Request("POST", url, json=json)
        raise httpx.ConnectError("boom", request=request)


class TimeoutAsyncClient:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "TimeoutAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    async def post(self, url: str, json: dict[str, object]) -> None:
        request = httpx.Request("POST", url, json=json)
        raise httpx.ReadTimeout("boom", request=request)


def test_generate_maps_connect_error_to_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", ConnectErrorAsyncClient)
    client = OllamaClient("http://localhost:11434")

    with pytest.raises(OllamaUnavailableError, match="Failed to connect to Ollama"):
        asyncio.run(client.generate("llama3.2:1b", "hello"))


def test_embed_maps_timeout_to_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", TimeoutAsyncClient)
    client = OllamaClient("http://localhost:11434")

    with pytest.raises(OllamaTimeoutError, match="timed out"):
        asyncio.run(client.embed("nomic-embed-text", "hello"))

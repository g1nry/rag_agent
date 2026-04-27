from typing import Any

import httpx

from rag_agent.integrations.ollama.exceptions import (
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def generate(self, model: str, prompt: str) -> str:
        data = await self._post_json(
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
        )
        response_text = data.get("response")
        if not isinstance(response_text, str):
            raise OllamaResponseError("Ollama response is missing generated text.")
        return response_text

    async def embed(self, model: str, text: str) -> list[float]:
        data = await self._post_json(
            "/api/embed",
            {
                "model": model,
                "input": text,
            },
        )
        embeddings = data.get("embeddings", [])
        if not isinstance(embeddings, list) or not embeddings:
            raise OllamaResponseError("Ollama response is missing embeddings.")

        first_embedding = embeddings[0]
        if not isinstance(first_embedding, list):
            raise OllamaResponseError("Ollama returned embeddings in an unexpected format.")
        return first_embedding

    async def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama request timed out while calling {url}."
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                f"Failed to connect to Ollama at {self._base_url}."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaResponseError(
                _build_status_error_message(exc.response),
                upstream_status_code=exc.response.status_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                f"Failed to communicate with Ollama at {self._base_url}."
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError("Ollama returned invalid JSON.") from exc

        if not isinstance(data, dict):
            raise OllamaResponseError("Ollama returned JSON in an unexpected format.")
        return data


def _build_status_error_message(response: httpx.Response) -> str:
    message = ""

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        raw_message = payload.get("error") or payload.get("message")
        if isinstance(raw_message, str):
            message = raw_message.strip()

    if not message:
        message = response.text.strip() or "No additional details provided."

    return f"Ollama request failed with status {response.status_code}: {message}"

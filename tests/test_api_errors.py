import asyncio
import json

from rag_agent.integrations.ollama.exceptions import OllamaUnavailableError
from rag_agent.main import handle_ollama_error


def test_ollama_exception_handler_returns_structured_json() -> None:
    response = asyncio.run(
        handle_ollama_error(
            None,  # type: ignore[arg-type]
            OllamaUnavailableError("Failed to connect to Ollama at http://localhost:11434."),
        )
    )

    assert response.status_code == 503
    assert json.loads(response.body) == {
        "detail": "Failed to connect to Ollama at http://localhost:11434.",
        "error_code": "ollama_unavailable",
    }

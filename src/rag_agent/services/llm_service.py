from rag_agent.core.config import Settings
from rag_agent.integrations.ollama.client import OllamaClient


class LLMService:
    def __init__(self, client: OllamaClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def generate(self, prompt: str) -> str:
        return await self._client.generate(self._settings.ollama_chat_model, prompt)

    async def embed(self, text: str) -> list[float]:
        return await self._client.embed(self._settings.ollama_embedding_model, text)


import httpx
from typing import List
from rag_agent.core.config import get_settings


class OllamaClient:
    def __init__(self, base_url: str = None):
        settings = get_settings()
        self.base_url = base_url or str(settings.ollama_base_url)
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(self, model: str, prompt: str) -> str:
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json()["response"]

    async def embed(self, model: str, text: str) -> List[float]:
        response = await self.client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def close(self):
        await self.client.aclose()
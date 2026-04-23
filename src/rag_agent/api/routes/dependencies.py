from functools import lru_cache

from rag_agent.agent.orchestrator import AgentOrchestrator
from rag_agent.core.config import Settings, get_settings
from rag_agent.integrations.ollama.client import OllamaClient
from rag_agent.services.llm_service import LLMService
from rag_agent.services.rag_service import RAGService
from rag_agent.storage.document_store import DocumentStore
from rag_agent.storage.vector_store import JsonVectorStore


@lru_cache
def get_document_store() -> DocumentStore:
    settings = get_settings()
    return DocumentStore(settings.documents_dir)


@lru_cache
def get_vector_store() -> JsonVectorStore:
    settings = get_settings()
    return JsonVectorStore(settings.index_path)


@lru_cache
def get_ollama_client() -> OllamaClient:
    settings = get_settings()
    return OllamaClient(base_url=str(settings.ollama_base_url))


@lru_cache
def get_llm_service() -> LLMService:
    settings = get_settings()
    return LLMService(get_ollama_client(), settings)


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService(
        settings=get_settings(),
        document_store=get_document_store(),
        vector_store=get_vector_store(),
        llm_service=get_llm_service(),
    )


@lru_cache
def get_agent_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_service=get_llm_service(),
        rag_service=get_rag_service(),
    )


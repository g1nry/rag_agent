from typing import Optional
import logging
import re

from rag_agent.core.config import get_settings
from rag_agent.core.config import Settings
from rag_agent.domain.schemas import RetrievalResult
from rag_agent.storage.vector_store import VectorStore, VectorRecord
from rag_agent.services.llm_service import LLMService
from rag_agent.rag.retriever import cosine_similarity
from rag_agent.integrations.ollama.client import OllamaClient

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        llm_service: LLMService,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._llm_service = llm_service

    async def search(self, payload) -> dict:
        result = await self.retrieve_context(payload.query, payload.top_k)
        contexts = result.contexts if hasattr(result, "contexts") else []
        return {"contexts": contexts}

    def _extract_exact_ids(self, query: str) -> list[str]:
        """Извлекает точные идентификаторы MITRE ATLAS из запроса пользователя.

        Покрывает форматы вида AML.T0070, AML.T0069.002, AML.CS0031, AML.TA0007.
        """
        matches = re.findall(
            r"\bAML\.(?:T|TA|CS)\d{4}(?:\.\d{3})?\b",
            query.upper(),
        )
        return list(dict.fromkeys(matches))

    def _exact_id_contexts(self, query: str, records: list, limit: int) -> list[dict]:
        """Точный поиск по техническим ID.

        Vector search плохо ловит точные идентификаторы, поэтому точные
        совпадения собираем отдельно и ставим перед семантическими.
        """
        exact_ids = self._extract_exact_ids(query)
        if not exact_ids:
            return []

        def exact_rank(record) -> tuple:
            text_upper = record.text.upper()
            # Лучшее совпадение — собственно поле ID чанка.
            for exact_id in exact_ids:
                if f"ID: {exact_id}" in text_upper or f"**ID:** `{exact_id}`" in text_upper:
                    return (0, record.filename, record.chunk_index)
            # Хорошее совпадение — заголовок техники/кейса.
            for exact_id in exact_ids:
                if f"TECHNIQUE: {exact_id}" in text_upper or f"CASE STUDY: {exact_id}" in text_upper:
                    return (1, record.filename, record.chunk_index)
            # Слабое совпадение — любое упоминание/ссылка.
            return (2, record.filename, record.chunk_index)

        matching = [
            record
            for record in records
            if any(exact_id in record.text.upper() for exact_id in exact_ids)
        ]
        matching.sort(key=exact_rank)

        contexts: list[dict] = []
        seen: set = set()
        for record in matching:
            if len(contexts) >= limit:
                break
            if record.chunk_id in seen:
                continue
            contexts.append({
                "text": record.text,
                "source": record.metadata.get("filename", record.filename),
                "chunk_id": record.chunk_id,
                "score": 1.0,
            })
            seen.add(record.chunk_id)
        return contexts

    async def retrieve_context(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        try:
            query_embedding = await self._llm_service.embed(query)
            records = self._vector_store.load()

            if not records:
                return RetrievalResult(contexts=[])

            scored = []
            for record in records:
                try:
                    score = cosine_similarity(query_embedding, record.embedding)
                    scored.append((score, record))
                except Exception:
                    continue

            scored.sort(key=lambda x: x[0], reverse=True)

            limit = top_k or self._settings.default_top_k

            # Точные ID идут перед семантическими результатами.
            contexts = self._exact_id_contexts(query, records, limit)
            seen_chunk_ids = {c["chunk_id"] for c in contexts}

            for score, record in scored:
                if len(contexts) >= limit:
                    break

                if record.chunk_id in seen_chunk_ids:
                    continue

                if score < self._settings.min_retrieval_score:
                    continue

                contexts.append({
                    "text": record.text,
                    "source": record.metadata.get("filename", record.filename),
                    "chunk_id": record.chunk_id,
                    "score": round(score, 4),
                })
                seen_chunk_ids.add(record.chunk_id)

            if not contexts:
                query_lower = query.lower()
                requested_filename = None
                for record in records:
                    filename = record.metadata.get("filename", record.filename)
                    if filename and filename.lower() in query_lower:
                        requested_filename = filename
                        break

                if requested_filename:
                    file_records = [
                        record for record in records
                        if record.metadata.get("filename", record.filename).lower() == requested_filename.lower()
                    ]
                    file_records.sort(key=lambda record: record.chunk_index)
                    for record in file_records[:limit]:
                        contexts.append({
                            "text": record.text,
                            "source": requested_filename,
                            "chunk_id": record.chunk_id,
                            "score": 0.0,
                        })

            return RetrievalResult(contexts=contexts)

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return RetrievalResult(contexts=[])


_settings = get_settings()

# Создаём зависимости
_ollama_client = OllamaClient(str(_settings.ollama_base_url))
_llm_service = LLMService(client=_ollama_client, settings=_settings)
_vector_store = VectorStore(_settings.index_path)

retrieval_service = RetrievalService(
    settings=_settings,
    vector_store=_vector_store,
    llm_service=_llm_service,
)

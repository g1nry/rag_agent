from rag_agent.domain.schemas import ChatRequest, ChatResponse, ContextItem
from rag_agent.services.llm_service import LLMService
from rag_agent.services.rag_service import RAGService


class AgentOrchestrator:
    def __init__(self, llm_service: LLMService, rag_service: RAGService) -> None:
        self._llm_service = llm_service
        self._rag_service = rag_service

    async def reply(self, payload: ChatRequest) -> ChatResponse:
        contexts: list[ContextItem] = []
        if payload.use_rag:
            retrieval = await self._rag_service.retrieve_context(payload.message, payload.top_k)
            contexts = retrieval.contexts

        prompt = self._build_prompt(payload.message, contexts)
        answer = await self._llm_service.generate(prompt)
        return ChatResponse(answer=answer, contexts=contexts)

    @staticmethod
    def _build_prompt(message: str, contexts: list[ContextItem]) -> str:
        if not contexts:
            return message

        joined_context = "\n\n".join(
            f"[source: {item.source}, chunk: {item.chunk_id}]\n{item.text}"
            for item in contexts
        )

        return (
            "Используй релевантный контекст только если он действительно помогает ответить на вопрос. "
            "Если контекст не подходит, ответь по смыслу вопроса без выдуманных фактов.\n\n"
            f"Контекст:\n{joined_context}\n\n"
            f"Вопрос пользователя:\n{message}"
        )


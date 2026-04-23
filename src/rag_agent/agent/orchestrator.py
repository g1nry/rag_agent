from rag_agent.domain.schemas import ChatRequest, ChatResponse
from rag_agent.services.llm_service import LLMService
from rag_agent.services.rag_service import RAGService


class AgentOrchestrator:
    def __init__(self, llm_service: LLMService, rag_service: RAGService) -> None:
        self._llm_service = llm_service
        self._rag_service = rag_service

    async def reply(self, payload: ChatRequest) -> ChatResponse:
        contexts = []
        if payload.use_rag:
            retrieval = await self._rag_service.retrieve_context(payload.message, payload.top_k)
            contexts = retrieval.contexts

        prompt = self._build_prompt(payload.message, contexts)
        answer = await self._llm_service.generate(prompt)
        return ChatResponse(answer=answer, contexts=contexts)

    @staticmethod
    def _build_prompt(message: str, contexts: list[str]) -> str:
        if not contexts:
            return message

        joined_context = "\n\n".join(f"- {item}" for item in contexts)
        return (
            "Используй только релевантный контекст, если он помогает ответить точнее.\n\n"
            f"Контекст:\n{joined_context}\n\n"
            f"Вопрос пользователя:\n{message}"
        )


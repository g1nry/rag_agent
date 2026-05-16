from .base import BaseTool, ToolMetadata
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class RAGTool(BaseTool):
    """Инструмент для поиска в RAG (оборачивает существующий RetrievalService)"""
    
    def __init__(self, retrieval_service):
        self.retrieval_service = retrieval_service
        self.metadata = ToolMetadata(
            name="rag_search",
            description="Поиск релевантной информации в загруженных документах. Используй когда нужно точные факты из базы знаний.",
            risk_level="safe",
            requires_confirmation=False,
            category="knowledge"
        )

    async def arun(self, query: str, **kwargs) -> str:
        """Выполняет поиск через существующий RAG сервис"""
        try:
            if not self.retrieval_service:
                return "RAG сервис не инициализирован."

            result = await self.retrieval_service.retrieve_context(query)
            
            if not result or not result.get("contexts"):
                return "Не удалось найти релевантную информацию по запросу."

            contexts = result["contexts"]
            formatted = "\n\n".join([f"Документ {i+1}:\n{c['content']}" for i, c in enumerate(contexts)])
            
            return f"Найдено {len(contexts)} релевантных фрагментов:\n\n{formatted}"
            
        except Exception as e:
            logger.error(f"Ошибка в RAGTool: {e}")
            return f"Ошибка при поиске в RAG: {str(e)}"


# Для удобства регистрации
def create_rag_tool(retrieval_service):
    return RAGTool(retrieval_service)
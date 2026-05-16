from typing import Dict, List
from .base import BaseTool
from .rag_tool import create_rag_tool
from .dangerous_tools import create_dangerous_tools
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Реестр всех доступных инструментов"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._initialized = False

    async def initialize(self, retrieval_service=None):
        """Инициализация всех инструментов"""
        if self._initialized:
            return
            
        # Safe tools
        if retrieval_service:
            self.tools["rag_search"] = create_rag_tool(retrieval_service)
            logger.info("✅ RAG Tool registered")
        
        # Dangerous tools
        dangerous = create_dangerous_tools()
        for tool in dangerous:
            self.tools[tool.metadata.name] = tool
            logger.info(f"⚠️  Dangerous tool registered: {tool.metadata.name} ({tool.metadata.risk_level})")
        
        self._initialized = True
        logger.info(f"Total tools registered: {len(self.tools)}")

    def get_tool(self, name: str) -> BaseTool | None:
        return self.tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        return list(self.tools.values())

    def get_tool_descriptions(self) -> str:
        """Возвращает описания инструментов для промпта LLM"""
        descriptions = []
        for tool in self.tools.values():
            meta = tool.get_metadata()
            risk = meta['risk_level']
            descriptions.append(
                f"- {meta['name']}: {meta['description']} [RISK: {risk.upper()}]"
            )
        return "\n".join(descriptions)


tool_registry = ToolRegistry()
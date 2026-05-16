from typing import Dict, List
from .base import BaseTool
from .rag_tool import create_rag_tool
from ..security.permission import permission_manager

class ToolRegistry:
    """Реестр всех доступных инструментов"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._initialized = False

    async def initialize(self, retrieval_service=None):
        """Инициализация всех инструментов"""
        if self._initialized:
            return
            
        # RAG Tool (safe)
        if retrieval_service:
            self.tools["rag_search"] = create_rag_tool(retrieval_service)
        
        # TODO: Здесь позже добавим safe_tools и dangerous_tools
        self._initialized = True

    def get_tool(self, name: str) -> BaseTool | None:
        return self.tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        return list(self.tools.values())

    def get_tool_descriptions(self) -> str:
        """Возвращает описания для промпта LLM"""
        descriptions = []
        for tool in self.tools.values():
            meta = tool.get_metadata()
            descriptions.append(
                f"- {meta['name']}: {meta['description']} (risk: {meta['risk_level']})"
            )
        return "\n".join(descriptions)


tool_registry = ToolRegistry()
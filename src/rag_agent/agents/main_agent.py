from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage

from ..tools.registry import tool_registry
from ..security.permission import permission_manager
from ..core.config import get_config
import logging

logger = logging.getLogger(__name__)


class RedTeamAgent:
    """Главный ReAct-агент для исследования уязвимостей LLM"""
    
    def __init__(self):
        self.llm = None
        self.graph = None
        self.config = get_config()
        self.initialized = False
    
    async def initialize(self, retrieval_service=None):
        """Инициализация агента"""
        if self.initialized:
            return
            
        await tool_registry.initialize(retrieval_service)
        
        self.llm = ChatOllama(
            model=self.config.get("ollama", {}).get("model", "llama3.2:latest"),
            base_url=self.config.get("ollama", {}).get("base_url", "http://localhost:11434"),
            temperature=0.7,
            num_ctx=8192,
        )
        
        tools = [tool for tool in tool_registry.get_all_tools()]
        
        # Создаём ReAct агент
        self.graph = create_react_agent(
            model=self.llm,
            tools=tools,
            # Можно добавить custom system prompt позже
        )
        
        self.initialized = True
        logger.info(f"✅ RedTeamAgent initialized with {len(tools)} tools")
    
    async def ainvoke(self, message: str, thread_id: str = "default", user_confirmed: bool = False, **kwargs) -> Dict[str, Any]:
        """Основной метод вызова агента"""
        if not self.initialized:
            await self.initialize()
        
        inputs = {"messages": [HumanMessage(content=message)]}
        
        try:
            result = await self.graph.ainvoke(inputs)
            final_message = result["messages"][-1].content
            
            return {
                "response": final_message,
                "thread_id": thread_id,
                "success": True,
                "used_tools": self._extract_used_tools(result)
            }
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "response": f"Ошибка при обработке запроса: {str(e)}",
                "success": False
            }
    
    def _extract_used_tools(self, result) -> List[str]:
        """Извлекает названия использованных инструментов (для отладки)"""
        tools = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    tools.append(call.get("name"))
        return tools


# Singleton
red_team_agent = RedTeamAgent()
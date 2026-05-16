from typing import Dict, Any, List
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import logging

from ..tools.registry import tool_registry
from ..tools.langchain_adapter import LangChainToolAdapter
from ..core.config import get_config  # если нет — замени на твой способ получения конфига

logger = logging.getLogger(__name__)


class RedTeamAgent:
    """Главный ReAct-агент"""
    
    def __init__(self):
        self.llm = None
        self.graph = None
        self.initialized = False
        self.config = get_config() if 'get_config' in globals() else {}

    async def initialize(self, retrieval_service=None):
        if self.initialized:
            return
            
        await tool_registry.initialize(retrieval_service)
        
        self.llm = ChatOllama(
            model=self.config.get("ollama", {}).get("model", "llama3.2:latest"),
            base_url=self.config.get("ollama", {}).get("base_url", "http://localhost:11434"),
            temperature=0.7,
        )
        
        # Адаптируем наши инструменты под LangChain
        raw_tools = tool_registry.get_all_tools()
        langchain_tools = [LangChainToolAdapter(tool) for tool in raw_tools]
        
        self.graph = create_react_agent(
            model=self.llm,
            tools=langchain_tools,
        )
        
        self.initialized = True
        logger.info(f"✅ RedTeamAgent initialized with {len(langchain_tools)} tools")


    async def ainvoke(self, message: str, thread_id: str = "default", **kwargs) -> Dict[str, Any]:
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
            }
        except Exception as e:
            logger.error(f"Agent execution error: {e}", exc_info=True)
            return {
                "response": f"Ошибка агента: {str(e)}",
                "success": False
            }


red_team_agent = RedTeamAgent()
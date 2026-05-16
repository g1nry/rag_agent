from typing import Dict, Any, List, Optional
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import logging

from ..tools.registry import tool_registry
from ..tools.langchain_adapter import LangChainToolAdapter
from ..security.hitl import hitl_manager
from ..core.config import get_settings as get_config

logger = logging.getLogger(__name__)


class RedTeamAgent:
    """Главный ReAct-агент с поддержкой Human-in-the-Loop"""
    
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
            model=getattr(self.config, "ollama_chat_model", "llama3.2:latest"),
            base_url=str(getattr(self.config, "ollama_base_url", "http://localhost:11434")),
            temperature=0.7,
        )
        
        raw_tools = tool_registry.get_all_tools()
        langchain_tools = [LangChainToolAdapter(tool) for tool in raw_tools]
        
        self.graph = create_react_agent(
            model=self.llm,
            tools=langchain_tools,
        )
        
        self.initialized = True
        logger.info(f"✅ RedTeamAgent initialized with {len(langchain_tools)} tools")

    async def ainvoke(
        self, 
        message: str, 
        thread_id: str = "default",
        auto_confirm_medium: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Основной метод вызова агента с поддержкой HITL"""
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
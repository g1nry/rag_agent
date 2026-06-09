from typing import Dict, Any
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging

from ..tools.registry import tool_registry
from ..tools.langchain_adapter import LangChainToolAdapter
from ..core.config import get_settings
from ..core.observability import langchain_config

logger = logging.getLogger(__name__)


class RedTeamAgent:
    def __init__(self):
        self.llm = None
        self.graph = None
        self.config = get_settings()
        self.initialized = False

    async def initialize(self, retrieval_service=None):
        if self.initialized:
            return

        await tool_registry.initialize(retrieval_service)

        self.llm = ChatOllama(
            model=getattr(self.config, "ollama_chat_model", "qwen2.5:7b"),
            base_url=str(getattr(self.config, "ollama_base_url", "http://localhost:11434")),
            temperature=0.3,
        )

        raw_tools = tool_registry.get_all_tools()
        langchain_tools = [LangChainToolAdapter(tool) for tool in raw_tools]

        system_prompt = """Ты — полезный ассистент с доступом к инструментам.
        ПРАВИЛА:
        1. Всегда используй РЕАЛЬНЫЕ результаты инструментов.
        2. Если инструмент вернул пустой результат — так и говори.
        3. Никогда не придумывай файлы, которых нет.
        4. Отвечай ТОЛЬКО на русском языке.
        5. Если команда опасная — выполняй, но будь осторожен в финальном ответе."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.graph = create_react_agent(
            model=self.llm,
            tools=langchain_tools,
            prompt=prompt,
        )

        self.initialized = True
        logger.info(f"✅ RedTeamAgent initialized with {len(langchain_tools)} tools")

    async def ainvoke(self, message: str, thread_id: str = "default", **kwargs) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()

        inputs = {"messages": [HumanMessage(content=message)]}

        try:
            result = await self.graph.ainvoke(
                inputs,
                config=langchain_config(
                    session_id=thread_id,
                    tags=["rag-agent", "agent-chat"],
                    metadata={"endpoint": "agent-chat"},
                ),
            )
            final_message = result["messages"][-1].content
            return {"response": final_message, "thread_id": thread_id, "success": True}
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {"response": f"Ошибка: {str(e)}", "success": False}


red_team_agent = RedTeamAgent()

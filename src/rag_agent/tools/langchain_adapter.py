from langchain.tools import BaseTool as LangchainBaseTool
from .base import BaseTool


class LangChainToolAdapter(LangchainBaseTool):
    """Адаптер, который превращает наш BaseTool в LangChain Tool"""
    
    def __init__(self, tool: BaseTool):
        super().__init__(
            name=tool.metadata.name,
            description=tool.metadata.description,
            args_schema=None  # можно улучшить позже
        )
        self._tool = tool

    async def _arun(self, **kwargs):
        return await self._tool.arun(**kwargs)

    def _run(self, **kwargs):
        # Синхронная заглушка (LangGraph иногда вызывает)
        import asyncio
        return asyncio.run(self._tool.arun(**kwargs))
#!/usr/bin/env python3
"""
Минимальный тест RedTeamAgent (только Ollama + dangerous tools, без RAG)
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.agents.main_agent import RedTeamAgent
from rag_agent.tools.dangerous_tools import create_dangerous_tools
from rag_agent.tools.registry import ToolRegistry


async def main():
    print("🧪 Минимальный тест RedTeamAgent (без RAG)")
    print("=" * 50)
    
    # Создаём агент вручную без retrieval_service
    agent = RedTeamAgent()
    
    # Инициализируем только dangerous tools
    print("📦 Регистрация dangerous tools...")
    dangerous_tools = create_dangerous_tools()
    
    # Временно подменяем registry
    from rag_agent.tools import registry as reg_module
    reg_module.tool_registry.tools = {t.metadata.name: t for t in dangerous_tools}
    reg_module.tool_registry._initialized = True
    
    print("🚀 Инициализация агента...")
    await agent.initialize(retrieval_service=None)
    
    print("\n✅ Агент готов!")
    print("Попробуй ввести команду, например:")
    print("  - 'какая сейчас дата?'")
    print("  - 'создай файл test.txt с содержимым hello'")
    print("  - 'выполни команду ls -la'")
    print("\nВведите сообщение (или 'exit'):\n")
    
    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Агент работает...")
            result = await agent.ainvoke(message=user_input, thread_id="test")
            
            print(f"\n📝 Ответ:\n{result['response']}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
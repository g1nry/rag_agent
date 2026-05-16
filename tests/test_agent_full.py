#!/usr/bin/env python3
"""
Полноценный тест RedTeamAgent (RAG + dangerous tools)
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.agents import red_team_agent
from rag_agent.services.retrieval_service import retrieval_service


async def main():
    print("🚀 Полноценный тест RedTeamAgent (RAG + Tools)")
    print("=" * 55)
    
    print("📦 Инициализация...")
    await red_team_agent.initialize(retrieval_service=retrieval_service)
    print("✅ Агент готов!\n")
    
    print("Примеры команд:")
    print("  - 'какие документы у тебя есть?'")
    print("  - 'выполни команду ls -la'")
    print("  - 'создай файл test.txt с текстом hello world'")
    print("\nВведите сообщение (или 'exit'):\n")
    
    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input:
                continue
            
            print("\n🤖 Агент думает...")
            result = await red_team_agent.ainvoke(
                message=user_input,
                thread_id="full_test"
            )
            
            print(f"\n📝 Ответ:\n{result['response']}\n")
            print("-" * 55)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
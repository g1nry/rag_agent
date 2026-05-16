#!/usr/bin/env python3
"""
Простой CLI для тестирования RedTeamAgent локально
"""
import asyncio
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.agents import red_team_agent
from rag_agent.services.retrieval_service import retrieval_service


async def main():
    print("🚀 Инициализация RedTeamAgent...")
    await red_team_agent.initialize(retrieval_service=retrieval_service)
    print("✅ Агент готов к работе!")
    print("\nВведите сообщение (или 'exit' для выхода):\n")
    
    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 До свидания!")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Агент думает...")
            result = await red_team_agent.ainvoke(
                message=user_input,
                thread_id="cli_session"
            )
            
            print(f"\n📝 Ответ:\n{result['response']}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
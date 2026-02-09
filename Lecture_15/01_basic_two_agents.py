"""
Лекция 15: Базовый диалог двух агентов
========================================
Простейший пример: User и Assistant общаются друг с другом.
"""

import os
from autogen import ConversableAgent
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# 1. Создание агентов
# ============================================================

print("="*60)
print("🤖 СОЗДАНИЕ АГЕНТОВ")
print("="*60)
print()

# User Agent (представляет пользователя)
user_agent = ConversableAgent(
    name="user",
    system_message="Ты пользователь. Задавай вопросы и отвечай на вопросы ассистента.",
    llm_config=False,  # Не использует LLM, только передаёт сообщения
    human_input_mode="ALWAYS",  # Всегда спрашивает у человека
)

# Assistant Agent (помощник)
assistant_agent = ConversableAgent(
    name="assistant",
    system_message="Ты полезный помощник. Отвечай на русском языке.",
    llm_config={
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0,
    },
    human_input_mode="NEVER",  # Никогда не спрашивает у человека
)

print("✅ User Agent создан:")
print(f"   Имя: {user_agent.name}")
print(f"   Режим: {user_agent.human_input_mode}")
print()

print("✅ Assistant Agent создан:")
print(f"   Имя: {assistant_agent.name}")
print(f"   Модель: gpt-4o-mini")
print(f"   Режим: {assistant_agent.human_input_mode}")
print()

# ============================================================
# 2. Запуск диалога
# ============================================================

print("="*60)
print("💬 ЗАПУСК ДИАЛОГА")
print("="*60)
print()

# User начинает разговор
user_agent.initiate_chat(
    recipient=assistant_agent,
    message="Привет! Расскажи мне о мультиагентных системах.",
    max_turns=3,  # Максимум 3 обмена репликами
)

print()
print("="*60)
print("✅ ДИАЛОГ ЗАВЕРШЁН")
print("="*60)

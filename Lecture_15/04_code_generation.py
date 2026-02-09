"""
Лекция 15: Генерация кода с несколькими агентами
=================================================
Code Writer, Code Reviewer, Tester работают вместе над кодом.
"""

import os
from autogen import ConversableAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# 1. Создание агентов для разработки кода
# ============================================================

print("="*60)
print("💻 СОЗДАНИЕ АГЕНТОВ ДЛЯ РАЗРАБОТКИ КОДА")
print("="*60)
print()

llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0,
}

# Code Writer - пишет код
code_writer = ConversableAgent(
    name="CodeWriter",
    system_message="""Ты программист. Твоя задача - писать чистый, рабочий код на Python.
Всегда добавляй комментарии. Отвечай на русском языке, но код пиши на английском.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Code Reviewer - проверяет код
code_reviewer = ConversableAgent(
    name="CodeReviewer",
    system_message="""Ты ревьюер кода. Твоя задача - проверять код на:
- Правильность логики
- Читаемость
- Оптимизацию
- Ошибки
Давай конструктивную обратную связь. Отвечай на русском языке.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Tester - тестирует код
tester = ConversableAgent(
    name="Tester",
    system_message="""Ты тестировщик. Твоя задача - проверять код на работоспособность.
Предлагай тестовые случаи и проверяй граничные условия.
Отвечай на русском языке.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# User Agent
user_agent = ConversableAgent(
    name="User",
    system_message="Ты пользователь. Задавай задачи по разработке кода.",
    llm_config=False,
    human_input_mode="ALWAYS",
)

print("✅ CodeWriter создан - пишет код")
print("✅ CodeReviewer создан - проверяет код")
print("✅ Tester создан - тестирует код")
print("✅ User Agent создан")
print()

# ============================================================
# 2. Создание группового чата
# ============================================================

print("="*60)
print("💬 СОЗДАНИЕ ГРУППОВОГО ЧАТА")
print("="*60)
print()

agents = [user_agent, code_writer, code_reviewer, tester]

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=15,
    speaker_selection_method="auto",
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

print("✅ GroupChat создан:")
print(f"   Участников: {len(agents)}")
print(f"   Метод выбора: auto (LLM выбирает)")
print()

# ============================================================
# 3. Запуск группового чата
# ============================================================

print("="*60)
print("🚀 ЗАПУСК ГРУППОВОГО ЧАТА")
print("="*60)
print()

user_agent.initiate_chat(
    recipient=group_chat_manager,
    message="Напиши функцию на Python, которая вычисляет факториал числа. Включи обработку ошибок.",
)

print()
print("="*60)
print("✅ ГРУППОВОЙ ЧАТ ЗАВЕРШЁН")
print("="*60)
print()
print("💡 Что произошло:")
print("   1. User задал задачу")
print("   2. CodeWriter написал код")
print("   3. CodeReviewer проверил код")
print("   4. Tester предложил тесты")
print("   5. CodeWriter улучшил код")
print("   6. User дал финальное одобрение")

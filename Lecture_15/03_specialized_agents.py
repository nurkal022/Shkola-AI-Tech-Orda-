"""
Лекция 15: Специализированные агенты
=====================================
Агенты с разными ролями: Planner, Executor, Reviewer.
Каждый выполняет свою функцию в решении задачи.
"""

import os
from autogen import ConversableAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# 1. Создание специализированных агентов
# ============================================================

print("="*60)
print("🎭 СОЗДАНИЕ СПЕЦИАЛИЗИРОВАННЫХ АГЕНТОВ")
print("="*60)
print()

llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0,
}

# Planner Agent - планирует решение задачи
planner_agent = ConversableAgent(
    name="Planner",
    system_message="""Ты планировщик. Твоя задача - разбить сложную задачу на шаги.
Создавай чёткий план действий. Отвечай на русском языке.
Формат: 1. Шаг 1, 2. Шаг 2, и т.д.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Executor Agent - выполняет шаги плана
executor_agent = ConversableAgent(
    name="Executor",
    system_message="""Ты исполнитель. Твоя задача - выполнять шаги плана.
Действуй последовательно и сообщай о результатах каждого шага.
Отвечай на русском языке.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Reviewer Agent - проверяет результаты
reviewer_agent = ConversableAgent(
    name="Reviewer",
    system_message="""Ты рецензент. Твоя задача - проверять результаты выполнения.
Указывай что сделано хорошо, что нужно улучшить.
Отвечай на русском языке.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# User Agent
user_agent = ConversableAgent(
    name="User",
    system_message="Ты пользователь. Задавай задачи и давай финальное одобрение.",
    llm_config=False,
    human_input_mode="ALWAYS",
)

print("✅ Planner Agent создан - планирует решение")
print("✅ Executor Agent создан - выполняет шаги")
print("✅ Reviewer Agent создан - проверяет результаты")
print("✅ User Agent создан")
print()

# ============================================================
# 2. Создание группового чата
# ============================================================

print("="*60)
print("💬 СОЗДАНИЕ ГРУППОВОГО ЧАТА")
print("="*60)
print()

agents = [user_agent, planner_agent, executor_agent, reviewer_agent]

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=15,
    speaker_selection_method="auto",  # LLM выбирает следующего говорящего
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

print("✅ GroupChat создан:")
print(f"   Участников: {len(agents)}")
print(f"   Метод выбора: {group_chat.speaker_selection_method} (LLM выбирает)")
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
    message="Создай план изучения Python для начинающих (3-4 шага) и опиши каждый шаг.",
)

print()
print("="*60)
print("✅ ГРУППОВОЙ ЧАТ ЗАВЕРШЁН")
print("="*60)
print()
print("💡 Что произошло:")
print("   1. User задал задачу")
print("   2. Planner создал план")
print("   3. Executor описал каждый шаг")
print("   4. Reviewer проверил результат")
print("   5. User дал финальное одобрение")

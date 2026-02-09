"""
Лекция 15: Решение сложных задач мультиагентной системой
=========================================================
Несколько агентов работают вместе над сложной задачей.
Каждый вносит свой вклад в решение.
"""

import os
from autogen import ConversableAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# 1. Создание команды агентов
# ============================================================

print("="*60)
print("👥 СОЗДАНИЕ КОМАНДЫ АГЕНТОВ")
print("="*60)
print()

llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0,
}

# Researcher - исследует проблему
researcher = ConversableAgent(
    name="Researcher",
    system_message="""Ты исследователь. Твоя задача - собирать информацию о проблеме.
Анализируй разные аспекты и предоставляй факты. Отвечай на русском языке.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Analyst - анализирует данные
analyst = ConversableAgent(
    name="Analyst",
    system_message="""Ты аналитик. Твоя задача - анализировать информацию от Researcher.
Выявляй закономерности, связи и важные моменты. Отвечай на русском языке.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Strategist - предлагает стратегию решения
strategist = ConversableAgent(
    name="Strategist",
    system_message="""Ты стратег. Твоя задача - предлагать стратегии решения проблемы.
Используй информацию от Researcher и Analyst. Отвечай на русском языке.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Implementer - реализует решение
implementer = ConversableAgent(
    name="Implementer",
    system_message="""Ты реализатор. Твоя задача - конкретизировать стратегию Strategist.
Создавай конкретные шаги и действия. Отвечай на русском языке.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# User Agent
user_agent = ConversableAgent(
    name="User",
    system_message="Ты пользователь. Задавай сложные задачи для решения.",
    llm_config=False,
    human_input_mode="ALWAYS",
)

print("✅ Researcher создан - исследует проблему")
print("✅ Analyst создан - анализирует данные")
print("✅ Strategist создан - предлагает стратегию")
print("✅ Implementer создан - реализует решение")
print("✅ User Agent создан")
print()

# ============================================================
# 2. Создание группового чата
# ============================================================

print("="*60)
print("💬 СОЗДАНИЕ ГРУППОВОГО ЧАТА")
print("="*60)
print()

agents = [user_agent, researcher, analyst, strategist, implementer]

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=20,
    speaker_selection_method="auto",
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

print("✅ GroupChat создан:")
print(f"   Участников: {len(agents)}")
print(f"   Максимум раундов: {group_chat.max_round}")
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
    message="Как можно улучшить качество онлайн-обучения? Предложи конкретные решения.",
)

print()
print("="*60)
print("✅ ГРУППОВОЙ ЧАТ ЗАВЕРШЁН")
print("="*60)
print()
print("💡 Что произошло:")
print("   1. User задал сложную задачу")
print("   2. Researcher исследовал проблему")
print("   3. Analyst проанализировал данные")
print("   4. Strategist предложил стратегию")
print("   5. Implementer конкретизировал решение")
print("   6. User дал финальное одобрение")

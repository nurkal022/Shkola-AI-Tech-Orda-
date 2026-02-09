"""
Лекция 15: Групповой чат с несколькими агентами
================================================
Несколько агентов общаются в групповом чате.
Каждый агент имеет свою роль и специализацию.
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
print("👥 СОЗДАНИЕ СПЕЦИАЛИЗИРОВАННЫХ АГЕНТОВ")
print("="*60)
print()

llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0,
}

# Writer Agent - пишет контент
writer_agent = ConversableAgent(
    name="Writer",
    system_message="""Ты писатель. Твоя задача - создавать качественный текстовый контент.
Отвечай на русском языке. Будь креативным и детальным.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Critic Agent - критикует и даёт обратную связь
critic_agent = ConversableAgent(
    name="Critic",
    system_message="""Ты критик. Твоя задача - анализировать контент и давать конструктивную обратную связь.
Указывай на сильные и слабые стороны. Отвечай на русском языке.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Editor Agent - редактирует и улучшает
editor_agent = ConversableAgent(
    name="Editor",
    system_message="""Ты редактор. Твоя задача - редактировать и улучшать контент на основе обратной связи.
Отвечай на русском языке. Делай текст лучше.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# User Agent - представляет пользователя
user_agent = ConversableAgent(
    name="User",
    system_message="Ты пользователь. Задавай задачи и давай финальное одобрение.",
    llm_config=False,
    human_input_mode="ALWAYS",
)

print("✅ Writer Agent создан")
print("✅ Critic Agent создан")
print("✅ Editor Agent создан")
print("✅ User Agent создан")
print()

# ============================================================
# 2. Создание группового чата
# ============================================================

print("="*60)
print("💬 СОЗДАНИЕ ГРУППОВОГО ЧАТА")
print("="*60)
print()

agents = [user_agent, writer_agent, critic_agent, editor_agent]

group_chat = GroupChat(
    agents=agents,
    messages=[],  # История сообщений
    max_round=12,  # Максимум раундов общения
    speaker_selection_method="round_robin",  # По очереди
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

print("✅ GroupChat создан:")
print(f"   Участников: {len(agents)}")
print(f"   Максимум раундов: {group_chat.max_round}")
print(f"   Метод выбора: {group_chat.speaker_selection_method}")
print()

# ============================================================
# 3. Запуск группового чата
# ============================================================

print("="*60)
print("🚀 ЗАПУСК ГРУППОВОГО ЧАТА")
print("="*60)
print()

# User начинает разговор
user_agent.initiate_chat(
    recipient=group_chat_manager,
    message="Напиши короткий стих о любви",
)

print()
print("="*60)
print("✅ ГРУППОВОЙ ЧАТ ЗАВЕРШЁН")
print("="*60)
print()
print("💡 Что произошло:")
print("   1. User задал задачу")
print("   2. Writer написал рассказ")
print("   3. Critic дал обратную связь")
print("   4. Editor улучшил рассказ")
print("   5. User дал финальное одобрение")

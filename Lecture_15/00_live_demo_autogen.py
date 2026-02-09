"""
Лекция 15: Live Demo - Пошаговое создание мультиагентной системы
=================================================================
Интерактивная демонстрация для лекции.
Переписываем код по этапам, показывая архитектуру AutoGen.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# ЭТАП 1: Импорты
# ============================================================
print("="*60)
print("📦 ЭТАП 1: ИМПОРТЫ")
print("="*60)
print()

from autogen import ConversableAgent, GroupChat, GroupChatManager

print("✅ Импортировали:")
print("   • ConversableAgent - базовый класс для агентов")
print("   • GroupChat - групповой чат")
print("   • GroupChatManager - менеджер группового чата")
print()

# ============================================================
# ЭТАП 2: Настройка LLM
# ============================================================
print("="*60)
print("🧠 ЭТАП 2: НАСТРОЙКА LLM")
print("="*60)
print()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0,
}

print("✅ LLM конфигурация:")
print(f"   Модель: {llm_config['model']}")
print(f"   Temperature: {llm_config['temperature']}")
print()

# ============================================================
# ЭТАП 3: Создание первого агента
# ============================================================
print("="*60)
print("🤖 ЭТАП 3: СОЗДАНИЕ ПЕРВОГО АГЕНТА")
print("="*60)
print()

assistant_agent = ConversableAgent(
    name="Assistant",
    system_message="Ты полезный помощник. Отвечай на русском языке.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

print("✅ Assistant Agent создан:")
print(f"   Имя: {assistant_agent.name}")
print(f"   System message: {assistant_agent.system_message[:50]}...")
print(f"   Human input mode: {assistant_agent.human_input_mode}")
print()

# ============================================================
# ЭТАП 4: Создание User агента
# ============================================================
print("="*60)
print("👤 ЭТАП 4: СОЗДАНИЕ USER АГЕНТА")
print("="*60)
print()

user_agent = ConversableAgent(
    name="User",
    system_message="Ты пользователь.",
    llm_config=False,  # Не использует LLM
    human_input_mode="ALWAYS",  # Всегда спрашивает у человека
)

print("✅ User Agent создан:")
print(f"   Имя: {user_agent.name}")
print(f"   LLM: Нет (только передаёт сообщения)")
print(f"   Human input mode: {user_agent.human_input_mode}")
print()

# ============================================================
# ЭТАП 5: Простой диалог двух агентов
# ============================================================
print("="*60)
print("💬 ЭТАП 5: ПРОСТОЙ ДИАЛОГ")
print("="*60)
print()

print("Запускаем диалог между User и Assistant...")
print("─" * 60)

user_agent.initiate_chat(
    recipient=assistant_agent,
    message="Привет! Расскажи кратко о мультиагентных системах.",
    max_turns=2,
)

print("─" * 60)
print()

# ============================================================
# ЭТАП 6: Создание специализированных агентов
# ============================================================
print("="*60)
print("🎭 ЭТАП 6: СПЕЦИАЛИЗИРОВАННЫЕ АГЕНТЫ")
print("="*60)
print()

writer_agent = ConversableAgent(
    name="Writer",
    system_message="Ты писатель. Создавай качественный контент. Отвечай на русском.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

critic_agent = ConversableAgent(
    name="Critic",
    system_message="Ты критик. Давай конструктивную обратную связь. Отвечай на русском.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

print("✅ Writer Agent создан")
print("✅ Critic Agent создан")
print()

# ============================================================
# ЭТАП 7: Создание группового чата
# ============================================================
print("="*60)
print("💬 ЭТАП 7: СОЗДАНИЕ ГРУППОВОГО ЧАТА")
print("="*60)
print()

agents = [user_agent, writer_agent, critic_agent]

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=10,
    speaker_selection_method="round_robin",  # По очереди
)

print("✅ GroupChat создан:")
print(f"   Участников: {len(agents)}")
print(f"   Максимум раундов: {group_chat.max_round}")
print(f"   Метод выбора: {group_chat.speaker_selection_method}")
print()

# ============================================================
# ЭТАП 8: Создание GroupChatManager
# ============================================================
print("="*60)
print("⚙️  ЭТАП 8: СОЗДАНИЕ GroupChatManager")
print("="*60)
print()

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

print("✅ GroupChatManager создан:")
print("   Управляет групповым чатом")
print("   Выбирает следующего говорящего")
print("   Координирует общение агентов")
print()

# ============================================================
# ЭТАП 9: Запуск группового чата
# ============================================================
print("="*60)
print("🚀 ЭТАП 9: ЗАПУСК ГРУППОВОГО ЧАТА")
print("="*60)
print()

print("Запускаем групповой чат...")
print("─" * 60)

user_agent.initiate_chat(
    recipient=group_chat_manager,
    message="Напиши короткий рассказ о роботе (2-3 предложения).",
)

print("─" * 60)
print()

# ============================================================
# Итоги
# ============================================================
print("="*60)
print("✅ LIVE DEMO ЗАВЕРШЕНА")
print("="*60)
print()
print("💡 Ключевые концепции AutoGen:")
print("   1. ConversableAgent - базовый класс агента")
print("   2. System message - определяет роль агента")
print("   3. Human input mode - когда спрашивать человека")
print("   4. GroupChat - групповой чат с несколькими агентами")
print("   5. GroupChatManager - управляет общением")
print("   6. Speaker selection - метод выбора следующего говорящего")
print()

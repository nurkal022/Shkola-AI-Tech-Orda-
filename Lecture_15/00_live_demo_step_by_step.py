"""
Лекция 15: Live Demo - Пошаговое создание (упрощённая версия)
==============================================================
Упрощённая версия для переписывания на лекции.
Раскомментируйте по этапам.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# ЭТАП 1: Импорты
# ============================================================
# from autogen import ConversableAgent, GroupChat, GroupChatManager

# ============================================================
# ЭТАП 2: Конфигурация LLM
# ============================================================
# llm_config = {
#     "model": "gpt-4o-mini",
#     "api_key": os.getenv("OPENAI_API_KEY"),
#     "temperature": 0,
# }

# ============================================================
# ЭТАП 3: Создание агентов
# ============================================================
# assistant = ConversableAgent(
#     name="Assistant",
#     system_message="Ты помощник. Отвечай на русском.",
#     llm_config=llm_config,
#     human_input_mode="NEVER",
# )
#
# user = ConversableAgent(
#     name="User",
#     system_message="Ты пользователь.",
#     llm_config=False,
#     human_input_mode="ALWAYS",
# )

# ============================================================
# ЭТАП 4: Простой диалог
# ============================================================
# user.initiate_chat(
#     recipient=assistant,
#     message="Привет!",
#     max_turns=2,
# )

# ============================================================
# ЭТАП 5: Групповой чат
# ============================================================
# group_chat = GroupChat(
#     agents=[user, assistant],
#     messages=[],
#     max_round=10,
# )
#
# manager = GroupChatManager(
#     groupchat=group_chat,
#     llm_config=llm_config,
# )
#
# user.initiate_chat(recipient=manager, message="Привет!")

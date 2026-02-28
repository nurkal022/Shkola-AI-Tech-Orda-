"""
Лекция 18: Агент для углублённых исследований
Live Demo — Версия для совместного написания
==============================================
Минимальный агент-исследователь:
  - ищет из нескольких источников
  - синтезирует информацию
  - сохраняет отчёт в файл
"""

import os
import requests
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден!")
    exit(1)

# ─────────────────────────────────────────────
# Шаг 1: Инструменты исследователя
# ─────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Ищет актуальную информацию в интернете.
    Args:
        query: Поисковый запрос (лучше на английском)
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        output = f"Результаты поиска '{query}':\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r['title']}\n   {r['body'][:200]}\n   URL: {r['href']}\n\n"
        return output
    except Exception as e:
        return f"Ошибка поиска: {e}"


@tool
def wikipedia_summary(topic: str) -> str:
    """Получает справку из Wikipedia.
    Args:
        topic: Тема на английском (Python, AI agents, Kazakhstan)
    """
    try:
        headers = {"User-Agent": "TechOrda-Education/1.0"}
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return f"Статья '{topic}' не найдена в Wikipedia"
        data = r.json()
        extract = data.get("extract", "")[:700]
        return f"Wikipedia — {data.get('title', topic)}:\n\n{extract}"
    except Exception as e:
        return f"Ошибка Wikipedia: {e}"


@tool
def save_report(filename: str, content: str) -> str:
    """Сохраняет готовый исследовательский отчёт в markdown файл.
    Args:
        filename: Имя файла (например, 'research_ai.md')
        content: Содержимое отчёта в markdown формате
    """
    try:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ Отчёт сохранён: {filepath} ({len(content.split())} слов)"
    except Exception as e:
        return f"Ошибка сохранения: {e}"


# ─────────────────────────────────────────────
# Шаг 2: Создаём агента-исследователя
# ─────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [web_search, wikipedia_summary, save_report]

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты профессиональный AI-исследователь.
Для каждой темы:
1. Получи базовую информацию через wikipedia_summary
2. Найди актуальные данные через web_search (2-3 запроса)
3. Составь структурированный отчёт на РУССКОМ языке
4. Сохрани отчёт через save_report

Формат отчёта (markdown):
# [Тема]
## Что это такое?
[объяснение]
## Ключевые факты
- факт 1
- факт 2
## Актуальное состояние (2025)
[тренды и новости]
## Заключение
[выводы]
## Источники
- Wikipedia
- [найденные URL]""",
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

# ─────────────────────────────────────────────
# Шаг 3: Запускаем исследование
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 АГЕНТ ДЛЯ УГЛУБЛЁННЫХ ИССЛЕДОВАНИЙ")
    print("=" * 60)
    print()

    topic = "AI Agents"

    print(f"📚 Тема: {topic}")
    print()

    result = agent_executor.invoke({
        "input": (
            f"Проведи углублённое исследование темы: '{topic}'. "
            f"Используй wikipedia_summary и web_search (минимум 2 поисковых запроса). "
            f"Составь полный отчёт и сохрани в файл 'research_ai_agents.md'."
        )
    })

    print()
    print("=" * 60)
    print("📊 РЕЗЮМЕ:")
    print("=" * 60)
    print(result["output"])

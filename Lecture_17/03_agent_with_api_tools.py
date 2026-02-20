"""
Лекция 17: Агент с API-инструментами
======================================
Создание LangChain агента с кастомными инструментами
для вызова внешних API. Агент сам решает когда и какой
инструмент использовать.
"""

import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Загружаем .env
load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# 1. Определяем инструменты (Tools)
# ============================================================

print("=" * 60)
print("🔧 1. ОПРЕДЕЛЯЕМ ИНСТРУМЕНТЫ (TOOLS)")
print("=" * 60)
print()


@tool
def get_weather(city: str) -> str:
    """Получает текущую погоду для указанного города.
    Используйте этот инструмент когда пользователь спрашивает о погоде.

    Args:
        city: Название города (на английском), например 'Astana', 'London'
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        data = response.json()

        current = data["current_condition"][0]
        result = (
            f"Погода в {city}:\n"
            f"🌡️ Температура: {current['temp_C']}°C\n"
            f"💨 Ветер: {current['windspeedKmph']} км/ч\n"
            f"💧 Влажность: {current['humidity']}%\n"
            f"☁️ Описание: {current['weatherDesc'][0]['value']}"
        )
        return result
    except Exception as e:
        return f"Ошибка получения погоды: {e}"


@tool
def search_wikipedia(query: str) -> str:
    """Ищет информацию в Wikipedia по заданному запросу.
    Используйте этот инструмент когда нужно найти факты или определения.

    Args:
        query: Поисковый запрос (на английском), например 'Python programming'
    """
    try:
        # Поиск в Wikipedia API
        headers = {"User-Agent": "TechOrda-Education/1.0"}
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        }
        response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
        search_results = response.json()

        if not search_results["query"]["search"]:
            return f"Ничего не найдено по запросу: {query}"

        # Берём первый результат и получаем его содержимое
        title = search_results["query"]["search"][0]["title"]
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        response = requests.get(summary_url, headers=headers, timeout=10)
        data = response.json()

        extract = data.get("extract", "Нет описания")
        # Ограничиваем длину
        if len(extract) > 500:
            extract = extract[:500] + "..."

        return f"Wikipedia: {data['title']}\n\n{extract}"
    except Exception as e:
        return f"Ошибка поиска в Wikipedia: {e}"


@tool
def get_random_fact() -> str:
    """Возвращает случайный интересный факт.
    Используйте когда пользователь просит рассказать что-нибудь интересное.
    """
    try:
        url = "https://uselessfacts.jsph.pl/api/v2/facts/random"
        response = requests.get(url, timeout=10)
        data = response.json()
        return f"Интересный факт: {data['text']}"
    except Exception as e:
        return f"Ошибка: {e}"


@tool
def get_github_repo_info(repo: str) -> str:
    """Получает информацию о GitHub репозитории.
    Используйте когда пользователь спрашивает о GitHub проекте.

    Args:
        repo: Путь к репозиторию в формате 'owner/repo', например 'langchain-ai/langchain'
    """
    try:
        url = f"https://api.github.com/repos/{repo}"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return f"Репозиторий {repo} не найден"

        data = response.json()
        result = (
            f"GitHub: {data['full_name']}\n"
            f"📝 Описание: {data.get('description', 'N/A')}\n"
            f"⭐ Звёзды: {data['stargazers_count']}\n"
            f"🍴 Форки: {data['forks_count']}\n"
            f"👁️ Watchers: {data['watchers_count']}\n"
            f"💻 Язык: {data.get('language', 'N/A')}\n"
            f"📅 Обновлён: {data['updated_at'][:10]}"
        )
        return result
    except Exception as e:
        return f"Ошибка: {e}"


# Список всех инструментов
tools = [get_weather, search_wikipedia, get_random_fact, get_github_repo_info]

print("✅ Инструменты определены:")
for t in tools:
    print(f"   🔧 {t.name}: {t.description[:60]}...")
print()

# ============================================================
# 2. Создаём LLM и агента
# ============================================================

print("=" * 60)
print("🤖 2. СОЗДАЁМ LLM И АГЕНТА")
print("=" * 60)
print()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# Промпт для агента
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты полезный AI-ассистент с доступом к внешним инструментам. "
        "Используй инструменты для получения актуальной информации. "
        "Всегда отвечай на русском языке. "
        "Если пользователь спрашивает о погоде — используй get_weather. "
        "Если нужна информация — используй search_wikipedia. "
        "Если спрашивают о GitHub — используй get_github_repo_info."
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Создаём агента
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("✅ Агент создан с инструментами:")
print(f"   • LLM: gpt-4o-mini")
print(f"   • Инструментов: {len(tools)}")
print()

# ============================================================
# 3. Тестируем агента с разными запросами
# ============================================================

print("=" * 60)
print("🧪 3. ТЕСТИРУЕМ АГЕНТА")
print("=" * 60)
print()

# Запрос 1: Погода
print("─" * 60)
print("📝 Запрос 1: 'Какая погода в Астане?'")
print("─" * 60)
result = agent_executor.invoke({"input": "Какая погода в Астане?"})
print()
print(f"🤖 Ответ: {result['output']}")
print()

# Запрос 2: Wikipedia
print("─" * 60)
print("📝 Запрос 2: 'Что такое машинное обучение?'")
print("─" * 60)
result = agent_executor.invoke({"input": "Что такое машинное обучение? Найди в Wikipedia."})
print()
print(f"🤖 Ответ: {result['output']}")
print()

# Запрос 3: GitHub
print("─" * 60)
print("📝 Запрос 3: 'Расскажи о репозитории langchain-ai/langchain'")
print("─" * 60)
result = agent_executor.invoke({"input": "Расскажи о GitHub репозитории langchain-ai/langchain"})
print()
print(f"🤖 Ответ: {result['output']}")
print()

# ============================================================
# Итоги
# ============================================================

print("=" * 60)
print("✅ ИТОГИ: АГЕНТ С API-ИНСТРУМЕНТАМИ")
print("=" * 60)
print()
print("📚 Что мы изучили:")
print("   1. @tool декоратор — создание кастомных инструментов")
print("   2. Docstring — описание для LLM (когда использовать)")
print("   3. Args — параметры инструмента")
print("   4. AgentExecutor — запуск агента с инструментами")
print("   5. Агент сам выбирает какой инструмент использовать")
print()
print("💡 Ключевые принципы:")
print("   • Хороший docstring = правильный выбор инструмента")
print("   • Обработка ошибок в каждом инструменте")
print("   • Timeout для всех HTTP-запросов")
print("   • Ограничение длины ответов")
print()

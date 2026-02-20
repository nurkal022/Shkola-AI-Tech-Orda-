"""
Лекция 17: Мульти-инструментальный исследовательский агент
===========================================================
Полноценный агент, комбинирующий API-вызовы и веб-скрапинг
для глубокого исследования темы. Объединяет все изученные
техники в один мощный агент.
"""

import os
import json
import requests
from bs4 import BeautifulSoup
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
# 1. Полный набор инструментов
# ============================================================

print("=" * 60)
print("🔧 ПОЛНЫЙ НАБОР ИНСТРУМЕНТОВ ИССЛЕДОВАТЕЛЯ")
print("=" * 60)
print()


# --- API инструменты ---

@tool
def wikipedia_search(query: str) -> str:
    """Поиск и получение информации из Wikipedia.
    Используй для получения определений, фактов, исторических данных.

    Args:
        query: Поисковый запрос на английском языке
    """
    try:
        # Шаг 1: Поиск
        headers = {"User-Agent": "TechOrda-Education/1.0"}
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 5,
        }
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        results = response.json()["query"]["search"]

        if not results:
            return f"Wikipedia: ничего не найдено по запросу '{query}'"

        # Шаг 2: Получение содержимого топ-3 результатов
        output = f"Wikipedia результаты для '{query}':\n\n"
        for i, result in enumerate(results[:3], 1):
            title = result["title"]
            # Получаем summary
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            try:
                headers = {"User-Agent": "TechOrda-Education/1.0"}
                summary_resp = requests.get(summary_url, headers=headers, timeout=5)
                summary_data = summary_resp.json()
                extract = summary_data.get("extract", "")
                if len(extract) > 300:
                    extract = extract[:300] + "..."
                output += f"{i}. {title}\n   {extract}\n\n"
            except Exception:
                output += f"{i}. {title}\n   (не удалось загрузить описание)\n\n"

        return output
    except Exception as e:
        return f"Ошибка Wikipedia: {e}"


@tool
def github_search(query: str) -> str:
    """Поиск репозиториев на GitHub по ключевым словам.
    Используй для поиска open-source проектов и библиотек.

    Args:
        query: Поисковый запрос, например 'langchain agents python'
    """
    try:
        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 5,
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if not data.get("items"):
            return f"GitHub: ничего не найдено по '{query}'"

        result = f"GitHub репозитории для '{query}':\n\n"
        for i, repo in enumerate(data["items"], 1):
            result += f"{i}. {repo['full_name']}\n"
            result += f"   Описание: {repo.get('description', 'N/A')}\n"
            result += f"   ⭐ {repo['stargazers_count']} | 🍴 {repo['forks_count']} | 💻 {repo.get('language', 'N/A')}\n"
            result += f"   URL: {repo['html_url']}\n\n"

        return result
    except Exception as e:
        return f"Ошибка GitHub: {e}"


@tool
def get_weather(city: str) -> str:
    """Получает текущую погоду для города.

    Args:
        city: Город на английском (Astana, Moscow, London)
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        data = response.json()
        current = data["current_condition"][0]
        return (
            f"Погода в {city}: {current['temp_C']}°C, "
            f"ветер {current['windspeedKmph']} км/ч, "
            f"влажность {current['humidity']}%, "
            f"{current['weatherDesc'][0]['value']}"
        )
    except Exception as e:
        return f"Ошибка: {e}"


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Получает курс обмена между двумя валютами.

    Args:
        from_currency: Исходная валюта (USD, EUR, KZT, RUB)
        to_currency: Целевая валюта (USD, EUR, KZT, RUB)
    """
    try:
        url = f"https://open.er-api.com/v6/latest/{from_currency}"
        response = requests.get(url, timeout=10)
        data = response.json()

        if to_currency not in data.get("rates", {}):
            return f"Валюта {to_currency} не найдена"

        rate = data["rates"][to_currency]
        return f"Курс: 1 {from_currency} = {rate:.4f} {to_currency}"
    except Exception as e:
        return f"Ошибка: {e}"


# --- Скрапинг инструменты ---

@tool
def scrape_page(url: str) -> str:
    """Скрапит веб-страницу и извлекает текстовое содержимое.
    Используй для получения информации с любой веб-страницы.

    Args:
        url: Полный URL страницы (https://...)
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Educational Research Bot)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Удаляем ненужное
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Извлекаем заголовки и параграфы
        content_parts = []

        title = soup.find("title")
        if title:
            content_parts.append(f"Заголовок: {title.get_text(strip=True)}")

        for heading in soup.find_all(["h1", "h2", "h3"])[:10]:
            content_parts.append(f"\n## {heading.get_text(strip=True)}")

        for p in soup.find_all("p")[:20]:
            text = p.get_text(strip=True)
            if len(text) > 30:
                content_parts.append(text)

        text = "\n".join(content_parts)
        if len(text) > 2000:
            text = text[:2000] + "\n...[обрезано]"

        return f"Содержимое {url}:\n\n{text}"
    except Exception as e:
        return f"Ошибка скрапинга: {e}"


@tool
def get_hacker_news(count: int = 5) -> str:
    """Получает последние новости с Hacker News.

    Args:
        count: Количество новостей (1-10)
    """
    try:
        count = min(max(count, 1), 10)
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        response = requests.get(url, timeout=10)
        story_ids = response.json()[:count]

        result = "Топ новости Hacker News:\n\n"
        for i, sid in enumerate(story_ids, 1):
            story = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                timeout=5,
            ).json()
            result += (
                f"{i}. {story.get('title', 'N/A')}\n"
                f"   Score: {story.get('score', 0)} | "
                f"Комментарии: {story.get('descendants', 0)}\n\n"
            )

        return result
    except Exception as e:
        return f"Ошибка: {e}"


@tool
def save_report(filename: str, content: str) -> str:
    """Сохраняет отчёт исследования в файл.
    Используй для сохранения итоговых результатов.

    Args:
        filename: Имя файла (без пути), например 'research_ai.md'
        content: Содержимое отчёта
    """
    try:
        # Сохраняем в папку output
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return f"✅ Отчёт сохранён: {filepath}"
    except Exception as e:
        return f"Ошибка сохранения: {e}"


# Все инструменты
tools = [
    wikipedia_search,
    github_search,
    get_weather,
    get_exchange_rate,
    scrape_page,
    get_hacker_news,
    save_report,
]

print("✅ Все инструменты определены:")
for t in tools:
    print(f"   🔧 {t.name}")
print()

# ============================================================
# 2. Создаём исследовательского агента
# ============================================================

print("=" * 60)
print("🤖 СОЗДАЁМ ИССЛЕДОВАТЕЛЬСКОГО АГЕНТА")
print("=" * 60)
print()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты исследовательский AI-агент с доступом к множеству инструментов.

Твои возможности:
- Wikipedia — для поиска определений и фактов
- GitHub — для поиска open-source проектов
- Погода — текущая погода в любом городе
- Курсы валют — актуальные обменные курсы
- Веб-скрапинг — извлечение данных с веб-страниц
- Hacker News — последние IT новости
- Сохранение отчётов — запись результатов в файл

Правила:
1. Используй несколько инструментов для полного ответа
2. Отвечай на русском языке
3. Структурируй ответ с заголовками и списками
4. Если просят исследовать тему — используй 2-3 источника
5. Можешь сохранять результаты в файл через save_report"""
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,  # Больше итераций для сложных задач
)

print("✅ Исследовательский агент создан")
print(f"   Инструментов: {len(tools)}")
print(f"   Max итераций: 10")
print()

# ============================================================
# 3. Исследовательские задачи
# ============================================================

print("=" * 60)
print("🧪 ИССЛЕДОВАТЕЛЬСКИЕ ЗАДАЧИ")
print("=" * 60)
print()

# Задача 1: Исследование темы
print("─" * 60)
print("📝 Задача 1: Исследование темы 'AI Agents'")
print("─" * 60)

result = agent_executor.invoke({
    "input": (
        "Исследуй тему 'AI Agents' (AI агенты). "
        "Найди информацию в Wikipedia, "
        "найди популярные репозитории на GitHub по этой теме, "
        "и покажи последние новости из Hacker News. "
        "Составь краткий отчёт."
    )
})
print()
print("🤖 Результат исследования:")
print("─" * 60)
print(result["output"])
print()

# Задача 2: Комплексный запрос
print("─" * 60)
print("📝 Задача 2: Комплексный информационный запрос")
print("─" * 60)

result = agent_executor.invoke({
    "input": (
        "Мне нужна следующая информация: "
        "1) Текущая погода в Алматы "
        "2) Курс доллара к тенге "
        "3) Краткая информация о Python из Wikipedia "
        "Собери всё в один ответ."
    )
})
print()
print("🤖 Результат:")
print("─" * 60)
print(result["output"])
print()

# ============================================================
# Итоги
# ============================================================

print("=" * 60)
print("✅ ИТОГИ: МУЛЬТИ-ИНСТРУМЕНТАЛЬНЫЙ ИССЛЕДОВАТЕЛЬ")
print("=" * 60)
print()
print("📚 Архитектура агента:")
print()
print("   ┌─────────────────────┐")
print("   │   Пользователь      │")
print("   └──────────┬──────────┘")
print("              │")
print("   ┌──────────▼──────────┐")
print("   │    LLM (GPT-4o)     │")
print("   │  Анализ запроса     │")
print("   │  Выбор инструментов │")
print("   └──────────┬──────────┘")
print("              │")
print("   ┌──────────▼──────────┐")
print("   │   Agent Executor    │")
print("   │  Цикл: Think → Act │")
print("   └──────────┬──────────┘")
print("              │")
print("   ┌──────────▼──────────────────────┐")
print("   │         Инструменты             │")
print("   │  ┌──────┐ ┌──────┐ ┌──────────┐ │")
print("   │  │ Wiki │ │GitHub│ │ Скрапинг │ │")
print("   │  └──────┘ └──────┘ └──────────┘ │")
print("   │  ┌──────┐ ┌──────┐ ┌──────────┐ │")
print("   │  │Погода│ │Валюты│ │  Новости │ │")
print("   │  └──────┘ └──────┘ └──────────┘ │")
print("   └─────────────────────────────────┘")
print()
print("💡 Ключевые принципы:")
print("   1. Чёткие docstring — LLM понимает когда использовать инструмент")
print("   2. Обработка ошибок — каждый инструмент обрабатывает свои ошибки")
print("   3. Ограничение данных — не перегружаем контекст LLM")
print("   4. Комбинирование — агент использует несколько инструментов")
print("   5. Сохранение — результаты можно сохранить в файл")
print()

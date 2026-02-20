"""
Лекция 17: Агент с поиском в интернете
========================================
Агент, который умеет искать в интернете через DuckDuckGo,
а затем скрапить найденные страницы для получения деталей.
Комбинирует поиск + скрапинг + API для полноценного исследования.
"""

import os
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
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
# 1. Инструменты поиска
# ============================================================

print("=" * 60)
print("🔧 1. ИНСТРУМЕНТЫ ПОИСКА")
print("=" * 60)
print()


@tool
def web_search(query: str) -> str:
    """Ищет информацию в интернете через DuckDuckGo.
    Используй этот инструмент когда нужно найти актуальную информацию,
    новости, ответы на вопросы или любые данные из интернета.

    Args:
        query: Поисковый запрос (на любом языке)
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"Ничего не найдено по запросу: {query}"

        output = f"Результаты поиска для '{query}':\n\n"
        for i, r in enumerate(results, 1):
            title = r.get("title", "N/A")
            body = r.get("body", "N/A")
            href = r.get("href", "N/A")
            output += f"{i}. {title}\n   {body[:150]}\n   URL: {href}\n\n"

        return output
    except Exception as e:
        return f"Ошибка поиска: {e}"


@tool
def web_search_news(query: str) -> str:
    """Ищет последние новости через DuckDuckGo News.
    Используй когда пользователь спрашивает о новостях или
    последних событиях.

    Args:
        query: Поисковый запрос для новостей
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=5))

        if not results:
            return f"Новости не найдены по запросу: {query}"

        output = f"Новости по запросу '{query}':\n\n"
        for i, r in enumerate(results, 1):
            title = r.get("title", "N/A")
            body = r.get("body", "N/A")
            source = r.get("source", "N/A")
            date = r.get("date", "N/A")
            url = r.get("url", "N/A")
            output += (
                f"{i}. {title}\n"
                f"   Источник: {source} | Дата: {date}\n"
                f"   {body[:150]}\n"
                f"   URL: {url}\n\n"
            )

        return output
    except Exception as e:
        return f"Ошибка поиска новостей: {e}"


@tool
def scrape_url(url: str) -> str:
    """Загружает и извлекает текстовое содержимое веб-страницы.
    Используй после web_search, чтобы получить подробную информацию
    со страницы из результатов поиска.

    Args:
        url: Полный URL страницы (https://...)
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Educational Bot)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Удаляем ненужные элементы
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        # Собираем заголовки и параграфы
        parts = []
        title = soup.find("title")
        if title:
            parts.append(f"Заголовок: {title.get_text(strip=True)}\n")

        for el in soup.find_all(["h1", "h2", "h3", "p"])[:30]:
            text = el.get_text(strip=True)
            if len(text) > 20:
                if el.name.startswith("h"):
                    parts.append(f"\n## {text}")
                else:
                    parts.append(text)

        content = "\n".join(parts)
        if len(content) > 3000:
            content = content[:3000] + "\n\n...[текст обрезан]"

        return f"Содержимое {url}:\n\n{content}"
    except Exception as e:
        return f"Ошибка скрапинга {url}: {e}"


@tool
def wikipedia_summary(topic: str) -> str:
    """Получает краткую справку из Wikipedia.

    Args:
        topic: Тема на английском (Python, Machine learning, Kazakhstan)
    """
    try:
        headers = {"User-Agent": "TechOrda-Education/1.0"}
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return f"Статья '{topic}' не найдена в Wikipedia"

        data = response.json()
        extract = data.get("extract", "")
        if len(extract) > 500:
            extract = extract[:500] + "..."

        return f"Wikipedia — {data.get('title', topic)}:\n\n{extract}"
    except Exception as e:
        return f"Ошибка Wikipedia: {e}"


tools = [web_search, web_search_news, scrape_url, wikipedia_summary]

print("✅ Инструменты поиска определены:")
for t in tools:
    print(f"   🔧 {t.name}: {t.description.split(chr(10))[0]}")
print()

# ============================================================
# 2. Демонстрация DuckDuckGo без агента
# ============================================================

print("=" * 60)
print("🔍 2. ДЕМОНСТРАЦИЯ DUCKDUCKGO (БЕЗ АГЕНТА)")
print("=" * 60)
print()

print("💡 DuckDuckGo Search — бесплатный поиск без API-ключа")
print()

# Текстовый поиск
print("📝 Текстовый поиск: 'AI agents 2024'")
print("─" * 40)
result = web_search.invoke("AI agents 2024")
# Показываем первые 500 символов
print(result[:500])
print()

# Поиск новостей
print("📝 Поиск новостей: 'artificial intelligence'")
print("─" * 40)
result = web_search_news.invoke("artificial intelligence")
print(result[:500])
print()

# ============================================================
# 3. Создаём поискового агента
# ============================================================

print("=" * 60)
print("🤖 3. СОЗДАЁМ ПОИСКОВОГО АГЕНТА")
print("=" * 60)
print()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты AI-ассистент с доступом к поиску в интернете.

Твои инструменты:
- web_search — поиск в интернете (DuckDuckGo)
- web_search_news — поиск последних новостей
- scrape_url — загрузка и чтение содержимого веб-страницы
- wikipedia_summary — краткая справка из Wikipedia

Правила:
1. Для общих вопросов используй web_search
2. Для новостей используй web_search_news
3. Если нужно подробнее — скрапи страницу через scrape_url
4. Для определений и фактов — wikipedia_summary
5. Отвечай на русском языке
6. Давай структурированные ответы с источниками"""
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
    max_iterations=8,
)

print("✅ Поисковый агент создан")
print()

# ============================================================
# 4. Тестируем поискового агента
# ============================================================

print("=" * 60)
print("🧪 4. ТЕСТИРУЕМ ПОИСКОВОГО АГЕНТА")
print("=" * 60)
print()

# Запрос 1: Общий поиск
print("─" * 60)
print("📝 Запрос 1: 'Что такое LangChain и для чего он нужен?'")
print("─" * 60)
result = agent_executor.invoke({
    "input": "Что такое LangChain и для чего он нужен? Найди актуальную информацию."
})
print()
print(f"🤖 Ответ:\n{result['output'][:800]}")
print()

# # Запрос 2: Поиск новостей
# print("─" * 60)
# print("📝 Запрос 2: 'Последние новости об AI'")
# print("─" * 60)
# result = agent_executor.invoke({
#     "input": "Какие последние новости в мире искусственного интеллекта?"
# })
# print()
# print(f"🤖 Ответ:\n{result['output'][:800]}")
# print()

# # Запрос 3: Исследование с поиском + скрапинг
# print("─" * 60)
# print("📝 Запрос 3: 'Расскажи про фреймворк CrewAI'")
# print("─" * 60)
# result = agent_executor.invoke({
#     "input": "Расскажи подробно про фреймворк CrewAI для мультиагентных систем. Найди информацию в интернете."
# })
# print()
# print(f"🤖 Ответ:\n{result['output'][:800]}")
# print()


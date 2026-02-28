"""
Лекция 18: Агент для углублённых исследований — Полный пайплайн
================================================================
Объединяем все компоненты в одну систему:

  Фаза 1: Планирование   — LLM разбивает тему на подтемы
  Фаза 2: Сбор данных    — Агент ищет по каждой подтеме
  Фаза 3: Синтез         — LLM объединяет находки
  Фаза 4: Отчёт          — Генерируем и сохраняем документ
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден!")
    exit(1)

print("=" * 60)
print("🔬 АГЕНТ ДЛЯ УГЛУБЛЁННЫХ ИССЛЕДОВАНИЙ")
print("=" * 60)
print()
print("Пайплайн:")
print("  [Тема] → Планировщик → Агент-Исследователь → Генератор → [Отчёт]")
print()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# Фаза 1: Инструменты исследователя
# ============================================================

print("=" * 60)
print("🔧 ФАЗА 1: ИНСТРУМЕНТЫ ИССЛЕДОВАТЕЛЯ")
print("=" * 60)
print()


@tool
def search_web(query: str) -> str:
    """Ищет актуальную информацию через DuckDuckGo.
    Используй для общих фактов, статей, мнений экспертов.

    Args:
        query: Поисковый запрос (лучше на английском)
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"Ничего не найдено: {query}"
        output = f"🔍 Поиск: '{query}'\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r['title']}\n   {r['body'][:250]}\n   URL: {r['href']}\n\n"
        return output
    except Exception as e:
        return f"Ошибка поиска: {e}"


@tool
def search_news(query: str) -> str:
    """Ищет последние новости через DuckDuckGo News.
    Используй для актуальных событий и трендов.

    Args:
        query: Запрос для новостей
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=5))
        if not results:
            return f"Новости не найдены: {query}"
        output = f"📰 Новости: '{query}'\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('title', 'N/A')}\n"
            output += f"   {r.get('source', '')} | {r.get('date', '')}\n"
            output += f"   {r.get('body', '')[:200]}\n\n"
        return output
    except Exception as e:
        return f"Ошибка новостей: {e}"


@tool
def wikipedia_search(topic: str) -> str:
    """Получает подробную информацию из Wikipedia.
    Используй для базовых определений и исторических фактов.

    Args:
        topic: Тема на английском
    """
    try:
        headers = {"User-Agent": "TechOrda-DeepResearch/1.0"}

        # Поиск нужной статьи
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "format": "json",
            "srlimit": 3,
        }
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=search_params,
            headers=headers,
            timeout=10,
        )
        results = resp.json()["query"]["search"]

        if not results:
            return f"Wikipedia: '{topic}' не найдено"

        # Берём первый результат
        title = results[0]["title"]
        summary_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        )
        data = requests.get(summary_url, headers=headers, timeout=10).json()

        extract = data.get("extract", "")[:800]
        output = f"📖 Wikipedia: {data.get('title', title)}\n\n{extract}"

        related = [r["title"] for r in results[1:3]]
        if related:
            output += f"\n\nСвязанные статьи: {', '.join(related)}"

        return output
    except Exception as e:
        return f"Ошибка Wikipedia: {e}"


@tool
def scrape_page(url: str) -> str:
    """Читает и извлекает текст с конкретной веб-страницы.
    Используй чтобы получить детали из статьи по URL.

    Args:
        url: Полный URL страницы (https://...)
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Educational Research Bot)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        parts = []
        title = soup.find("title")
        if title:
            parts.append(f"# {title.get_text(strip=True)}\n")

        for el in soup.find_all(["h1", "h2", "h3", "p"])[:35]:
            text = el.get_text(strip=True)
            if len(text) > 25:
                parts.append(f"## {text}" if el.name.startswith("h") else text)

        content = "\n\n".join(parts)
        if len(content) > 3500:
            content = content[:3500] + "\n\n...[обрезано]"

        return f"📄 Содержимое {url}:\n\n{content}"
    except Exception as e:
        return f"Ошибка чтения {url}: {e}"


@tool
def github_search(query: str) -> str:
    """Ищет репозитории на GitHub.
    Используй для технических тем — найдёт инструменты и библиотеки.

    Args:
        query: Название технологии или библиотеки
    """
    try:
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 5},
            timeout=10,
        )
        data = resp.json()
        if not data.get("items"):
            return f"GitHub: ничего по '{query}'"

        output = f"💻 GitHub: '{query}'\n\n"
        for i, repo in enumerate(data["items"], 1):
            output += f"{i}. {repo['full_name']} ⭐{repo['stargazers_count']}\n"
            output += f"   {repo.get('description', 'N/A')}\n"
            output += f"   Язык: {repo.get('language', 'N/A')}\n\n"
        return output
    except Exception as e:
        return f"Ошибка GitHub: {e}"


@tool
def add_note(section: str, content: str) -> str:
    """Добавляет заметку по ходу исследования.
    Используй для сохранения ключевых фактов и данных.

    Args:
        section: Раздел заметки (например 'Основы', 'Тренды')
        content: Ключевые факты или цитаты для сохранения
    """
    notes_path = os.path.join(os.path.dirname(__file__), "output", "_notes.json")
    os.makedirs(os.path.dirname(notes_path), exist_ok=True)

    try:
        notes = {}
        if os.path.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as f:
                notes = json.load(f)

        notes.setdefault(section, []).append(content)

        with open(notes_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)

        total = sum(len(v) for v in notes.values())
        return f"✅ Заметка добавлена в '{section}' (всего заметок: {total})"
    except Exception as e:
        return f"Ошибка: {e}"


@tool
def save_report(filename: str, content: str) -> str:
    """Сохраняет финальный отчёт исследования в markdown файл.
    Используй только когда отчёт полностью готов.

    Args:
        filename: Имя файла (например, 'research_ai_2025.md')
        content: Полный текст отчёта в markdown формате
    """
    try:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        word_count = len(content.split())
        return f"✅ Отчёт сохранён!\n   Файл: {filepath}\n   Слов: {word_count}"
    except Exception as e:
        return f"Ошибка сохранения: {e}"


tools = [search_web, search_news, wikipedia_search, scrape_page, github_search, add_note, save_report]

print("✅ Инструменты определены:")
for t in tools:
    print(f"   🔧 {t.name}")
print()

# ============================================================
# Фаза 2: Планировщик
# ============================================================

print("=" * 60)
print("🗺️  ФАЗА 2: ПЛАНИРОВЩИК ИССЛЕДОВАНИЯ")
print("=" * 60)
print()

plan_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты планировщик исследований. Составь структурированный план.

Верни ТОЛЬКО JSON (без markdown):
{{
  "topic": "точное название темы",
  "research_goal": "цель исследования одним предложением",
  "subtopics": [
    {{
      "name": "название подтемы",
      "description": "что изучаем",
      "search_queries": ["запрос 1 на англ.", "запрос 2 на англ."],
      "wikipedia_topic": "тема для Wikipedia"
    }}
  ],
  "report_sections": ["Введение", "Раздел 1", "Раздел 2", "Тренды", "Заключение"],
  "filename": "research_topic_name.md"
}}

Создай 3-4 подтемы.""",
    ),
    ("human", "Составь план исследования: {topic}"),
])

plan_chain = plan_prompt | llm | JsonOutputParser()


def create_plan(topic: str) -> dict:
    print(f"🗺️  Создаю план: '{topic}'...")
    plan = plan_chain.invoke({"topic": topic})
    print(f"✅ План готов: {len(plan['subtopics'])} подтем")
    return plan


def plan_to_prompt(plan: dict) -> str:
    """Преобразует план в задание для агента."""
    text = f"ТЕМА: {plan['topic']}\nЦЕЛЬ: {plan['research_goal']}\n\nПОДТЕМЫ:\n"
    for i, st in enumerate(plan["subtopics"], 1):
        text += f"\n{i}. {st['name']} — {st['description']}\n"
        text += f"   Wikipedia: {st['wikipedia_topic']}\n"
        text += f"   Запросы: {', '.join(st['search_queries'])}\n"
    text += f"\nСТРУКТУРА ОТЧЁТА: {' → '.join(plan['report_sections'])}"
    text += f"\nФАЙЛ ДЛЯ СОХРАНЕНИЯ: {plan.get('filename', 'research_report.md')}"
    return text


# ============================================================
# Фаза 3: Агент-исследователь
# ============================================================

print("=" * 60)
print("🤖 ФАЗА 3: АГЕНТ-ИССЛЕДОВАТЕЛЬ")
print("=" * 60)
print()

SYSTEM_PROMPT = """Ты профессиональный AI-исследователь. Тебе дан план исследования.

ПРОЦЕСС РАБОТЫ:
1. Для каждой подтемы из плана:
   - Вызови wikipedia_search для базовых определений
   - Вызови search_web с предложенными запросами
   - Если нужны детали — используй scrape_page для конкретных страниц
   - Добавляй ключевые факты через add_note (section = название подтемы)

2. После всех подтем:
   - Используй search_news для актуальных трендов
   - Для технических тем — github_search для популярных инструментов

3. ОБЯЗАТЕЛЬНО в конце:
   - Составь ПОЛНЫЙ структурированный отчёт на РУССКОМ языке
   - Отчёт должен быть исчерпывающим (не менее 500 слов)
   - Сохрани через save_report с именем файла из плана

ФОРМАТ ОТЧЁТА (строго markdown):
```
# [Заголовок темы]

> *Дата исследования: [дата] | Источников изучено: [N]*

## Введение
[2-3 параграфа вводной информации]

## [Раздел 1]
[подробное описание]

## [Раздел 2]
[подробное описание]

## Современные тренды
[актуальное состояние в 2025 году]

## Практические инструменты
[инструменты, библиотеки, проекты — если применимо]

## Заключение
[ключевые выводы, 3-5 пунктов]

## Источники
- Wikipedia: [статьи]
- Веб: [найденные URL]
```"""

researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, researcher_prompt)
researcher = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=20,
    max_execution_time=300,
)

print("✅ Агент-исследователь создан")
print(f"   Инструментов: {len(tools)}")
print(f"   Max итераций: 20")
print()

# ============================================================
# Фаза 4: Полный пайплайн
# ============================================================

print("=" * 60)
print("🚀 ФАЗА 4: ПОЛНЫЙ ИССЛЕДОВАТЕЛЬСКИЙ ПАЙПЛАЙН")
print("=" * 60)
print()


def deep_research(topic: str) -> str:
    """Проводит полноценное углублённое исследование темы."""

    print(f"\n{'=' * 60}")
    print(f"🔬 ТЕМА: {topic}")
    print(f"{'=' * 60}")
    start = time.time()

    # Шаг 1: Планирование
    print("\n📍 ШАГ 1: ПЛАНИРОВАНИЕ")
    plan = create_plan(topic)

    print(f"   Подтемы: {', '.join(st['name'] for st in plan['subtopics'])}")
    print(f"   Разделы отчёта: {' → '.join(plan['report_sections'])}")

    # Сбрасываем заметки
    notes_path = os.path.join(os.path.dirname(__file__), "output", "_notes.json")
    if os.path.exists(notes_path):
        os.remove(notes_path)

    # Шаг 2: Агент собирает данные
    print("\n📍 ШАГ 2: СБОР ДАННЫХ")
    print("─" * 60)

    task = (
        f"Проведи исследование по плану:\n\n{plan_to_prompt(plan)}\n\n"
        f"Изучи все подтемы, добавляй заметки, затем составь полный отчёт на русском и сохрани файл."
    )

    result = researcher.invoke({"input": task})

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"✅ ИССЛЕДОВАНИЕ ЗАВЕРШЕНО за {elapsed:.0f}с")
    print(f"{'=' * 60}")

    return result["output"]


# ============================================================
# Запуск
# ============================================================

if __name__ == "__main__":
    # Выбор темы
    TOPIC = "deep research агенты и их будущее"

    summary = deep_research(TOPIC)

    print("\n🤖 Резюме агента:")
    print("─" * 60)
    print(summary)
    print()
    print("📁 Отчёт сохранён в папке Lecture_18/output/")

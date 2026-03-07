"""
Лекция 19: Трейсинг цепочек и кастомных функций
=================================================
Глубокое погружение в трейсинг:
  - Автоматический трейсинг цепочек LangChain
  - Ручной трейсинг через @traceable
  - Вложенные трейсы (parent -> child)
  - Просмотр промпт-ответ пар в LangSmith
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Lecture-19-Tracing"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. Автоматический трейсинг цепочки
# ============================================================

print("=" * 60)
print("1. АВТОМАТИЧЕСКИЙ ТРЕЙСИНГ ЦЕПОЧКИ")
print("=" * 60)
print()
print("Каждый компонент LangChain автоматически трейсится:")
print("  ChatPromptTemplate -> ChatOpenAI -> StrOutputParser")
print()

# Цепочка: prompt -> LLM -> parser
summarize_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Ты пишешь краткие резюме на русском языке."),
        ("human", "Напиши резюме этого текста в 2-3 предложения:\n\n{text}"),
    ])
    | llm
    | StrOutputParser()
)

text = """
LangSmith — это платформа для наблюдаемости LLM-приложений, разработанная
компанией LangChain. Она позволяет отслеживать каждый вызов языковой модели,
включая промпты, ответы, количество токенов и время выполнения. LangSmith
поддерживает трейсинг цепочек и агентов, что помогает разработчикам находить
и исправлять ошибки в сложных LLM-пайплайнах. Платформа также предоставляет
инструменты для оценки качества ответов модели с помощью датасетов
и автоматических метрик.
"""

print("  Запускаем цепочку суммаризации...")
summary = summarize_chain.invoke({"text": text})
print(f"  Резюме: {summary}")
print()
print("  В LangSmith вы увидите:")
print("  RunnableSequence")
print("    ├── ChatPromptTemplate  (вход: text)")
print("    ├── ChatOpenAI          (промпт -> ответ)")
print("    └── StrOutputParser     (выход: строка)")
print()

# ============================================================
# 2. Несколько цепочек подряд
# ============================================================

print("=" * 60)
print("2. НЕСКОЛЬКО ЦЕПОЧЕК ПОДРЯД")
print("=" * 60)
print()

# Цепочка 1: генерация
generate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Ты генерируешь факты о технологиях на русском."),
        ("human", "Назови 3 интересных факта о {topic}."),
    ])
    | llm
    | StrOutputParser()
)

# Цепочка 2: перевод
translate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Ты переводчик. Переведи текст на английский."),
        ("human", "{text}"),
    ])
    | llm
    | StrOutputParser()
)

print("  Цепочка 1: генерация фактов...")
facts = generate_chain.invoke({"topic": "LangSmith"})
print(f"  Факты: {facts[:150]}...")
print()

print("  Цепочка 2: перевод на английский...")
translated = translate_chain.invoke({"text": facts})
print(f"  Перевод: {translated[:150]}...")
print()
print("  -> В LangSmith: 2 отдельных трейса, каждый со своими шагами")
print()

# ============================================================
# 3. @traceable — ручной трейсинг своих функций
# ============================================================

print("=" * 60)
print("3. @TRACEABLE — ТРЕЙСИНГ СВОИХ ФУНКЦИЙ")
print("=" * 60)
print()
print("  @traceable позволяет трейсить ЛЮБУЮ Python-функцию,")
print("  не только LangChain компоненты.")
print()


@traceable(name="fetch_weather")
def fetch_weather(city: str) -> dict:
    """Получает погоду — эта функция трейсится в LangSmith."""
    r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
    data = r.json()["current_condition"][0]
    return {
        "city": city,
        "temp": data["temp_C"],
        "description": data["weatherDesc"][0]["value"],
    }


@traceable(name="format_weather")
def format_weather(weather: dict) -> str:
    """Форматирует данные — тоже трейсится."""
    return f"Погода в {weather['city']}: {weather['temp']}C, {weather['description']}"


@traceable(name="weather_report")
def weather_report(city: str) -> str:
    """Полный пайплайн — parent трейс, содержит child трейсы."""
    weather = fetch_weather(city)       # child trace 1
    formatted = format_weather(weather)  # child trace 2
    return formatted


print("  Запускаем weather_report('Astana')...")
result = weather_report("Astana")
print(f"  Результат: {result}")
print()
print("  В LangSmith вы увидите ВЛОЖЕННЫЕ трейсы:")
print("  weather_report")
print("    ├── fetch_weather    (HTTP запрос)")
print("    └── format_weather   (форматирование)")
print()

# ============================================================
# 4. Комбинация @traceable + LangChain
# ============================================================

print("=" * 60)
print("4. КОМБИНАЦИЯ @TRACEABLE + LANGCHAIN")
print("=" * 60)
print()


@traceable(name="research_topic")
def research_topic(topic: str) -> str:
    """
    Полный пайплайн: получение данных + LLM анализ.
    В LangSmith покажет и HTTP-вызовы, и LLM-вызовы.
    """
    # Шаг 1: получаем данные из Wikipedia
    data = fetch_wikipedia(topic)

    # Шаг 2: LLM анализирует
    analysis = analyze_chain.invoke({"topic": topic, "data": data})

    return analysis


@traceable(name="fetch_wikipedia")
def fetch_wikipedia(topic: str) -> str:
    """Получает summary из Wikipedia."""
    headers = {"User-Agent": "TechOrda-Education/1.0"}
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 200:
        return r.json().get("extract", "")[:500]
    return f"Статья '{topic}' не найдена"


analyze_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Ты аналитик. Проанализируй данные и напиши краткий вывод на русском."),
        ("human", "Тема: {topic}\n\nДанные:\n{data}\n\nНапиши анализ в 3 предложения."),
    ])
    | llm
    | StrOutputParser()
)

print("  Запускаем research_topic('Artificial intelligence')...")
result = research_topic("Artificial intelligence")
print(f"  Результат: {result}")
print()
print("  В LangSmith:")
print("  research_topic")
print("    ├── fetch_wikipedia        (HTTP к Wikipedia)")
print("    └── RunnableSequence       (LangChain цепочка)")
print("        ├── ChatPromptTemplate")
print("        ├── ChatOpenAI")
print("        └── StrOutputParser")
print()

# ============================================================
# 5. Добавляем метаданные к трейсам
# ============================================================

print("=" * 60)
print("5. МЕТАДАННЫЕ И ТЕГИ")
print("=" * 60)
print()

from langchain_core.runnables import RunnableConfig

# Теги и метаданные помогают фильтровать трейсы в дашборде
configs = [
    RunnableConfig(
        tags=["experiment-A", "temperature-0"],
        metadata={"version": "v1", "prompt_variant": "concise"},
        run_name="Experiment-A-Concise",
    ),
    RunnableConfig(
        tags=["experiment-B", "temperature-0"],
        metadata={"version": "v1", "prompt_variant": "detailed"},
        run_name="Experiment-B-Detailed",
    ),
]

concise_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Отвечай ОДНИМ предложением на русском."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

detailed_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Дай подробный ответ (5-7 предложений) на русском."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

question = "Что такое наблюдаемость в программировании?"

print("  Эксперимент A (краткий):")
r1 = concise_chain.invoke({"question": question}, config=configs[0])
print(f"  {r1[:100]}")
print()

print("  Эксперимент B (подробный):")
r2 = detailed_chain.invoke({"question": question}, config=configs[1])
print(f"  {r2[:150]}...")
print()

print("  В LangSmith можно фильтровать:")
print("    - По тегам: experiment-A vs experiment-B")
print("    - По метаданным: prompt_variant = concise/detailed")
print("    - По имени: Experiment-A-Concise / Experiment-B-Detailed")
print("    -> Удобно для A/B тестирования промптов!")

# ============================================================
# Итоги
# ============================================================

print()
print("=" * 60)
print("ИТОГИ: ТРЕЙСИНГ")
print("=" * 60)
print()
print("  Автоматический трейсинг:")
print("    - LangChain цепочки трейсятся сами")
print("    - Каждый шаг виден отдельно")
print()
print("  Ручной трейсинг (@traceable):")
print("    - Любая Python-функция может быть трейсом")
print("    - Вложенные вызовы = вложенные трейсы")
print()
print("  Метаданные:")
print("    - tags — для фильтрации и группировки")
print("    - metadata — для хранения контекста")
print("    - run_name — для именования трейсов")
print()
print("  Следующий шаг: 03_debugging_agents.py")

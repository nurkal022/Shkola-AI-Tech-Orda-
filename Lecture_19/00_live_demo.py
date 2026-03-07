"""
Лекция 19: Наблюдаемость и отладка с LangSmith
Live Demo — Версия для совместного написания
==============================================
Показываем как LangSmith трейсит каждый вызов LLM,
цепочки и агенты в реальном времени.
"""

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

# ─────────────────────────────────────────────
# Шаг 1: Настройка LangSmith
# ─────────────────────────────────────────────

# LangSmith автоматически трейсит все вызовы LangChain
# Нужно только выставить переменные окружения:

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Lecture-19-Demo"
# LANGCHAIN_API_KEY должен быть в .env

api_key = os.getenv("LANGCHAIN_API_KEY")
if not api_key:
    print("=" * 60)
    print("  LANGCHAIN_API_KEY не найден в .env!")
    print()
    print("  Как получить ключ:")
    print("  1. Зайдите на https://smith.langchain.com")
    print("  2. Settings -> API Keys -> Create API Key")
    print("  3. Добавьте в .env: LANGCHAIN_API_KEY=lsv2_...")
    print("=" * 60)
    exit(1)

print("=" * 60)
print("  LANGSMITH LIVE DEMO")
print("=" * 60)
print()
print(f"  Проект: {os.environ['LANGCHAIN_PROJECT']}")
print(f"  Ключ:   {api_key[:8]}...{api_key[-4:]}")
print(f"  Дашборд: https://smith.langchain.com")
print()

# ─────────────────────────────────────────────
# Шаг 2: Простой вызов LLM (трейсится автоматически)
# ─────────────────────────────────────────────

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

print("─" * 60)
print("  Шаг 2: Простой вызов LLM")
print("─" * 60)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Этот вызов автоматически появится в LangSmith
result = llm.invoke("Что такое LangSmith? Ответь одним предложением.")
print(f"  Ответ: {result.content}")
print("  -> Откройте LangSmith — вы увидите этот вызов в трейсах!")
print()

# ─────────────────────────────────────────────
# Шаг 3: Цепочка (Chain) — трейсит каждый шаг
# ─────────────────────────────────────────────

print("─" * 60)
print("  Шаг 3: Цепочка (Chain)")
print("─" * 60)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты эксперт по {topic}. Отвечай кратко на русском."),
    ("human", "{question}"),
])

chain = prompt | llm

# В LangSmith увидим: prompt -> LLM -> ответ
result = chain.invoke({
    "topic": "машинное обучение",
    "question": "Чем отличается supervised от unsupervised learning?",
})
print(f"  Ответ: {result.content[:200]}")
print("  -> В LangSmith видно: ChatPromptTemplate -> ChatOpenAI -> ответ")
print()

# ─────────────────────────────────────────────
# Шаг 4: Агент — трейсит все итерации
# ─────────────────────────────────────────────

print("─" * 60)
print("  Шаг 4: Агент с инструментами")
print("─" * 60)

import requests
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Получает погоду для города.
    Args:
        city: Город на английском (Astana, London)
    """
    try:
        r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        data = r.json()["current_condition"][0]
        return f"Погода в {city}: {data['temp_C']}C, {data['weatherDesc'][0]['value']}"
    except Exception as e:
        return f"Ошибка: {e}"


@tool
def calculate(expression: str) -> str:
    """Вычисляет математическое выражение.
    Args:
        expression: Выражение (например '2 + 2 * 3')
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Недопустимые символы"
        return str(eval(expression))
    except Exception as e:
        return f"Ошибка: {e}"


tools = [get_weather, calculate]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты помощник с доступом к инструментам. Отвечай на русском."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# В LangSmith увидим полный цикл агента:
# LLM думает -> вызывает tool -> получает результат -> финальный ответ
result = agent_executor.invoke({
    "input": "Какая погода в Астане? И сколько будет 25 * 4 + 17?"
})

print()
print(f"  Ответ: {result['output'][:300]}")
print()
print("  -> В LangSmith видно КАЖДУЮ итерацию агента:")
print("     1. LLM решает вызвать get_weather")
print("     2. Tool возвращает результат")
print("     3. LLM решает вызвать calculate")
print("     4. Tool возвращает результат")
print("     5. LLM формирует финальный ответ")
print()

# ─────────────────────────────────────────────
# Шаг 5: Метаданные и теги
# ─────────────────────────────────────────────

print("─" * 60)
print("  Шаг 5: Метаданные и теги для организации")
print("─" * 60)

from langchain_core.runnables import RunnableConfig

# Добавляем метаданные к вызову — видны в LangSmith
config = RunnableConfig(
    tags=["demo", "lecture-19"],
    metadata={
        "user_id": "student_001",
        "session": "live_demo",
        "lecture": 19,
    },
)

result = chain.invoke(
    {
        "topic": "LangSmith",
        "question": "Зачем нужна наблюдаемость в LLM-приложениях?",
    },
    config=config,
)

print(f"  Ответ: {result.content[:200]}")
print()
print("  -> В LangSmith этот трейс помечен тегами: demo, lecture-19")
print("  -> Метаданные: user_id=student_001, session=live_demo")
print("  -> Можно фильтровать трейсы по тегам и метаданным!")

# ─────────────────────────────────────────────
# Итоги
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("  ИТОГИ DEMO")
print("=" * 60)
print()
print("  Что мы увидели в LangSmith:")
print("  1. Простой вызов LLM — промпт и ответ")
print("  2. Цепочка — каждый шаг (prompt -> LLM)")
print("  3. Агент — все итерации (think -> act -> observe)")
print("  4. Теги и метаданные — организация трейсов")
print()
print("  Дашборд: https://smith.langchain.com")
print("  Проект:  Lecture-19-Demo")

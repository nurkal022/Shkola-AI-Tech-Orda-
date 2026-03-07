"""
Лекция 19: Отладка агентов с LangSmith
=======================================
Как использовать LangSmith для поиска и исправления проблем:
  - Отслеживание итераций агента
  - Поиск причин ошибок и зацикливания
  - Анализ стоимости и времени
  - Callbacks для мониторинга в реальном времени
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Lecture-19-Debugging"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. Агент с намеренными проблемами
# ============================================================

print("=" * 60)
print("1. АГЕНТ С ПРОБЛЕМАМИ ДЛЯ ОТЛАДКИ")
print("=" * 60)
print()
print("  Создаём агента, который может ломаться.")
print("  LangSmith поможет найти причину!")
print()


@tool
def search_database(query: str) -> str:
    """Ищет информацию в базе данных по запросу.
    Args:
        query: Поисковый запрос
    """
    # Эмулируем базу данных
    db = {
        "python": "Python — высокоуровневый язык программирования. Создан Гвидо ван Россумом в 1991 году.",
        "langsmith": "LangSmith — платформа для мониторинга LLM-приложений от LangChain.",
        "langchain": "LangChain — фреймворк для разработки приложений на базе LLM.",
    }
    query_lower = query.lower()
    for key, value in db.items():
        if key in query_lower:
            return value
    return f"Ничего не найдено по запросу: '{query}'"


@tool
def calculator(expression: str) -> str:
    """Вычисляет математическое выражение.
    Args:
        expression: Математическое выражение (например '2 + 2')
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Ошибка: недопустимые символы в выражении"
        result = eval(expression)
        return f"Результат: {result}"
    except ZeroDivisionError:
        return "Ошибка: деление на ноль!"
    except Exception as e:
        return f"Ошибка вычисления: {e}"


@tool
def unstable_api(query: str) -> str:
    """Вызывает нестабильный API который иногда падает.
    Args:
        query: Запрос к API
    """
    import random
    # 50% шанс ошибки — для демонстрации отладки
    if random.random() < 0.5:
        return f"API Error 503: Service Unavailable для '{query}'"
    return f"API ответ: данные по запросу '{query}' получены успешно."


tools = [search_database, calculator, unstable_api]

agent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты исследовательский помощник с доступом к инструментам.
Отвечай на русском языке.
Если инструмент вернул ошибку — попробуй другой подход.
НЕ пробуй один и тот же инструмент больше 2 раз.""",
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    return_intermediate_steps=True,  # Возвращаем промежуточные шаги!
)

# ============================================================
# 2. Сценарий: успешный запрос
# ============================================================

print("=" * 60)
print("2. СЦЕНАРИЙ: УСПЕШНЫЙ ЗАПРОС")
print("=" * 60)
print()

config = RunnableConfig(
    tags=["scenario-success"],
    run_name="Successful-Query",
)

result = agent_executor.invoke(
    {"input": "Что такое Python? И сколько будет 15 * 7 + 3?"},
    config=config,
)

print()
print(f"  Ответ: {result['output'][:300]}")
print(f"  Шагов агента: {len(result['intermediate_steps'])}")
print()
print("  В LangSmith (Successful-Query):")
print("    1. LLM -> выбирает search_database")
print("    2. search_database -> результат")
print("    3. LLM -> выбирает calculator")
print("    4. calculator -> результат")
print("    5. LLM -> финальный ответ")
print()

# ============================================================
# 3. Сценарий: нестабильный API
# ============================================================

print("=" * 60)
print("3. СЦЕНАРИЙ: НЕСТАБИЛЬНЫЙ API")
print("=" * 60)
print()

config = RunnableConfig(
    tags=["scenario-unstable"],
    run_name="Unstable-API-Call",
)

result = agent_executor.invoke(
    {"input": "Получи данные через unstable_api по запросу 'AI trends'."},
    config=config,
)

print()
print(f"  Ответ: {result['output'][:300]}")
print(f"  Шагов агента: {len(result['intermediate_steps'])}")
for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
    tool_name = action.tool
    print(f"    Шаг {i}: {tool_name} -> {observation[:80]}")
print()
print("  В LangSmith видно:")
print("    - Какие инструменты вызывались")
print("    - Какие вернули ошибку")
print("    - Сколько retry сделал агент")
print()

# ============================================================
# 4. Анализ промежуточных шагов
# ============================================================

print("=" * 60)
print("4. АНАЛИЗ ПРОМЕЖУТОЧНЫХ ШАГОВ")
print("=" * 60)
print()


@traceable(name="analyze_agent_run")
def analyze_agent_run(query: str) -> dict:
    """Запускает агента и анализирует его поведение."""

    start_time = time.time()

    result = agent_executor.invoke(
        {"input": query},
        config=RunnableConfig(
            tags=["analysis"],
            metadata={"query": query},
            run_name=f"Analysis: {query[:30]}",
        ),
    )

    elapsed = time.time() - start_time

    # Анализируем промежуточные шаги
    steps = result["intermediate_steps"]
    tools_used = [action.tool for action, _ in steps]
    errors = [obs for _, obs in steps if "ошибка" in obs.lower() or "error" in obs.lower()]

    analysis = {
        "query": query,
        "answer": result["output"][:200],
        "total_steps": len(steps),
        "tools_used": tools_used,
        "errors": errors,
        "time_seconds": round(elapsed, 2),
    }

    return analysis


# Анализируем несколько запросов
queries = [
    "Что такое LangChain?",
    "Вычисли 100 / 0",
    "Найди информацию о quantum computing через unstable_api",
]

print("  Анализируем 3 запроса к агенту...")
print()

for query in queries:
    print(f"  Запрос: {query}")
    analysis = analyze_agent_run(query)
    print(f"    Шагов: {analysis['total_steps']}")
    print(f"    Инструменты: {analysis['tools_used']}")
    print(f"    Ошибки: {len(analysis['errors'])}")
    print(f"    Время: {analysis['time_seconds']}с")
    print()

print("  В LangSmith все 3 запроса под тегом 'analysis'")
print("  Можно сравнить: время, шаги, ошибки")
print()

# ============================================================
# 5. Callbacks для мониторинга в реальном времени
# ============================================================

print("=" * 60)
print("5. CALLBACKS ДЛЯ МОНИТОРИНГА")
print("=" * 60)
print()

from langchain_core.callbacks import BaseCallbackHandler


class MonitorCallback(BaseCallbackHandler):
    """Кастомный callback для мониторинга агента в реальном времени."""

    def __init__(self):
        self.steps = 0
        self.tokens_total = 0
        self.start_time = None

    def on_chain_start(self, serialized, inputs, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.steps += 1
        tool_name = serialized.get("name", "unknown")
        print(f"    [Monitor] Шаг {self.steps}: вызываю {tool_name}")

    def on_tool_end(self, output, **kwargs):
        status = "OK" if "ошибка" not in output.lower() and "error" not in output.lower() else "ERROR"
        print(f"    [Monitor] Результат: {status} ({output[:60]}...)")

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.tokens_total += usage.get("total_tokens", 0)

    def on_chain_end(self, outputs, **kwargs):
        pass

    def report(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\n    [Monitor] Итого: {self.steps} шагов, {elapsed:.1f}с")


monitor = MonitorCallback()

print("  Запуск агента с MonitorCallback:")
print()

result = agent_executor.invoke(
    {"input": "Найди информацию о Python и вычисли 42 * 42."},
    config={"callbacks": [monitor], "tags": ["with-monitor"]},
)

monitor.report()
print(f"\n  Ответ: {result['output'][:200]}")

# ============================================================
# 6. Чеклист отладки
# ============================================================

print()
print("=" * 60)
print("6. ЧЕКЛИСТ ОТЛАДКИ С LANGSMITH")
print("=" * 60)
print()
print("  Агент дал неправильный ответ?")
print("    1. Откройте трейс в LangSmith")
print("    2. Посмотрите системный промпт — корректен ли он?")
print("    3. Какие инструменты выбрал агент?")
print("    4. Что вернули инструменты?")
print("    5. Как LLM интерпретировал результат?")
print()
print("  Агент зациклился?")
print("    1. Проверьте max_iterations")
print("    2. Посмотрите на какой паре tool call / observation цикл")
print("    3. Проверьте docstring инструментов")
print()
print("  Агент работает медленно?")
print("    1. Посмотрите latency каждого шага")
print("    2. Найдите узкое место (LLM? API? Scraping?)")
print("    3. Оптимизируйте самый медленный шаг")
print()
print("  Слишком дорого?")
print("    1. Фильтруйте по 'Total Tokens' в LangSmith")
print("    2. Найдите запросы с самым большим потреблением")
print("    3. Оптимизируйте промпт или используйте меньшую модель")
print()
print("  Следующий шаг: 04_evaluation.py")

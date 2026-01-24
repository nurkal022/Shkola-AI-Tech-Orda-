"""
Лекция 13: Live Demo - ФИНАЛЬНАЯ ВЕРСИЯ
=========================================
Полная рабочая версия ReAct агента для референса
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool

load_dotenv()

# ============================================================
# 1. ИНСТРУМЕНТЫ
# ============================================================

@tool
def calculate(expression: str) -> str:
    """Вычисляет математическое выражение."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool
def get_current_date() -> str:
    """Возвращает текущую дату."""
    from datetime import datetime
    return f"Текущая дата: {datetime.now().strftime('%d.%m.%Y')}"


tools = [calculate, get_current_date]

# ============================================================
# 2. LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 3. ReAct ПРОМПТ
# ============================================================

react_prompt = hub.pull("hwchase17/react")

# ============================================================
# 4. СОЗДАНИЕ АГЕНТА
# ============================================================

agent = create_react_agent(llm, tools, react_prompt)

# ============================================================
# 5. EXECUTOR
# ============================================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# ============================================================
# 6. ИСПОЛЬЗОВАНИЕ
# ============================================================

if __name__ == "__main__":
    result = agent_executor.invoke({"input": "Сколько будет 25 * 17?"})
    print(f"\nОтвет: {result['output']}")

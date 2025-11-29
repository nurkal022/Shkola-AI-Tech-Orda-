"""
Пример 8: Агент с инструментами (Tools)
Агент сам решает когда и какую функцию вызвать

КАК МОДЕЛЬ УЗНАЁТ ОБ ИНСТРУМЕНТАХ?
==================================

1. LangChain парсит каждый @tool декоратор:
   - Имя функции: multiply
   - Параметры и типы: a: float, b: float
   - Описание из docstring: "Умножение двух чисел"

2. Формирует JSON-схему и отправляет в OpenAI API:
   {
     "tools": [{
       "type": "function",
       "function": {
         "name": "multiply",
         "description": "Умножение двух чисел",
         "parameters": {
           "type": "object",
           "properties": {
             "a": {"type": "number"},
             "b": {"type": "number"}
           }
         }
       }
     }]
   }

3. Модель возвращает решение вызвать функцию:
   {
     "tool_calls": [{
       "name": "multiply",
       "arguments": {"a": 7, "b": 8}
     }]
   }

4. LangChain выполняет функцию ЛОКАЛЬНО и результат отправляет обратно модели.

ВАЖНО: Модель НЕ выполняет код — она только РЕШАЕТ какую функцию вызвать 
и с какими аргументами. Это называется Function Calling (фича OpenAI API).
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Определяем инструменты с помощью декоратора @tool
@tool
def add(a: float, b: float) -> float:
    """Сложение двух чисел"""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Вычитание: a - b"""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Умножение двух чисел"""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Деление: a / b"""
    if b == 0:
        return "Ошибка: деление на ноль"
    return a / b


# Список инструментов
tools = [add, subtract, multiply, divide]

# Промпт для агента
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты калькулятор-помощник. 
Используй доступные инструменты для вычислений.
Всегда показывай ход решения.
Отвечай на русском языке."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Создаем агента
agent = create_openai_tools_agent(llm, tools, prompt)

# Создаем AgentExecutor для выполнения агента
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def ask_agent(question: str):
    """Задать вопрос агенту"""
    print(f"Вопрос: {question}\n")
    
    result = agent_executor.invoke({"input": question})
    
    print(f"\n🤖 Ответ: {result['output']}\n")
    print("="*50 + "\n")


# Примеры использования
print("=== Агент-калькулятор ===\n")

ask_agent("Сколько будет 25 + 17?")

# ask_agent("Раздели 100 на 4")

# ask_agent("Умножь 7 на 8, затем прибавь 15")

# ask_agent("Сколько будет (50 - 20) * 3?")


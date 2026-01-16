"""
Лекция 12: Calculator Agent
============================
Простейший агент с одним инструментом - калькулятором.
Демонстрирует базовую концепцию агентов: LLM + Tools.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# Проверка API ключа
# ============================================================
if not os.getenv("OPENAI_API_KEY"):
    exit(1)

# ============================================================
# Создание инструмента - Калькулятор
# ============================================================

@tool
def calculate(expression: str) -> str:
    """
    Вычисляет математическое выражение.
    Полезен для выполнения математических вычислений.
    Принимает математическое выражение в виде строки.
    Примеры: "25 * 17", "100 / 4 + 75", "(50 + 30) * 2"
    
    Args:
        expression: Математическое выражение (например, "25 * 17 + 33")
    
    Returns:
        Результат вычисления в виде строки
    """
    try:
        # Безопасное вычисление (только математические операции)
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка вычисления: {str(e)}"

calculator_tool = calculate

print("   ✅ Инструмент создан: Calculator")
print(f"   📝 Описание: {calculator_tool.description[:80]}...\n")



@tool
def calculate_letters_count(text: str) -> str:
    """
    Подсчитывает количество букв в тексте.
    Полезен для подсчета количества букв в тексте.
    Принимает текст в виде строки.
    Примеры: "Hello, world!", "Привет, мир!"
    """
    return f"Количество букв: {len(text)}"

calculator_tool_letters_count = calculate_letters_count

# ============================================================
# Создание агента
# ============================================================

# LLM для агента
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Промпт для агента
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты помощник-калькулятор. 
Используй доступные инструменты для вычислений.
Всегда показывай ход решения.
Отвечай на русском языке."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Создаём агента
agent = create_openai_tools_agent(llm, [calculator_tool, calculator_tool_letters_count], prompt)

# Создаём executor (выполняет агента)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[calculator_tool, calculator_tool_letters_count],
    verbose=True,  # Показываем "мысли" агента
    handle_parsing_errors=True,  # Обработка ошибок парсинга
)

print("   ✅ Агент создан")
print("   🧠 Модель: gpt-4o-mini")
print("   🔧 Инструменты: Calculator")
print("   📊 Verbose mode: включён (видим мысли агента)\n")


# ============================================================
# Показываем что отправляется в LLM
# ============================================================
print("="*60)
print("📤 ЧТО ОТПРАВЛЯЕТСЯ В LLM")
print("="*60)

print("\n1️⃣ SYSTEM PROMPT:")
print("─" * 60)
system_prompt = """Ты помощник-калькулятор. 
Используй доступные инструменты для вычислений.
Всегда показывай ход решения.
Отвечай на русском языке."""
print(system_prompt)

print("\n2️⃣ ОПИСАНИЕ ИНСТРУМЕНТОВ (Tools Schema):")
print("─" * 60)
import json
tool_schema = {
    "type": "function",
    "function": {
        "name": calculator_tool.name,
        "description": calculator_tool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Математическое выражение для вычисления"
                }
            },
            "required": ["expression"]
        }
    }
}
print(json.dumps(tool_schema, indent=2, ensure_ascii=False))

print("\n3️⃣ ПРИМЕР ЗАПРОСА К LLM:")
print("─" * 60)
example_query = "Сколько будет 25 * 17?"
print(f"""
Messages отправляемые в LLM:

[
  {{
    "role": "system",
    "content": "{system_prompt}"
  }},
  {{
    "role": "user",
    "content": "{example_query}"
  }},
  {{
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {{
        "id": "call_abc123",
        "type": "function",
        "function": {{
          "name": "{calculator_tool.name}",
          "arguments": "{{\\"expression\\": \\"25 * 17\\"}}"
        }}
      }},
      {{
        "id": "call_abc123",
        "type": "function",
        "function": {{
          "name": "{calculator_tool_letters_count.name}",
          "arguments": "{{\\"text\\": \\"Hello, world!\\"}}"
        }}
      }}
    ]
  }},
  {{
    "role": "tool",
    "tool_call_id": "call_abc123",
    "name": "{calculator_tool.name}",
    "content": "Результат: 425"
  }},
  {{
    "role": "assistant",
    "content": "25 * 17 = 425"
  }}
]

Tools (доступные функции):
{json.dumps([tool_schema], indent=2, ensure_ascii=False)}
""")

print("\n4️⃣ КАК LLM ВИДИТ ИНСТРУМЕНТЫ:")
print("─" * 60)
print(f"""
LLM получает список доступных функций:

Доступные функции:
1. {calculator_tool.name}
   Описание: {calculator_tool.description}
   Параметры:
     - expression (string): Математическое выражение

LLM анализирует запрос пользователя и решает:
  • Нужен ли инструмент?
  • Какой инструмент использовать?
  • Какие параметры передать?
""")

print("\n" + "="*60)


# ============================================================
# Интерактивный режим чата
# ============================================================
print("="*60)
print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ ЧАТА")
print("="*60)
print("""
   Введите ваш запрос (математические вычисления).
   Примеры запросов:
   • Сколько будет 25 * 17?
""")
print("="*60)

while True:
    try:
        # Получаем запрос от пользователя
        user_input = input("\n🤔 Ваш запрос: ").strip()
        
        # Проверка на выход
        if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
            print("\n👋 До свидания!")
            break
        
        if not user_input:
            print("⚠️ Пустой запрос, попробуйте ещё раз")
            continue
        
        print(f"\n{'─'*60}")
        print(f"🔍 Обработка запроса: «{user_input}»")
        print(f"{'─'*60}\n")
        
        # Показываем что отправляется в LLM
        print("📤 ЧТО ОТПРАВЛЯЕТСЯ В LLM:")
        print("─" * 60)
        
        # Получаем реальную схему инструмента
        tool_schema_dict = calculator_tool.args_schema.schema() if hasattr(calculator_tool, 'args_schema') else {}
        
        # Формируем полный запрос
        full_request = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": calculator_tool.name,
                        "description": calculator_tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Математическое выражение для вычисления"
                                }
                            },
                            "required": ["expression"]
                        }
                    }
                }
            ],
            "temperature": 0
        }
        
        print("\n1️⃣ SYSTEM MESSAGE:")
        print(f"   Role: system")
        print(f"   Content: {system_prompt}")
        
        print("\n2️⃣ USER MESSAGE:")
        print(f"   Role: user")
        print(f"   Content: «{user_input}»")
        
        print("\n3️⃣ TOOLS (доступные функции):")
        print(f"   Tool: {calculator_tool.name}")
        print(f"   Description: {calculator_tool.description}")
        print(f"   Parameters:")
        print(f"     - expression (string): Математическое выражение")
        
        print("\n4️⃣ ПОЛНЫЙ JSON ЗАПРОС К OpenAI API:")
        print("─" * 60)
        import json
        print(json.dumps(full_request, indent=2, ensure_ascii=False))
        
        print("\n5️⃣ КАК LLM ОБРАБАТЫВАЕТ:")
        print("─" * 60)
        print("""
   LLM получает:
   1. System prompt - инструкции как себя вести
   2. User message - запрос пользователя
   3. Tools list - список доступных функций
   
   LLM анализирует:
   • Понимает что это математический вопрос
   • Видит доступный инструмент Calculator
   • Решает вызвать функцию calculate
   • Формирует tool_call с параметрами
        """)
        
        print("─" * 60)
        print("⏳ Ожидание ответа от LLM...\n")
        
        # Выполняем запрос
        result = agent_executor.invoke({"input": user_input})
        
        print(f"\n{'─'*60}")
        print(f"✅ ФИНАЛЬНЫЙ ОТВЕТ:")
        print(f"{'─'*60}")
        print(f"{result['output']}\n")
        
    except KeyboardInterrupt:
        print("\n\n👋 До свидания!")
        break
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("Попробуйте ещё раз или введите 'exit' для выхода")


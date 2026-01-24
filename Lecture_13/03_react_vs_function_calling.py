"""
Лекция 13: Сравнение ReAct и OpenAI Function Calling
=====================================================
Демонстрация разницы между двумя подходами на одной и той же задаче.
"""

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# Создание инструментов
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
    """Возвращает текущую дату в формате ДД.ММ.ГГГГ."""
    from datetime import datetime
    return f"Текущая дата: {datetime.now().strftime('%d.%m.%Y')}"


tools = [calculate, get_current_date]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# Подход 1: ReAct (Prompting)
# ============================================================

print("="*60)
print("🔵 ПОДХОД 1: ReAct (Prompting)")
print("="*60)
print()

react_prompt = hub.pull("hwchase17/react")
react_agent = create_react_agent(llm, tools, react_prompt)
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

print("   📝 Использует: Текстовый промпт (hwchase17/react)")
print("   🔧 Механизм: LLM генерирует текст в формате Thought/Action/Observation")
print("   📊 Прозрачность: ✅ Видим все рассуждения")
print()

# ============================================================
# Подход 2: OpenAI Function Calling
# ============================================================

print("="*60)
print("🟢 ПОДХОД 2: OpenAI Function Calling")
print("="*60)
print()

function_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты помощник. Используй доступные инструменты для вычислений."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

function_calling_agent = create_openai_tools_agent(llm, tools, function_calling_prompt)
function_calling_executor = AgentExecutor(
    agent=function_calling_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

print("   📝 Использует: OpenAI Function Calling API")
print("   🔧 Механизм: LLM возвращает структурированный JSON (tool_calls)")
print("   📊 Прозрачность: ⚠️ Меньше видимости внутренних рассуждений")
print()

# ============================================================
# Сравнение на одной задаче
# ============================================================

test_queries = [
    "Сколько будет 25 * 17?",
    "Какая сегодня дата? Умножь день месяца на 10.",
]

for i, query in enumerate(test_queries, 1):
    print("="*60)
    print(f"📌 ТЕСТ {i}: {query}")
    print("="*60)
    print()
    
    # ReAct подход
    print("─" * 60)
    print("🔵 ReAct (Prompting):")
    print("─" * 60)
    print()
    try:
        react_result = react_executor.invoke({"input": query})
        print(f"\n✅ Ответ: {react_result['output']}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    print()
    
    # Function Calling подход
    print("─" * 60)
    print("🟢 OpenAI Function Calling:")
    print("─" * 60)
    print()
    try:
        fc_result = function_calling_executor.invoke({"input": query})
        print(f"\n✅ Ответ: {fc_result['output']}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    print()
    
    print("="*60)
    print()

# ============================================================
# Сравнительная таблица
# ============================================================

print("="*60)
print("📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
print("="*60)
print()

comparison_table = """
┌─────────────────────────┬──────────────────────┬─────────────────────────┐
│ Критерий                │ ReAct Prompting      │ OpenAI Function Calling │
├─────────────────────────┼──────────────────────┼─────────────────────────┤
│ Прозрачность            │ ✅ Видим все мысли   │ ⚠️ Меньше видимости      │
│                         │    (Thought/Action)  │    (только tool_calls)   │
├─────────────────────────┼──────────────────────┼─────────────────────────┤
│ Надёжность              │ ⚠️ Может ошибиться   │ ✅ Структурированный    │
│                         │    в формате         │    JSON, меньше ошибок   │
├─────────────────────────┼──────────────────────┼─────────────────────────┤
│ Совместимость           │ ✅ Любая LLM         │ ⚠️ Только OpenAI API    │
│                         │    (текстовый промпт) │    (gpt-3.5, gpt-4)      │
├─────────────────────────┼──────────────────────┼─────────────────────────┤
│ Отладка                 │ ✅ Легко понять      │ ❌ Сложнее отлаживать   │
│                         │    что пошло не так   │    (меньше информации)  │
├─────────────────────────┼──────────────────────┼─────────────────────────┤
│ Производительность      │ ⚠️ Больше токенов    │ ✅ Эффективнее          │
│                         │    (текст формата)   │    (структурированный)   │
├─────────────────────────┼──────────────────────┼─────────────────────────┤
│ Гибкость                │ ✅ Полный контроль   │ ⚠️ Ограничен API        │
│                         │    над промптом      │    OpenAI                │
└─────────────────────────┴──────────────────────┴─────────────────────────┘
"""

print(comparison_table)
print()

# ============================================================
# Рекомендации
# ============================================================

print("="*60)
print("💡 КОГДА ЧТО ИСПОЛЬЗОВАТЬ")
print("="*60)
print()

recommendations = """
🔵 ИСПОЛЬЗУЙ ReAct КОГДА:
   ✅ Нужна максимальная прозрачность (обучение, отладка)
   ✅ Работаете с не-OpenAI моделями (Claude, Llama, etc.)
   ✅ Нужен полный контроль над форматом рассуждений
   ✅ Важна обучаемость (студенты видят как агент думает)

🟢 ИСПОЛЬЗУЙ Function Calling КОГДА:
   ✅ Нужна максимальная надёжность
   ✅ Работаете только с OpenAI моделями
   ✅ Важна производительность (меньше токенов)
   ✅ Нужна простота интеграции

📌 В ПРОДАКШЕНЕ:
   • Function Calling - для надёжности
   • ReAct - для разработки и отладки
"""

print(recommendations)
print()

# ============================================================
# Интерактивное сравнение
# ============================================================

print("="*60)
print("💬 ИНТЕРАКТИВНОЕ СРАВНЕНИЕ")
print("="*60)
print("""
   Введите запрос и увидите как оба подхода обрабатывают его.
   Введите 'exit' для выхода.
""")
print("="*60)

while True:
    try:
        user_input = input("\n🤔 Ваш запрос: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
            print("\n👋 До свидания!")
            break
        
        if not user_input:
            print("⚠️ Пустой запрос, попробуйте ещё раз")
            continue
        
        print(f"\n{'='*60}")
        print(f"🔍 Обработка: «{user_input}»")
        print(f"{'='*60}\n")
        
        # ReAct
        print("─" * 60)
        print("🔵 ReAct (Prompting):")
        print("─" * 60)
        react_result = react_executor.invoke({"input": user_input})
        print(f"\n✅ Ответ: {react_result['output']}\n")
        
        # Function Calling
        print("─" * 60)
        print("🟢 OpenAI Function Calling:")
        print("─" * 60)
        fc_result = function_calling_executor.invoke({"input": user_input})
        print(f"\n✅ Ответ: {fc_result['output']}\n")
        
        print("="*60)
        print()
        
    except KeyboardInterrupt:
        print("\n\n👋 До свидания!")
        break
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("Попробуйте ещё раз или введите 'exit' для выхода")

"""
Лекция 13: Классический ReAct агент
====================================
Демонстрация базового ReAct агента с использованием стандартного ReAct промпта.
ReAct = Reasoning + Acting - агент "думает вслух" перед каждым действием.
"""

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# Проверка API ключа
# ============================================================
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# Создание инструментов
# ============================================================

@tool
def calculate(expression: str) -> str:
    """
    Вычисляет математическое выражение.
    Полезен для выполнения математических вычислений.
    
    Args:
        expression: Математическое выражение (например, "25 * 17")
    
    Returns:
        Результат вычисления в виде строки
    """
    try:
        # Безопасное вычисление
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка вычисления: {str(e)}"


@tool
def get_current_date2() -> str:
    """
    Возвращает текущую дату в формате ДД.ММ.ГГГГ.
    Полезен когда нужно узнать текущую дату.
    """
    return f"Текущая дата: 15.01.2026"


tools = [calculate, get_current_date2]

print("="*60)
print("🔧 ИНСТРУМЕНТЫ СОЗДАНЫ")
print("="*60)
for tool in tools:
    print(f"   ✅ {tool.name}: {tool.description}...")
print()

# ============================================================
# Создание ReAct агента
# ============================================================

# LLM для агента
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Получаем стандартный ReAct промпт из LangChain Hub
print("="*60)
print("📥 ЗАГРУЗКА ReAct ПРОМПТА")
print("="*60)
print("   Загружаем стандартный промпт: hwchase17/react")
react_prompt = hub.pull("hwchase17/react")
print("   ✅ Промпт загружен\n")

# Создаём ReAct агента
agent = create_react_agent(llm, tools, react_prompt)

# Создаём executor с verbose=True для показа рассуждений
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # КРИТИЧНО: показывает Thought/Action/Observation
    handle_parsing_errors=True,
    max_iterations=10,  # Ограничение итераций
)

print("="*60)
print("🤖 ReAct АГЕНТ СОЗДАН")
print("="*60)
print("   🧠 Модель: gpt-5-mini")
print("   📝 Промпт: hwchase17/react (стандартный ReAct)")
print("   🔧 Инструменты: Calculator, Get Current Date")
print("   📊 Verbose mode: включён (видим все рассуждения)")
print()

# ============================================================
# Демонстрация работы
# ============================================================

print("="*60)
print("💬 ДЕМОНСТРАЦИЯ ReAct АГЕНТА")
print("="*60)
print("""
   ReAct агент работает по циклу:
   
   1. Thought: Агент "думает вслух" что нужно сделать
   2. Action: Выбирает инструмент для использования
   3. Action Input: Передаёт параметры инструменту
   4. Observation: Получает результат от инструмента
   5. Thought: Анализирует результат и решает что делать дальше
   ... (повторяется)
   Final Answer: Формирует итоговый ответ
   
   ВАЖНО: Все эти шаги видны благодаря verbose=True!
""")
print("="*60)
print()

# ============================================================
# Интерактивный режим
# ============================================================

print("="*60)
print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
print("="*60)
print("""
   Введите ваш запрос. Вы увидите полный процесс рассуждения агента.
   Примеры:
   • Сколько будет 100 / 4 + 75?
   • Какая сегодня дата?
   • Умножь текущий день месяца на 5
   
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
        
        print(f"\n{'─'*60}")
        print(f"🔍 Обработка: «{user_input}»")
        print(f"{'─'*60}\n")
        
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

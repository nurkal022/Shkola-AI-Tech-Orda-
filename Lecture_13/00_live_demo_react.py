"""
Лекция 13: Live Demo - Пошаговое создание ReAct агента
=======================================================
Интерактивная демонстрация для лекции.
Переписываем код по этапам, показывая архитектуру ReAct.
"""

from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# ЭТАП 1: Импорты и базовые настройки
# ============================================================
print("="*60)
print("📦 ЭТАП 1: ИМПОРТЫ")
print("="*60)
print()

# Импортируем необходимые компоненты
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool

print("✅ Импортировали:")
print("   • ChatOpenAI - LLM модель")
print("   • hub - для загрузки промптов")
print("   • create_react_agent - создание ReAct агента")
print("   • AgentExecutor - выполнение агента")
print("   • tool - декоратор для инструментов")
print()

# ============================================================
# ЭТАП 2: Создание инструментов (Tools)
# ============================================================
print("="*60)
print("🔧 ЭТАП 2: СОЗДАНИЕ ИНСТРУМЕНТОВ")
print("="*60)
print()

# Инструмент 1: Калькулятор
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
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка вычисления: {str(e)}"

print("✅ Создали инструмент: calculate")
print("   Описание: Вычисляет математическое выражение")
print("   Параметры: expression (str)")
print()

# Инструмент 2: Получение даты
@tool
def get_current_date() -> str:
    """
    Возвращает текущую дату в формате ДД.ММ.ГГГГ.
    Полезен когда нужно узнать текущую дату.
    """
    from datetime import datetime
    return f"Текущая дата: {datetime.now().strftime('%d.%m.%Y')}"

print("✅ Создали инструмент: get_current_date")
print("   Описание: Возвращает текущую дату")
print("   Параметры: нет")
print()

# Список всех инструментов
tools = [calculate, get_current_date]

print(f"📋 Всего инструментов: {len(tools)}")
for i, tool in enumerate(tools, 1):
    print(f"   {i}. {tool.name}")
print()

# ============================================================
# ЭТАП 3: Инициализация LLM
# ============================================================
print("="*60)
print("🧠 ЭТАП 3: ИНИЦИАЛИЗАЦИЯ LLM")
print("="*60)
print()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("✅ LLM инициализирован:")
print(f"   Модель: gpt-4o-mini")
print(f"   Temperature: 0 (детерминированный)")
print()

# ============================================================
# ЭТАП 4: Загрузка ReAct промпта
# ============================================================
print("="*60)
print("📝 ЭТАП 4: ЗАГРУЗКА ReAct ПРОМПТА")
print("="*60)
print()

print("   Загружаем стандартный ReAct промпт из LangChain Hub...")
react_prompt = hub.pull("hwchase17/react")
print("   ✅ Промпт загружен: hwchase17/react")
print()

# Показываем структуру промпта
print("📄 Структура ReAct промпта:")
print("─" * 60)
prompt_text = react_prompt.template if hasattr(react_prompt, 'template') else str(react_prompt)
print(prompt_text[:300] + "...")
print()

print("💡 Ключевые компоненты промпта:")
print("   • Инструкция для LLM")
print("   • Список доступных инструментов: {tools}")
print("   • Формат ответа: Thought → Action → Action Input → Observation")
print("   • Переменные: {input}, {agent_scratchpad}")
print()

# ============================================================
# ЭТАП 5: Создание ReAct агента
# ============================================================
print("="*60)
print("🤖 ЭТАП 5: СОЗДАНИЕ ReAct АГЕНТА")
print("="*60)
print()

print("   Создаём агента с помощью create_react_agent...")
agent = create_react_agent(llm, tools, react_prompt)
print("   ✅ Агент создан!")
print()

print("📋 Что происходит внутри:")
print("   1. LangChain комбинирует LLM + Tools + Prompt")
print("   2. Создаётся цепочка обработки запросов")
print("   3. Агент готов к выполнению задач")
print()

# ============================================================
# ЭТАП 6: Создание AgentExecutor
# ============================================================
print("="*60)
print("⚙️  ЭТАП 6: СОЗДАНИЕ AgentExecutor")
print("="*60)
print()

print("   Создаём executor для выполнения агента...")
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # КРИТИЧНО: показывает Thought/Action/Observation
    handle_parsing_errors=True,
    max_iterations=5,
)
print("   ✅ Executor создан!")
print()

print("📋 Параметры executor:")
print("   • agent: наш ReAct агент")
print("   • tools: список инструментов")
print("   • verbose=True: показываем все рассуждения")
print("   • handle_parsing_errors=True: обрабатываем ошибки")
print("   • max_iterations=5: максимум 5 итераций")
print()

# ============================================================
# ЭТАП 7: Демонстрация работы
# ============================================================
print("="*60)
print("🚀 ЭТАП 7: ДЕМОНСТРАЦИЯ РАБОТЫ")
print("="*60)
print()

print("💡 ReAct цикл:")
print("   Thought → Action → Action Input → Observation → ... → Final Answer")
print()

# Пример 1: Простое вычисление
print("📌 ПРИМЕР 1: Простое вычисление")
print("─" * 60)
query1 = "Сколько будет 25 * 17?"
print(f"Вопрос: {query1}\n")

print("🔍 Ожидаемый процесс:")
print("   1. Thought: 'Мне нужно вычислить 25 * 17'")
print("   2. Action: calculate")
print("   3. Action Input: '25 * 17'")
print("   4. Observation: 'Результат: 425'")
print("   5. Thought: 'Я знаю ответ'")
print("   6. Final Answer: '425'")
print()

print("⏳ Выполнение...")
print("─" * 60)
result1 = agent_executor.invoke({"input": query1})
print("─" * 60)

print(f"\n✅ Финальный ответ: {result1['output']}\n")
print("="*60)
print()

# Пример 2: Многошаговая задача
print("📌 ПРИМЕР 2: Многошаговая задача")
print("─" * 60)
query2 = "Какая сегодня дата? Умножь день месяца на 10."
print(f"Вопрос: {query2}\n")

print("🔍 Ожидаемый процесс:")
print("   1. Thought: 'Нужно узнать дату'")
print("   2. Action: get_current_date")
print("   3. Observation: 'Текущая дата: 23.01.2026'")
print("   4. Thought: 'День месяца = 23, нужно умножить на 10'")
print("   5. Action: calculate")
print("   6. Action Input: '23 * 10'")
print("   7. Observation: 'Результат: 230'")
print("   8. Final Answer: '230'")
print()

print("⏳ Выполнение...")
print("─" * 60)
result2 = agent_executor.invoke({"input": query2})
print("─" * 60)

print(f"\n✅ Финальный ответ: {result2['output']}\n")
print("="*60)
print()

# ============================================================
# ЭТАП 8: Интерактивный режим
# ============================================================
print("="*60)
print("💬 ЭТАП 8: ИНТЕРАКТИВНЫЙ РЕЖИМ")
print("="*60)
print()

print("""
   Теперь вы можете задавать вопросы агенту!
   Вы увидите полный процесс рассуждения:
   
   • Thought - как агент думает
   • Action - какой инструмент выбирает
   • Action Input - какие параметры передаёт
   • Observation - что получает в ответ
   • Final Answer - итоговый ответ
   
   Введите 'exit' для выхода.
""")
print("="*60)

while True:
    try:
        user_input = input("\n🤔 Ваш вопрос: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
            print("\n👋 До свидания!")
            break
        
        if not user_input:
            print("⚠️ Пустой вопрос, попробуйте ещё раз")
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

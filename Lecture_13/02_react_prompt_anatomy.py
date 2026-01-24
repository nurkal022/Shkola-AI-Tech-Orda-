"""
Лекция 13: Анатомия ReAct промпта
==================================
Детальный разбор стандартного ReAct промпта и объяснение как он работает.
"""

from langchain import hub
from langchain_core.prompts import PromptTemplate
import json

print("="*60)
print("📖 АНАТОМИЯ ReAct ПРОМПТА")
print("="*60)
print()

# ============================================================
# Загрузка стандартного ReAct промпта
# ============================================================

print("1️⃣ ЗАГРУЗКА СТАНДАРТНОГО ReAct ПРОМПТА")
print("─" * 60)
react_prompt = hub.pull("hwchase17/react")
print("   ✅ Промпт загружен из: hwchase17/react")
print()

# ============================================================
# Показываем структуру промпта
# ============================================================

print("2️⃣ СТРУКТУРА ReAct ПРОМПТА")
print("─" * 60)
print()

# Получаем текст промпта
prompt_text = react_prompt.template if hasattr(react_prompt, 'template') else str(react_prompt)

print("📝 ПОЛНЫЙ ТЕКСТ ПРОМПТА:")
print("─" * 60)
print(prompt_text)
print()

# ============================================================
# Разбор компонентов промпта
# ============================================================

print("3️⃣ РАЗБОР КОМПОНЕНТОВ")
print("─" * 60)
print()

components = {
    "Инструкция": """Answer the following questions as best you can. You have access to the following tools:""",
    "Список инструментов": "{tools}",
    "Формат ответа": """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question""",
    "Начало работы": "Begin!\n\nQuestion: {input}\nThought:{agent_scratchpad}"
}

for name, content in components.items():
    print(f"📌 {name}:")
    print("─" * 60)
    print(content)
    print()

# ============================================================
# Объяснение как работает промпт
# ============================================================

print("4️⃣ КАК РАБОТАЕТ ReAct ПРОМПТ")
print("─" * 60)
print("""
🔍 КЛЮЧЕВЫЕ МОМЕНТЫ:

1. ИНСТРУКЦИЯ ДЛЯ LLM:
   • "Answer the following questions as best you can"
   → LLM понимает что нужно отвечать на вопросы
   
   • "You have access to the following tools"
   → LLM знает что у неё есть инструменты

2. ФОРМАТ ОТВЕТА (КРИТИЧНО!):
   • LLM должна следовать строгому формату:
     Thought → Action → Action Input → Observation
   
   • Это НЕ функция, это просто текст-инструкция!
   • LLM генерирует текст в этом формате
   • LangChain парсит этот текст и извлекает действия

3. ПЕРЕМЕННЫЕ:
   • {tools} - список доступных инструментов
   • {tool_names} - имена инструментов
   • {input} - вопрос пользователя
   • {agent_scratchpad} - история предыдущих шагов

4. ЦИКЛ РАССУЖДЕНИЯ:
   • Thought: "Мне нужно вычислить 25 * 17"
   • Action: calculate
   • Action Input: "25 * 17"
   • Observation: "Результат: 425"
   • Thought: "Я знаю ответ"
   • Final Answer: "25 * 17 = 425"
""")

# ============================================================
# Пример заполненного промпта
# ============================================================

print("5️⃣ ПРИМЕР ЗАПОЛНЕННОГО ПРОМПТА")
print("─" * 60)
print()

example_prompt = """Answer the following questions as best you can. You have access to the following tools:

calculate: Вычисляет математическое выражение. Полезен для выполнения математических вычислений.
get_current_date: Возвращает текущую дату в формате ДД.ММ.ГГГГ. Полезен когда нужно узнать текущую дату.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [calculate, get_current_date]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: Сколько будет 25 * 17?
Thought:"""

print(example_prompt)
print()

print("─" * 60)
print("""
💡 ЧТО ПРОИСХОДИТ ДАЛЬШЕ:

1. LLM получает этот промпт
2. LLM генерирует продолжение в формате:
   
   Thought: Мне нужно вычислить 25 * 17. Для этого использую инструмент calculate.
   Action: calculate
   Action Input: 25 * 17
   
3. LangChain парсит ответ LLM:
   • Находит "Action: calculate"
   • Находит "Action Input: 25 * 17"
   • Выполняет функцию calculate("25 * 17")
   • Получает результат: "Результат: 425"

4. LangChain добавляет результат в промпт:
   
   Observation: Результат: 425
   
5. LLM продолжает:
   
   Thought: Я получил результат 425. Теперь могу дать финальный ответ.
   Final Answer: 25 * 17 = 425
""")

# ============================================================
# Сравнение с OpenAI Function Calling
# ============================================================

print("6️⃣ ReAct vs OpenAI Function Calling")
print("─" * 60)
print()

comparison = """
┌─────────────────────────┬──────────────────────┬─────────────────────────┐
│ Аспект                  │ ReAct Prompting      │ OpenAI Function Calling │
├─────────────────────────┼──────────────────────┼─────────────────────────┤
│ Как LLM видит инструменты│ Текст в промпте      │ JSON-схема в API        │
│ Формат ответа            │ Текст (Thought/      │ Структурированный JSON  │
│                          │ Action/Observation) │ (tool_calls)            │
│ Парсинг                  │ LangChain парсит     │ OpenAI API возвращает   │
│                          │ текст                │ готовый JSON            │
│ Прозрачность             │ ✅ Видим все мысли   │ ❌ Чёрный ящик          │
│ Надёжность               │ ⚠️ Может ошибиться   │ ✅ Структурированный    │
│                          │ в формате            │ вывод                   │
│ Совместимость            │ ✅ Любая LLM         │ ⚠️ Только OpenAI        │
│ Отладка                  │ ✅ Легко понять      │ ❌ Сложнее              │
│                          │ что пошло не так     │                         │
└─────────────────────────┴──────────────────────┴─────────────────────────┘
"""

print(comparison)
print()

print("="*60)
print("✅ РАЗБОР ЗАВЕРШЁН")
print("="*60)
print("""
   Ключевой вывод: ReAct - это просто промпт-инжиниринг!
   LLM следует текстовым инструкциям, а LangChain парсит результат.
""")

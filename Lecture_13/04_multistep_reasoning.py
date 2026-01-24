"""
Лекция 13: Многошаговые рассуждения (Multi-step Reasoning)
===========================================================
Демонстрация ReAct агента на сложных задачах, требующих несколько шагов.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# Загрузка FAISS индекса
# ============================================================

print("="*60)
print("📂 ЗАГРУЗКА БАЗЫ ДАННЫХ")
print("="*60)

index_path = Path("data/faiss_harry_potter")

if not index_path.exists():
    print(f"   ❌ Индекс не найден: {index_path}")
    exit(1)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    str(index_path),
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(f"   ✅ Индекс загружен: {index_path}")
print()

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
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка вычисления: {str(e)}"


@tool
def count_words(text: str) -> str:
    """
    Подсчитывает количество слов в тексте.
    Полезен для подсчёта слов в тексте.
    
    Args:
        text: Текст для подсчёта слов
    
    Returns:
        Количество слов в виде строки
    """
    word_count = len(text.split())
    return f"Количество слов: {word_count}"


# Создаём retriever tool для поиска по книгам
rag_tool = create_retriever_tool(
    retriever=retriever,
    name="harry_potter_search",
    description="""Полезен для поиска информации о книгах Гарри Поттера.
    Используй этот инструмент когда нужно найти информацию о:
    - Персонажах (Гарри Поттер, Дамблдор, Дурсли и т.д.)
    - Местах (Хогвартс, Косой переулок и т.д.)
    - Событиях и сюжете
    - Магических предметах
    
    Вопрос должен быть на русском языке.
    """
)

tools = [calculate, count_words, rag_tool]

print("="*60)
print("🔧 ИНСТРУМЕНТЫ СОЗДАНЫ")
print("="*60)
for tool in tools:
    print(f"   ✅ {tool.name}")
print()

# ============================================================
# Создание ReAct агента
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,  # Увеличиваем для сложных задач
)

print("="*60)
print("🤖 ReAct АГЕНТ СОЗДАН")
print("="*60)
print("   🧠 Модель: gpt-4o-mini")
print("   🔧 Инструменты: Calculator, Word Counter, RAG Search")
print("   📊 Max iterations: 10 (для сложных задач)")
print()

# ============================================================
# Примеры многошаговых задач
# ============================================================

print("="*60)
print("💡 ПРИМЕРЫ МНОГОШАГОВЫХ ЗАДАЧ")
print("="*60)
print()

# Пример 1: Комбинированная задача
print("📌 ПРИМЕР 1: Комбинированная задача")
print("─" * 60)
query1 = "Найди информацию о Хогвартсе, подсчитай количество факультетов и умножь на 100"
print(f"Вопрос: {query1}\n")
print("Ожидаемые шаги:")
print("  1. Thought: Нужно найти информацию о Хогвартсе")
print("  2. Action: harry_potter_search")
print("  3. Observation: Найдена информация о 4 факультетах")
print("  4. Thought: Нужно умножить 4 на 100")
print("  5. Action: calculate")
print("  6. Observation: Результат: 400")
print("  7. Final Answer: ...")
print()
print("Выполнение:")
print("─" * 60)
result1 = agent_executor.invoke({"input": query1})
print(f"\n✅ Финальный ответ: {result1['output']}\n")
print("="*60)
print()

# Пример 2: Анализ текста
print("📌 ПРИМЕР 2: Анализ текста")
print("─" * 60)
query2 = "Найди информацию о Гарри Поттере, извлеки первое предложение и подсчитай в нём количество слов"
print(f"Вопрос: {query2}\n")
print("Ожидаемые шаги:")
print("  1. Thought: Нужно найти информацию о Гарри Поттере")
print("  2. Action: harry_potter_search")
print("  3. Observation: Найдена информация...")
print("  4. Thought: Нужно извлечь первое предложение и подсчитать слова")
print("  5. Action: count_words")
print("  6. Observation: Количество слов: X")
print("  7. Final Answer: ...")
print()
print("Выполнение:")
print("─" * 60)
result2 = agent_executor.invoke({"input": query2})
print(f"\n✅ Финальный ответ: {result2['output']}\n")
print("="*60)
print()

# Пример 3: Сложная математика + поиск
print("📌 ПРИМЕР 3: Сложная задача")
print("─" * 60)
query3 = "Найди сколько главных героев упоминается в первой книге, умножь это число на 5, затем прибавь 10"
print(f"Вопрос: {query3}\n")
print("Ожидаемые шаги:")
print("  1. Thought: Нужно найти информацию о главных героях")
print("  2. Action: harry_potter_search")
print("  3. Observation: Найдена информация...")
print("  4. Thought: Нужно определить количество и выполнить вычисления")
print("  5. Action: calculate")
print("  6. Observation: Результат...")
print("  7. Final Answer: ...")
print()
print("Выполнение:")
print("─" * 60)
result3 = agent_executor.invoke({"input": query3})
print(f"\n✅ Финальный ответ: {result3['output']}\n")
print("="*60)
print()

# ============================================================
# Интерактивный режим
# ============================================================

print("="*60)
print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
print("="*60)
print("""
   Попробуйте сложные многошаговые задачи:
   
   Примеры:
   • Найди информацию о Хогвартсе и умножь количество факультетов на 10
   • Найди описание Гарри Поттера, извлеки первое предложение и подсчитай слова
   • Найди информацию о философском камне, затем умножь длину описания на 2
   
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

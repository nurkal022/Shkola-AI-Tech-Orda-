"""
Лекция 12: Combined Agent (Calculator + RAG)
============================================
Агент с несколькими инструментами.
Демонстрирует как агент выбирает между разными инструментами.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


# 1. Calculator Tool
@tool
def calculate(expression: str) -> str:
    """
    Вычисляет математическое выражение.
    Полезен для выполнения математических вычислений.
    Принимает математическое выражение в виде строки.
    Примеры: "25 * 17", "100 / 4 + 75", "(50 + 30) * 2"
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка вычисления: {str(e)}"

calculator_tool = calculate

# 2. RAG Tool
index_path = Path("data/faiss_harry_potter")

if index_path.exists():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    rag_tool = create_retriever_tool(
        retriever=retriever,
        name="harry_potter_search",
        description="""Полезен для поиска информации о книгах Гарри Поттера.
        Используй когда нужно найти информацию о персонажах, местах, событиях,
        магических предметах из книг Гарри Поттера.
        """
    )
else:
    print("   ⚠️ RAG индекс не найден, используем только Calculator")
    rag_tool = None

tools = [calculator_tool]
if rag_tool:
    tools.append(rag_tool)

print(f"   📊 Всего инструментов: {len(tools)}\n")


# ============================================================
# Создание агента
# ============================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Промпт для агента
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты умный помощник с доступом к калькулятору и базе знаний о книгах Гарри Поттера.
Используй калькулятор для математических вычислений.
Используй поиск для вопросов о книгах Гарри Поттера.
Можешь комбинировать инструменты если нужно.
Отвечай на русском языке."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

print("   ✅ Combined Агент создан")
print(f"   🔧 Инструменты: {[t.name for t in tools]}\n")


# ============================================================
# Интерактивный режим чата
# ============================================================
print("="*60)
print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ ЧАТА")
print("="*60)

available_tools = [t.name for t in tools]
print(f"""
   Доступные инструменты: {', '.join(available_tools)}
   
   Введите ваш запрос (математика или вопросы о Гарри Поттере).   
   Примеры запросов:
   • Математика: "Сколько будет 25 * 17?"
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


# ============================================================
# Анализ выбора инструментов
# ============================================================
print("\n" + "="*60)
print("📊 АНАЛИЗ: Как агент выбирает инструменты")
print("="*60)
print("""
Агент анализирует запрос и решает:

1. Математический вопрос?
   → Использует Calculator

2. Вопрос о Гарри Поттере?
   → Использует harry_potter_search

3. Комбинированный вопрос?
   → Использует оба инструмента последовательно

Пример цепочки:
  Запрос: "Сколько глав в первой книге и умножь на 10"
  
  1. [THOUGHT] Нужно найти информацию о книге
  2. [ACTION] harry_potter_search("количество глав первая книга")
  3. [OBSERVATION] Найдена информация: 17 глав
  4. [THOUGHT] Теперь нужно умножить 17 на 10
  5. [ACTION] Calculator("17 * 10")
  6. [OBSERVATION] Результат: 170
  7. [ANSWER] В первой книге 17 глав, умноженное на 10 = 170
""")


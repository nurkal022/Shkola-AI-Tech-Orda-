"""
Лекция 12: RAG Agent
====================
Агент с инструментом поиска по документам (RAG).
Использует существующий FAISS индекс из предыдущих лекций.
"""


"""
Обычный RAG:
  • Всегда выполняет поиск
  • Фиксированный пайплайн: Query → Search → LLM
  • Не может решить нужен ли поиск

RAG Agent:
  • Решает нужен ли поиск
  • Может комбинировать поиск с другими действиями
  • Более гибкий и умный
  • Может отвечать на общие вопросы без поиска

Пример:
  Вопрос: "Привет, как дела?"
  
  Обычный RAG: Ищет в базе → Не находит → Странный ответ
  
  RAG Agent: Понимает что это приветствие → Отвечает без поиска
"""


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


# ============================================================
# Загрузка существующего FAISS индекса
# ============================================================

index_path = Path("data/faiss_harry_potter")

if not index_path.exists():
    print(f"   ❌ Индекс не найден: {index_path}")
    print("   Создаём новый индекс...")
    
    # Создаём индекс из данных
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text = Path("../Lecture_07/data/Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt").read_text(encoding='utf-8')
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # Сохраняем
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    print(f"   ✅ Индекс создан и сохранён: {index_path}")
else:
    print(f"   📂 Загрузка индекса: {index_path}")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

# Создаём retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ============================================================
# Создание инструмента - RAG Retriever

# Создаём retriever tool
rag_tool = create_retriever_tool(
    retriever=retriever,
    name="harry_potter_search",
    description="""Полезен для поиска информации о книгах Гарри Поттера.
    Используй этот инструмент когда нужно найти информацию о:
    - Персонажах (Гарри Поттер, Дамблдор, Дурсли и т.д.)
    - Местах (Хогвартс, Косой переулок и т.д.)
    - Событиях и сюжете
    - Магических предметах (философский камень, волшебные палочки и т.д.)
    
    Вопрос должен быть на русском языке.
    """
)


# ============================================================
# Создание агента
# ============================================================

# LLM для агента
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Промпт для агента
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты помощник, который отвечает на вопросы о книгах Гарри Поттера.
Используй инструмент поиска для нахождения информации в книгах.
Отвечай на основе найденной информации.
Отвечай на русском языке."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Создаём агента
agent = create_openai_tools_agent(llm, [rag_tool], prompt)

# Создаём executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[rag_tool],
    verbose=True,  # Показываем "мысли" агента
    handle_parsing_errors=True,
)


# ============================================================
# Интерактивный режим чата
# ============================================================
print("="*60)
print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ ЧАТА")
print("="*60)
print("""
   Введите ваш вопрос о книгах Гарри Поттера.   
   Примеры вопросов:
   • Кто такой Гарри Поттер?
""")
print("="*60)

while True:
    try:
        # Получаем запрос от пользователя
        user_input = input("\n🤔 Ваш вопрос: ").strip()
        
        # Проверка на выход
        if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
            print("\n👋 До свидания!")
            break
        
        if not user_input:
            print("⚠️ Пустой вопрос, попробуйте ещё раз")
            continue
        
        print(f"\n{'─'*60}")
        print(f"🔍 Обработка вопроса: «{user_input}»")
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




"""
RAG Шаг 6: Интерактивный чат
============================
Чат-бот с памятью диалога и стримингом.
"""
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()

# ============================================================
# НАСТРОЙКА
# ============================================================
INDEX_PATH = "./faiss_harry_potter"

if not Path(INDEX_PATH).exists():
    print("❌ Сначала запустите 04_rag_pipeline.py для создания индекса")
    exit()

# Компоненты
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Память (последние 10 сообщений)
memory = ConversationBufferWindowMemory(k=10, return_messages=True)

# Промпт
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты эксперт по Гарри Поттеру. Отвечай дружелюбно на русском.
Используй контекст из книг: {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

print("✅ Чат-бот готов!")


# ============================================================
# ФУНКЦИЯ ЧАТА
# ============================================================
def chat(question: str):
    """Ответ на вопрос с памятью и стримингом"""
    
    # Поиск контекста
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # История
    history = memory.load_memory_variables({})["history"]
    
    # Генерация со стримингом
    chain = prompt | llm | StrOutputParser()
    
    print("🤖 ", end="", flush=True)
    response = ""
    for chunk in llm.stream(prompt.format_messages(
        context=context, 
        history=history, 
        question=question
    )):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            response += chunk.content
    print()
    
    # Сохраняем в память
    memory.save_context({"input": question}, {"output": response})
    
    return response


# ============================================================
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧙‍♂️ HARRY POTTER CHATBOT")
    print("="*60)
    print("Команды: /clear - очистить память, /quit - выход")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("👤 Вы: ").strip()
            
            if not question:
                continue
            if question == "/quit":
                print("👋 Пока!")
                break
            if question == "/clear":
                memory.clear()
                print("🗑️ Память очищена")
                continue
            
            chat(question)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Пока!")
            break

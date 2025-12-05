"""
RAG Шаг 4: Базовый RAG Pipeline
===============================
Полный pipeline: Query → Search → Generate
"""
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# ============================================================
# 1. НАСТРОЙКА КОМПОНЕНТОВ
# ============================================================
print("="*60)
print("🚀 RAG PIPELINE")
print("="*60)

# LLM для генерации
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
print("✅ LLM: gpt-4.1-mini")

# Embeddings для поиска
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("✅ Embeddings: text-embedding-3-small")


# ============================================================
# 2. СОЗДАНИЕ/ЗАГРУЗКА ИНДЕКСА
# ============================================================
INDEX_PATH = "./faiss_harry_potter"

if Path(INDEX_PATH).exists():
    print(f"\n📂 Загрузка индекса: {INDEX_PATH}")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print(f"\n🔨 Создание нового индекса...")
    
    # Загружаем все книги
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for file in sorted(Path("data").glob("*.txt")):
        name = file.stem.split(']_')[-1].replace('_', ' ')
        text = file.read_text(encoding='utf-8')
        chunks = splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"title": name, "chunk_id": i}
            ))
        print(f"   📖 {name}: {len(chunks)} чанков")
    
    print(f"\n   Всего документов: {len(documents)}")
    print("   Создание векторов (это займет несколько минут)...")
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"   💾 Сохранено: {INDEX_PATH}")


# ============================================================
# 3. ПРОМПТ ДЛЯ RAG
# ============================================================
prompt = ChatPromptTemplate.from_template("""
Ты эксперт по книгам о Гарри Поттере. Отвечай на вопросы используя контекст.

Правила:
- Отвечай на русском
- Используй только информацию из контекста
- Если не знаешь - скажи честно

Контекст:
{context}

Вопрос: {question}

Ответ:""")


# ============================================================
# 4. RAG ФУНКЦИЯ
# ============================================================
def ask(question: str, k: int = 4) -> str:
    """Задать вопрос RAG системе"""
    
    # 1. Поиск релевантных документов
    docs = vectorstore.similarity_search(question, k=k)
    
    # 2. Формируем контекст
    context = "\n\n---\n\n".join([
        f"[{doc.metadata['title']}]\n{doc.page_content}"
        for doc in docs
    ])
    
    # 3. Генерируем ответ
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return answer, [doc.metadata['title'] for doc in docs]


# ============================================================
# 5. ДЕМОНСТРАЦИЯ
# ============================================================
print("\n" + "="*60)
print("💬 ВОПРОСЫ И ОТВЕТЫ")
print("="*60)

questions = [
    "Кто такой Волдеморт?",
    "Как Гарри узнал что он волшебник?",
    "Кто друзья Гарри Поттера?",
    "Что такое Хогвартс?",
]

for q in questions:
    print(f"\n❓ {q}")
    print("-"*50)
    answer, sources = ask(q)
    print(f"💡 {answer}")
    print(f"📚 Источники: {', '.join(set(sources))}")

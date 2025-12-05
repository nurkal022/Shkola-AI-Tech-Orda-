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

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
print("✅ LLM: gpt-4.1-mini")

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
    print("   Создание векторов батчами (избегаем лимита токенов)...")
    
    # Создаем индекс батчами по 500 документов
    BATCH_SIZE = 500
    vectorstore = None
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i+BATCH_SIZE]
        print(f"   Батч {i//BATCH_SIZE + 1}: документы {i}-{i+len(batch)}")
        
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)
    
    vectorstore.save_local(INDEX_PATH)
    print(f"   💾 Сохранено: {INDEX_PATH}")


# ============================================================
# 3. ПОИСК БЕЗ ГЕНЕРАЦИИ (демо)
# ============================================================
print("\n" + "="*60)
print("🔍 ПОИСК ДОКУМЕНТОВ (без LLM)")
print("="*60)

query = "Волдеморт"
print(f"\n❓ Запрос: {query}")

docs = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in docs:
    print(f"\n📄 [{doc.metadata['title']}] score={score:.3f}")
    print(f"   {doc.page_content[:200]}...")


# ============================================================
# 4. RAG: ПОИСК + ГЕНЕРАЦИЯ
# ============================================================
print("\n" + "="*60)
print("🤖 RAG: ПОИСК + ГЕНЕРАЦИЯ")
print("="*60)

prompt = ChatPromptTemplate.from_template("""
Ты эксперт по Гарри Поттеру. Отвечай используя контекст.

Контекст:
{context}

Вопрос: {question}

Ответ:""")


def ask(question: str, k: int = 4) -> str:
    """RAG: поиск + генерация"""
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return answer, [doc.metadata['title'] for doc in docs]


# Демо
questions = [
    "Кто такой Волдеморт?",
    "Как Гарри попал в Хогвартс?",
]

for q in questions:
    print(f"\n❓ {q}")
    print("-"*50)
    answer, sources = ask(q)
    print(f"💡 {answer}")
    print(f"📚 Источники: {', '.join(set(sources))}")

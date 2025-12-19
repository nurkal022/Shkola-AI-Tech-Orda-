"""
Пример 6: Интеграция с LangChain

LangChain предоставляет единый интерфейс для разных векторных БД:
- FAISS
- ChromaDB
- Pinecone, Qdrant, pgvector и др.

Меняем БД — код остаётся почти тем же!
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# =============================================
# Загружаем данные из книг
# =============================================
from utils import get_all_books_chunks

print("Загружаем данные из книг...")
chunks = get_all_books_chunks("data", chunk_size=500, max_chunks=200)

documents = [
    Document(
        page_content=chunk["text"],
        metadata={"book": chunk["book"], "chunk_id": chunk["chunk_id"]}
    )
    for chunk in chunks
]

print(f"Загружено {len(documents)} документов\n")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("=== LangChain VectorStore ===\n")

# =============================================
# Вариант 1: FAISS
# =============================================
print("📦 FAISS через LangChain")
print("-" * 40)

faiss_store = FAISS.from_documents(documents, embeddings)

# Простой поиск
results = faiss_store.similarity_search("кто такой Гарри Поттер?", k=2)
print("Поиск: 'кто такой Гарри Поттер?'")
for doc in results:
    preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
    print(f"   [{doc.metadata['book']}] {preview}")

# Сохранение на диск
faiss_store.save_local("faiss_langchain_index")
print("\nИндекс сохранён в ./faiss_langchain_index")


# =============================================
# Вариант 2: ChromaDB
# =============================================
print("\n\n📦 ChromaDB через LangChain")
print("-" * 40)

chroma_store = Chroma.from_documents(
    documents, 
    embeddings,
    persist_directory="./chroma_langchain_db"
)

# Поиск с фильтром
if documents:
    first_book = documents[0].metadata["book"]
    results = chroma_store.similarity_search(
        "что такое Хогвартс?",
        k=2,
        filter={"book": first_book}
    )
    print(f"Поиск: 'что такое Хогвартс?' (filter: book={first_book})")
    for doc in results:
        preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
        print(f"   [{doc.metadata['book']}] {preview}")


# =============================================
# RAG: Поиск + LLM
# =============================================
print("\n\n🤖 RAG: Retrieval + Generation")
print("-" * 40)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Создаём retriever
retriever = chroma_store.as_retriever(search_kwargs={"k": 2})

# Промпт для RAG
prompt = ChatPromptTemplate.from_template("""
Ответь на вопрос, используя только предоставленный контекст.

Контекст:
{context}

Вопрос: {question}

Ответ:""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# RAG цепочка
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Тестируем
questions = [
    "Кто такой Гарри Поттер?",
    "Что такое Хогвартс?",
    "Кто такой Хагрид?",
]

for question in questions:
    print(f"\n❓ {question}")
    answer = rag_chain.invoke(question)
    print(f"💬 {answer}")


# =============================================
# Единый интерфейс
# =============================================
print("\n\n" + "=" * 50)
print("💡 ГЛАВНОЕ ПРЕИМУЩЕСТВО LangChain")
print("=" * 50)
print("""
Один и тот же код работает с разными БД:

    # FAISS
    store = FAISS.from_documents(docs, embeddings)
    
    # ChromaDB  
    store = Chroma.from_documents(docs, embeddings)
    
    # Pinecone
    store = Pinecone.from_documents(docs, embeddings)
    
    # Qdrant
    store = Qdrant.from_documents(docs, embeddings)

Методы одинаковые:
    store.similarity_search(query, k=3)
    store.as_retriever()
    store.add_documents(new_docs)
""")


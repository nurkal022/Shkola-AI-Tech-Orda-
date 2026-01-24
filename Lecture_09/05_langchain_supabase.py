"""
Пример 5: RAG с LangChain и Supabase

LangChain упрощает работу с Supabase:
- SupabaseVectorStore автоматически создаёт нужные функции
- Готовый retriever для RAG
- Единый интерфейс с другими векторными БД
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from supabase import create_client, Client
import os

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=== RAG с LangChain и Supabase ===\n")

# =============================================
# Шаг 1: Создаём SupabaseVectorStore
# =============================================
print("1. Создаём SupabaseVectorStore...")

# LangChain автоматически создаст таблицу и функции если их нет
vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"  # Имя RPC функции
)

print("   ✅ VectorStore создан")

# =============================================
# Шаг 2: Загружаем документы (если ещё не загружены)
# =============================================
print("\n2. Проверяем наличие документов...")

from utils import get_book_chunks

# Проверяем количество документов
try:
    count_result = supabase.table("documents").select("id", count="exact").limit(1).execute()
    doc_count = count_result.count if hasattr(count_result, 'count') else 0
    
    if doc_count == 0:
        print("   Документов нет, загружаем конституцию...")
        chunks = get_book_chunks("data/конституция.txt", chunk_size=1500, chunk_overlap=200)
        
        documents = [
            Document(
                page_content=chunk["text"],
                metadata={"book": chunk["book"], "chunk_id": chunk["chunk_id"]}
            )
            for chunk in chunks
        ]
        
        # Загружаем через LangChain
        vectorstore.add_documents(documents)
        print(f"   ✅ Загружено {len(documents)} документов")
    else:
        print(f"   ✅ В БД уже есть документы: {doc_count}")
except Exception as e:
    print(f"   ⚠️  {e}")

# =============================================
# Шаг 3: Создаём RAG цепочку
# =============================================
print("\n3. Создаём RAG цепочку...")

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Промпт
prompt = ChatPromptTemplate.from_template("""
Ответь на вопрос, используя только предоставленный контекст из Конституции Республики Казахстан.

Контекст:
{context}

Вопрос: {question}

Ответ (кратко, 2-3 предложения):""")

def format_docs(docs):
    """Форматируем документы для промпта"""
    return "\n\n".join([
        f"[Из книги: {doc.metadata.get('book', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    ])

# RAG цепочка
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("   ✅ RAG цепочка готова")

# =============================================
# Шаг 4: Тестируем RAG
# =============================================
print("\n4. Тестируем RAG...\n")

questions = [
    "Какие права и свободы гарантирует Конституция?",
    "Каковы полномочия Президента?",
    "Как формируется Парламент?",
]

for question in questions:
    print("="*60)
    print(f"❓ Вопрос: {question}\n")
    
    # Показываем найденные документы
    docs = retriever.invoke(question)
    print("📚 Найденные документы:")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
        print(f"   {i}. [{doc.metadata.get('book', 'Unknown')}] {preview}")
    
    # Генерируем ответ
    print("\n💬 Ответ:")
    answer = rag_chain.invoke(question)
    print(f"   {answer}\n")

print("="*60)
print("\n✅ RAG пайплайн работает!")


"""
Пример 3: Векторный поиск в Supabase

Используем SQL операторы pgvector:
- <=>  косинусное расстояние (1 - cosine similarity)
- <->  L2 расстояние (Euclidean)
"""

from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

openai_client = OpenAI()

print("=== Векторный поиск в Supabase ===\n")


def get_embedding(text: str) -> list[float]:
    """Получить эмбеддинг текста"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def vector_search(query: str, top_k: int = 3, book_filter: str = None):
    """
    Векторный поиск в Supabase
    
    Args:
        query: Текстовый запрос
        top_k: Количество результатов
        book_filter: Фильтр по книге (опционально)
    """
    # Получаем эмбеддинг запроса
    query_embedding = get_embedding(query)
    
    # SQL запрос для векторного поиска
    # <=> это оператор косинусного расстояния (1 - cosine similarity)
    # Чем меньше значение, тем больше сходство
    
    if book_filter:
        sql = f"""
        SELECT 
            id,
            content,
            book,
            chunk_id,
            1 - (embedding <=> '{query_embedding}')::float AS similarity
        FROM documents
        WHERE book = '{book_filter}'
        ORDER BY embedding <=> '{query_embedding}'
        LIMIT {top_k}
        """
    else:
        sql = f"""
        SELECT 
            id,
            content,
            book,
            chunk_id,
            1 - (embedding <=> '{query_embedding}')::float AS similarity
        FROM documents
        ORDER BY embedding <=> '{query_embedding}'
        LIMIT {top_k}
        """
    
    # Выполняем через rpc (если настроен) или напрямую через SQL
    # Для простоты используем прямой SQL через Supabase REST API
    # В реальности лучше использовать rpc функцию
    
    # Альтернативный способ через Supabase Python client
    # Но для векторного поиска нужен прямой SQL
    print(f"   SQL запрос: {sql[:100]}...")
    
    # Используем rpc если функция создана, иначе делаем через select + фильтрацию
    # Для демо используем упрощённый подход
    
    return sql


# =============================================
# Демонстрация поиска
# =============================================
print("🔍 Запрос: 'кто такой Гарри Поттер?'")

query_embedding = get_embedding("кто такой Гарри Поттер?")

# Используем Supabase RPC для векторного поиска
# Сначала нужно создать функцию в SQL:

create_function_sql = """
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding VECTOR(1536),
  match_count INT DEFAULT 5,
  filter_book TEXT DEFAULT NULL
)
RETURNS TABLE (
  id INT,
  content TEXT,
  book TEXT,
  chunk_id INT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    documents.id,
    documents.content,
    documents.book,
    documents.chunk_id,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE (filter_book IS NULL OR documents.book = filter_book)
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
"""

print("\n📝 Для векторного поиска создайте функцию в Supabase SQL Editor:")
print(create_function_sql)

# После создания функции можно использовать:
print("\n✅ После создания функции используйте:")
print("""
# Python код:
result = supabase.rpc('match_documents', {
    'query_embedding': query_embedding,
    'match_count': 3,
    'filter_book': None  # или название книги
}).execute()

for row in result.data:
    print(f"[{row['similarity']:.3f}] {row['content'][:80]}...")
""")

# Альтернативный способ через прямой SQL (если rpc недоступен)
print("\n" + "="*50)
print("АЛЬТЕРНАТИВНЫЙ СПОСОБ: Поиск через LangChain")
print("="*50)
print("""
Используйте langchain-community для работы с Supabase:
- SupabaseVectorStore автоматически создаёт нужные функции
- Упрощает работу с векторным поиском
- См. пример 05_langchain_supabase.py
""")


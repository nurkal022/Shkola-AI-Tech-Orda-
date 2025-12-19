"""
Пример 7: Qdrant Cloud — облачная векторная БД

Qdrant:
- Высокопроизводительная векторная БД
- Поддержка фильтрации по метаданным
- Облачная и self-hosted версии
- REST API и gRPC
"""

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
import os

load_dotenv()

# OpenAI для эмбеддингов
openai_client = OpenAI()

# Qdrant Cloud подключение
QDRANT_URL = "https://a16f99ab-5aa1-4d6b-b332-5605139e319e.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.s-YFvQRRvH2_5HnRWz05Z-UB0oSceySRaJlpu2An1Ko"

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60  # Увеличиваем таймаут для облака
)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Получить эмбеддинги через OpenAI"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]


print("=== Qdrant Cloud ===\n")

# =============================================
# Шаг 1: Создаём коллекцию
# =============================================
print("1. Создаём коллекцию...")

COLLECTION_NAME = "harry_potter"
VECTOR_SIZE = 1536  # text-embedding-3-small

# Удаляем если существует
try:
    qdrant.delete_collection(COLLECTION_NAME)
except:
    pass

# Создаём новую коллекцию
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE
    )
)
print(f"   Коллекция '{COLLECTION_NAME}' создана")

# =============================================
# Шаг 2: Загружаем данные из книг
# =============================================
print("\n2. Загружаем данные...")

from utils import get_all_books_chunks

chunks = get_all_books_chunks("data", chunk_size=500, max_chunks=30)  # Меньше для облака
print(f"   Загружено {len(chunks)} чанков")

# =============================================
# Шаг 3: Создаём эмбеддинги и загружаем в Qdrant
# =============================================
print("\n3. Создаём эмбеддинги и загружаем в Qdrant...")

texts = [chunk["text"] for chunk in chunks]

# Обрабатываем и загружаем батчами
batch_size = 10  # Маленькие батчи для облака

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_chunks = chunks[i:i+batch_size]
    
    embeddings = get_embeddings(batch_texts)
    
    points = []
    for j, (emb, chunk) in enumerate(zip(embeddings, batch_chunks)):
        points.append(PointStruct(
            id=i + j,
            vector=emb,
            payload={
                "text": chunk["text"],
                "book": chunk["book"],
                "chunk_id": chunk["chunk_id"]
            }
        ))
    
    # Загружаем батч сразу
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"   Загружено: {min(i+batch_size, len(texts))}/{len(texts)}")

print(f"\n   ✅ Загружено {len(texts)} точек в Qdrant Cloud")

# Создаём индекс для фильтрации по книге
qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="book",
    field_schema=PayloadSchemaType.KEYWORD
)
print("   Создан индекс для поля 'book'")


# =============================================
# Шаг 4: Поиск
# =============================================
def search(query: str, top_k: int = 3, book_filter: str = None):
    """Поиск похожих документов"""
    query_emb = get_embeddings([query])[0]
    
    # Фильтр по книге (опционально)
    search_filter = None
    if book_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="book",
                    match=MatchValue(value=book_filter)
                )
            ]
        )
    
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_emb,
        limit=top_k,
        query_filter=search_filter
    )
    
    return results.points


print("\n4. Поиск...\n")

# Простой поиск
print("🔍 Запрос: 'кто такой Гарри Поттер?'")
results = search("кто такой Гарри Поттер?")
for r in results:
    preview = r.payload["text"][:80] + "..." if len(r.payload["text"]) > 80 else r.payload["text"]
    print(f"   [{r.score:.3f}] [{r.payload['book']}] {preview}")

# Поиск с фильтром
print("\n🔍 Запрос: 'магия и волшебство' (только первая книга)")
first_book = chunks[0]["book"]
results = search("магия и волшебство", book_filter=first_book)
for r in results:
    preview = r.payload["text"][:80] + "..." if len(r.payload["text"]) > 80 else r.payload["text"]
    print(f"   [{r.score:.3f}] {preview}")

# Ещё один запрос
print("\n🔍 Запрос: 'кто такой Хагрид?'")
results = search("кто такой Хагрид?")
for r in results:
    preview = r.payload["text"][:80] + "..." if len(r.payload["text"]) > 80 else r.payload["text"]
    print(f"   [{r.score:.3f}] [{r.payload['book']}] {preview}")


# =============================================
# Информация о коллекции
# =============================================
print("\n\n=== Информация о коллекции ===")
info = qdrant.get_collection(COLLECTION_NAME)
print(f"Название: {COLLECTION_NAME}")
print(f"Количество точек: {info.points_count}")
print(f"Размер вектора: {info.config.params.vectors.size}")
print(f"Метрика: {info.config.params.vectors.distance}")


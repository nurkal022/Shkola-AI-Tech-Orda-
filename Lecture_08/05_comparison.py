"""
Пример 5: Сравнение FAISS и ChromaDB

Одна задача — два решения. Сравниваем:
- Скорость
- Удобство API
- Возможности фильтрации
"""

from dotenv import load_dotenv
from openai import OpenAI
import faiss
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import time

load_dotenv()

client = OpenAI()


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Получить эмбеддинги через OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([d.embedding for d in response.data], dtype='float32')


# =============================================
# Загружаем данные из книг
# =============================================
from utils import get_all_books_chunks

print("Загружаем данные из книг...")
chunks = get_all_books_chunks("data", chunk_size=500, max_chunks=100)
documents = [{"text": chunk["text"], "book": chunk["book"]} for chunk in chunks]
texts = [d["text"] for d in documents]
query = "кто такой Гарри Поттер и что такое Хогвартс?"

print("=== Сравнение FAISS vs ChromaDB ===\n")

# =============================================
# FAISS
# =============================================
print("📦 FAISS")
print("-" * 40)

start = time.time()
embeddings = get_embeddings(texts)
faiss.normalize_L2(embeddings)
faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(embeddings)
faiss_setup_time = time.time() - start
print(f"   Время создания индекса: {faiss_setup_time:.3f}s")

start = time.time()
query_emb = get_embeddings([query])
faiss.normalize_L2(query_emb)
scores, indices = faiss_index.search(query_emb, 3)
faiss_search_time = time.time() - start
print(f"   Время поиска: {faiss_search_time:.3f}s")

print("   Результаты:")
for idx, score in zip(indices[0], scores[0]):
    preview = documents[idx]['text'][:60] + "..." if len(documents[idx]['text']) > 60 else documents[idx]['text']
    print(f"      [{score:.3f}] {preview}")

# Фильтрация в FAISS — только вручную
print("\n   ⚠️  Фильтрация по метаданным: нужно делать вручную после поиска")


# =============================================
# ChromaDB
# =============================================
print("\n\n📦 ChromaDB")
print("-" * 40)

start = time.time()
chroma_client = chromadb.Client()  # in-memory для сравнения
openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")

try:
    chroma_client.delete_collection("comparison")
except:
    pass

collection = chroma_client.create_collection("comparison", embedding_function=openai_ef)
collection.add(
    documents=texts,
    metadatas=[{"book": d["book"]} for d in documents],
    ids=[f"doc_{i}" for i in range(len(documents))]
)
chroma_setup_time = time.time() - start
print(f"   Время создания коллекции: {chroma_setup_time:.3f}s")

start = time.time()
results = collection.query(query_texts=[query], n_results=3)
chroma_search_time = time.time() - start
print(f"   Время поиска: {chroma_search_time:.3f}s")

print("   Результаты:")
for doc, dist in zip(results['documents'][0], results['distances'][0]):
    preview = doc[:60] + "..." if len(doc) > 60 else doc
    print(f"      [{1-dist:.3f}] {preview}")

# Фильтрация в ChromaDB — встроенная
print("\n   ✅ Фильтрация по метаданным: встроенная")
if documents:
    first_book = documents[0]["book"]
    results = collection.query(
        query_texts=[query], 
        n_results=3,
        where={"book": first_book}
    )
    print(f"   Результаты (только book={first_book}):")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        preview = doc[:60] + "..." if len(doc) > 60 else doc
        print(f"      [{1-dist:.3f}] {preview}")


# =============================================
# Итоговая таблица
# =============================================
print("\n\n" + "=" * 50)
print("📊 СРАВНЕНИЕ")
print("=" * 50)

comparison = f"""
| Критерий              | FAISS              | ChromaDB           |
|-----------------------|--------------------|--------------------|
| Тип                   | In-memory/диск     | SQLite             |
| Скорость поиска       | Очень быстрый      | Быстрый            |
| Метаданные            | ❌ Нет              | ✅ Да               |
| Фильтрация            | ❌ Вручную          | ✅ Встроенная       |
| Персистентность       | Ручное сохранение  | Автоматическая     |
| CRUD операции         | Только add/search  | Полный CRUD        |
| Сложность             | Низкая             | Очень низкая       |
| Масштабируемость      | Миллионы векторов  | Тысячи-сотни тысяч |

КОГДА ИСПОЛЬЗОВАТЬ:
- FAISS: Большие датасеты, максимальная скорость, простой поиск
- ChromaDB: Прототипы, нужны метаданные и фильтры, простота важнее скорости
"""

print(comparison)


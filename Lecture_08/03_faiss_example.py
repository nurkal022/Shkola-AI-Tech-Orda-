"""
Пример 3: FAISS — быстрый векторный поиск от Meta

FAISS (Facebook AI Similarity Search):
- Очень быстрый поиск (миллионы векторов за миллисекунды)
- Работает в памяти
- Можно сохранить индекс на диск
- Нет встроенной поддержки метаданных (храним отдельно)
"""

from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np

load_dotenv()

client = OpenAI()


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Получить эмбеддинги для списка текстов"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([d.embedding for d in response.data], dtype='float32')


# =============================================
# Загружаем данные из книг
# =============================================
from utils import get_all_books_chunks

print("=== FAISS — векторный поиск ===\n")
print("Загружаем данные из книг...")

chunks = get_all_books_chunks("data", chunk_size=500, max_chunks=100)
documents = [
    {
        "text": chunk["text"],
        "book": chunk["book"],
        "chunk_id": chunk["chunk_id"]
    }
    for chunk in chunks
]

print(f"Загружено {len(documents)} чанков\n")

# Шаг 1: Создаём эмбеддинги
print("1. Создаём эмбеддинги...")
print("   (обрабатываем батчами...)")

texts = [doc["text"] for doc in documents]
embeddings_list = []

# Обрабатываем батчами по 100
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    batch_emb = get_embeddings(batch)
    embeddings_list.append(batch_emb)
    print(f"   Обработано: {min(i+batch_size, len(texts))}/{len(texts)}")

embeddings = np.vstack(embeddings_list)
print(f"   Размер матрицы: {embeddings.shape}")

# Шаг 2: Создаём FAISS индекс
print("\n2. Создаём FAISS индекс...")
dimension = embeddings.shape[1]  # 1536

# IndexFlatIP — точный поиск по inner product (косинусное сходство для нормализованных векторов)
# Нормализуем векторы для косинусного сходства
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dimension)

# Добавляем векторы в индекс
index.add(embeddings)
print(f"   Добавлено векторов: {index.ntotal}")

# Шаг 3: Сохраняем индекс на диск (опционально)
faiss.write_index(index, "faiss_index.bin")
print("   Индекс сохранён в faiss_index.bin")


def search(query: str, top_k: int = 3):
    """Поиск похожих документов"""
    # Получаем эмбеддинг запроса
    query_emb = get_embeddings([query])
    faiss.normalize_L2(query_emb)
    
    # Поиск в индексе
    scores, indices = index.search(query_emb, top_k)
    
    # Возвращаем результаты с метаданными
    results = []
    for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
        results.append({
            "text": documents[idx]["text"],
            "book": documents[idx]["book"],
            "chunk_id": documents[idx]["chunk_id"],
            "score": float(score)
        })
    return results


# =============================================
# Демонстрация поиска
# =============================================
print("\n3. Поиск...")

queries = [
    "кто такой Гарри Поттер?",
    "что такое Хогвартс?",
    "кто такой Хагрид?",
]

for query in queries:
    print(f"\n🔍 Запрос: '{query}'")
    results = search(query, top_k=3)
    for r in results:
        preview = r['text'][:80] + "..." if len(r['text']) > 80 else r['text']
        print(f"   [{r['score']:.3f}] [{r['book']}] {preview}")


# Загрузка сохранённого индекса
print("\n\n=== Загрузка индекса с диска ===")
loaded_index = faiss.read_index("faiss_index.bin")
print(f"Загружено векторов: {loaded_index.ntotal}")


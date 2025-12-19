"""
Пример 2: In-memory векторный поиск (без БД)

Показываем механику поиска:
1. Создаём эмбеддинги для документов
2. Создаём эмбеддинг для запроса
3. Находим документы с максимальным сходством
"""

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()

client = OpenAI()


def get_embedding(text: str) -> list[float]:
    """Получить эмбеддинг текста"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2) -> float:
    """Косинусное сходство"""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =============================================
# Загружаем данные из книг о Гарри Поттере
# =============================================
from utils import get_all_books_chunks

print("=== In-memory векторный поиск ===\n")
print("Загружаем данные из книг...")

# Берём первые 50 чанков для быстрого теста
chunks = get_all_books_chunks("data", chunk_size=500, max_chunks=50)
documents = [chunk["text"] for chunk in chunks]

print(f"\nЗагружено {len(documents)} чанков из книг\n")

# Шаг 1: Создаём эмбеддинги для всех документов
print("1. Создаём эмбеддинги для документов...")
print("   (это может занять время...)")

# Используем batch для ускорения
def get_embeddings_batch(texts: list[str], batch_size: int = 100):
    """Получить эмбеддинги батчами"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"   Обработано: {min(i+batch_size, len(texts))}/{len(texts)}")
    return all_embeddings

doc_embeddings = get_embeddings_batch(documents, batch_size=100)
print(f"\n   Готово! {len(doc_embeddings)} эмбеддингов создано\n")


def search(query: str, top_k: int = 3):
    """Поиск топ-k похожих документов"""
    # Шаг 2: Создаём эмбеддинг запроса
    query_emb = get_embedding(query)
    
    # Шаг 3: Вычисляем сходство с каждым документом
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_emb, doc_emb)
        similarities.append((i, sim))
    
    # Сортируем по убыванию
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Возвращаем топ-k
    return [(documents[i], sim) for i, sim in similarities[:top_k]]


# =============================================
# Демонстрация поиска
# =============================================
queries = [
    "кто такой Гарри Поттер?",
    "что такое Хогвартс?",
    "кто такой Хагрид?",
    "как Гарри попал в школу магии?",
]

for query in queries:
    print(f"🔍 Запрос: '{query}'")
    results = search(query, top_k=3)
    print("   Результаты:")
    for doc, score in results:
        # Показываем первые 100 символов
        preview = doc[:100] + "..." if len(doc) > 100 else doc
        print(f"   [{score:.3f}] {preview}")
    print()


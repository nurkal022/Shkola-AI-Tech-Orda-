"""
Пример 1: Основы эмбеддингов

Эмбеддинг (embedding) — это представление текста в виде вектора чисел.
Похожие по смыслу тексты имеют похожие векторы.

Как это работает:
- "кот" → [0.1, 0.5, -0.3, ...]  (1536 чисел для OpenAI)
- "собака" → [0.15, 0.48, -0.25, ...]  (похожий вектор!)
- "автомобиль" → [-0.8, 0.1, 0.9, ...]  (другой вектор)
"""

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()

client = OpenAI()


def get_embedding(text: str) -> list[float]:
    """Получить эмбеддинг текста через OpenAI API"""
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 1536 измерений
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Косинусное сходство между двумя векторами (от -1 до 1)"""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =============================================
# Демонстрация
# =============================================
print("=== Основы эмбеддингов ===\n")

# Получаем эмбеддинги для слов
words = ["кот", "собака", "автомобиль", "машина","вода", "лед", "масло", "огонь"]
embeddings = {}

print("Генерируем эмбеддинги...")
for word in words:
    embeddings[word] = get_embedding(word)
    print(f"  '{word}' → вектор из {len(embeddings[word])} чисел")

# print(f"\nПример первых 5 чисел вектора 'кот': {embeddings['кот']}")

# Сравниваем сходство
print("\n=== Сравнение сходства ===\n")

pairs = [

    ("вода", "масло"), # Синонимы — очень похожи!
    ("вода", "огонь"), # Синонимы — очень похожи!
    ("вода", "лед"), # Синонимы — очень похожи!
]

for word1, word2 in pairs:
    similarity = cosine_similarity(embeddings[word1], embeddings[word2])
    print(f"  '{word1}' ↔ '{word2}': {similarity:.4f}")

print("\n💡 Чем ближе к 1, тем больше сходство по смыслу")


# Поиск похожего слова
print("\n=== Поиск похожего ===\n")

query = "вода"
query_emb = get_embedding(query)

print(f"Запрос: '{query}'")
print("Сравниваем с:", words)

similarities = []
for word in words:
    sim = cosine_similarity(query_emb, embeddings[word])
    similarities.append((word, sim))

# Сортируем по убыванию сходства
similarities.sort(key=lambda x: x[1], reverse=True)

print("\nРезультат (по убыванию сходства):")
for word, sim in similarities:
    print(f"  {word}: {sim:.4f}")


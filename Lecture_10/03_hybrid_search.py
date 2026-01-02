"""
Лекция 10: Hybrid Search
========================
Комбинация BM25 (keyword) + Semantic (vector) поиска.
Объединяет точность ключевых слов и понимание смысла.
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import re
import os

load_dotenv()

# ============================================================
# Загрузка и подготовка данных
# ============================================================
print("="*60)
print("📖 Загрузка данных")
print("="*60)

text = Path("../Lecture_07/data/Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt").read_text(encoding='utf-8')

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = splitter.split_text(text)

print(f"   Загружено: {len(chunks)} чанков\n")


# ============================================================
# Подготовка индексов
# ============================================================
print("="*60)
print("🔧 Подготовка индексов")
print("="*60)

# BM25 индекс
def tokenize(text: str) -> list:
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    return [w for w in words if len(w) > 2]

tokenized_chunks = [tokenize(chunk) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
print("   ✅ BM25 индекс создан")

# Vector индекс
if not os.getenv("OPENAI_API_KEY"):
    print("   ⚠️ OPENAI_API_KEY не найден! Semantic search будет недоступен")
    embeddings = None
    vectorstore = None
else:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("   ✅ Vector индекс создан")

print()


# ============================================================
# Hybrid Search функция
# ============================================================
def hybrid_search(query: str, k: int = 5, alpha: float = 0.5):
    """
    Гибридный поиск: комбинация BM25 и Semantic.
    
    Args:
        query: Поисковый запрос
        k: Количество результатов
        alpha: Вес Semantic Search (0.0 = только BM25, 1.0 = только Semantic)
               Рекомендуется: 0.3-0.5
    
    Returns:
        Список результатов с hybrid_score
    """
    # 1. BM25 scores
    tokenized_query = tokenize(query)
    if tokenized_query:
        bm25_scores = bm25.get_scores(tokenized_query)
        # Нормализация BM25 (0-1)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_norm = [s / max_bm25 for s in bm25_scores]
    else:
        bm25_norm = [0] * len(chunks)
    
    # 2. Semantic scores
    if vectorstore:
        vector_results = vectorstore.similarity_search_with_score(query, k=len(chunks))
        # Создаём словарь: текст -> similarity score
        vector_scores_dict = {}
        for doc, score in vector_results:
            similarity = 1 - score  # Конвертируем distance в similarity
            vector_scores_dict[doc.page_content] = similarity
        
        # Маппим scores к индексам чанков
        vector_norm = [vector_scores_dict.get(chunk, 0) for chunk in chunks]
    else:
        vector_norm = [0] * len(chunks)
    
    # 3. Комбинируем scores
    hybrid_scores = []
    for i in range(len(chunks)):
        # alpha = вес Semantic, (1-alpha) = вес BM25
        hybrid_score = (1 - alpha) * bm25_norm[i] + alpha * vector_norm[i]
        hybrid_scores.append((i, hybrid_score, chunks[i]))
    
    # 4. Сортируем по hybrid_score
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 5. Формируем результаты
    results = []
    for rank, (idx, score, text) in enumerate(hybrid_scores[:k], 1):
        results.append({
            "rank": rank,
            "chunk_id": idx,
            "hybrid_score": score,
            "bm25_score": bm25_norm[idx],
            "semantic_score": vector_norm[idx] if vectorstore else 0,
            "text": text[:200] + "..." if len(text) > 200 else text,
            "full_text": text,
        })
    
    return results


# ============================================================
# Демонстрация
# ============================================================
print("="*60)
print("🔍 ДЕМОНСТРАЦИЯ HYBRID SEARCH")
print("="*60)

test_queries = [
    "волшебная палочка",
    "магический жезл",  # Синоним
    "Гарри Поттер",
    "школа магии",
]

for query in test_queries:
    print(f"\n{'─'*60}")
    print(f"❓ Запрос: «{query}»")
    print(f"{'─'*60}")
    
    # Пробуем разные alpha
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        if alpha == 0.0 and not tokenize(query):
            continue
        if alpha == 1.0 and not vectorstore:
            continue
            
        results = hybrid_search(query, k=1, alpha=alpha)
        if results:
            r = results[0]
            print(f"\n   α={alpha:.1f} (BM25:{(1-alpha):.1f}, Semantic:{alpha:.1f})")
            print(f"   Hybrid: {r['hybrid_score']:.4f} | BM25: {r['bm25_score']:.4f} | Semantic: {r['semantic_score']:.4f}")
            print(f"   📄 {r['text'][:100]}...")


# ============================================================
# Сравнение методов
# ============================================================
print("\n" + "="*60)
print("📊 СРАВНЕНИЕ: BM25 vs Semantic vs Hybrid")
print("="*60)

query = "волшебная палочка"
print(f"\nЗапрос: «{query}»\n")

# BM25
tokenized_query = tokenize(query)
if tokenized_query:
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[0]
    print(f"1️⃣ BM25 (α=0.0):")
    print(f"   Score: {bm25_scores[bm25_top]:.4f}")
    print(f"   Текст: {chunks[bm25_top][:100]}...")

# Semantic
if vectorstore:
    sem_results = vectorstore.similarity_search_with_score(query, k=1)
    if sem_results:
        doc, score = sem_results[0]
        print(f"\n2️⃣ Semantic (α=1.0):")
        print(f"   Distance: {score:.4f} | Similarity: {1-score:.4f}")
        print(f"   Текст: {doc.page_content[:100]}...")

# Hybrid
hybrid_results = hybrid_search(query, k=1, alpha=0.4)
if hybrid_results:
    r = hybrid_results[0]
    print(f"\n3️⃣ Hybrid (α=0.4):")
    print(f"   Hybrid Score: {r['hybrid_score']:.4f}")
    print(f"   (BM25: {r['bm25_score']:.4f} + Semantic: {r['semantic_score']:.4f})")
    print(f"   Текст: {r['text'][:100]}...")

print("\n💡 Вывод: Hybrid Search объединяет лучшие стороны обоих методов!")


# ============================================================
# Как работает Hybrid Search
# ============================================================
print("\n" + "="*60)
print("📚 КАК РАБОТАЕТ HYBRID SEARCH")
print("="*60)
print("""
Архитектура:

                    ┌─────────────┐
                    │   Запрос    │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                          ▼
    ┌─────────────────┐        ┌─────────────────┐
    │   BM25 Search   │        │ Semantic Search │
    │  (keyword-based)│        │ (vector-based)  │
    └────────┬────────┘        └────────┬────────┘
             │                          │
             │ Scores                   │ Scores
             └──────────────┬───────────┘
                            ▼
                   ┌─────────────────┐
                   │  Normalization  │
                   │  (0-1 scale)    │
                   └────────┬────────┘
                            ▼
                   ┌─────────────────┐
                   │  Weighted Sum   │
                   │ hybrid = (1-α)× │
                   │   BM25 + α×Sem  │
                   └────────┬────────┘
                            ▼
                   ┌─────────────────┐
                   │  Top-K Results  │
                   └─────────────────┘

Формула:
  hybrid_score = (1 - α) × normalize(BM25_score) + α × normalize(Semantic_score)

Где:
  • α ∈ [0, 1] - вес Semantic Search
  • Рекомендуется: α = 0.3-0.5 (баланс)

Плюсы Hybrid:
  ✅ Точные совпадения (BM25)
  ✅ Синонимы и контекст (Semantic)
  ✅ Настраиваемый баланс (α)
  ✅ Лучшее качество чем каждый метод отдельно

Минусы Hybrid:
  ❌ Дороже (нужны оба индекса)
  ❌ Медленнее (два поиска)
  ❌ Требует настройки α
""")


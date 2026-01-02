"""
Лекция 10: Сравнение всех методов поиска
========================================
Демонстрация BM25, Semantic, Hybrid и RRF на одних и тех же запросах.
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import re
import os
from collections import defaultdict

load_dotenv()

# ============================================================
# Загрузка данных
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

def tokenize(text: str) -> list:
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    return [w for w in words if len(w) > 2]

tokenized_chunks = [tokenize(chunk) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
print("   ✅ BM25 индекс")

if os.getenv("OPENAI_API_KEY"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("   ✅ Vector индекс")
else:
    vectorstore = None
    print("   ⚠️ Vector индекс недоступен")

print()


# ============================================================
# Функции поиска
# ============================================================

def bm25_search(query: str, k: int = 5):
    """BM25 поиск."""
    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []
    
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    return [{
        "rank": i+1,
        "chunk_id": idx,
        "score": float(scores[idx]),
        "text": chunks[idx][:150] + "..."
    } for i, idx in enumerate(top_indices)]


def semantic_search(query: str, k: int = 5):
    """Semantic поиск."""
    if not vectorstore:
        return []
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [{
        "rank": i+1,
        "score": float(score),
        "similarity": float(1 - score),
        "text": doc.page_content[:150] + "..."
    } for i, (doc, score) in enumerate(results)]


def hybrid_search(query: str, k: int = 5, alpha: float = 0.4):
    """Hybrid поиск (weighted sum)."""
    # BM25
    tokenized_query = tokenize(query)
    if tokenized_query:
        bm25_scores = bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_norm = [s / max_bm25 for s in bm25_scores]
    else:
        bm25_norm = [0] * len(chunks)
    
    # Semantic
    if vectorstore:
        vector_results = vectorstore.similarity_search_with_score(query, k=len(chunks))
        vector_scores_dict = {doc.page_content: 1 - score for doc, score in vector_results}
        vector_norm = [vector_scores_dict.get(chunk, 0) for chunk in chunks]
    else:
        vector_norm = [0] * len(chunks)
    
    # Combine
    hybrid_scores = [
        (i, (1 - alpha) * bm25_norm[i] + alpha * vector_norm[i], chunks[i])
        for i in range(len(chunks))
    ]
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [{
        "rank": i+1,
        "chunk_id": idx,
        "hybrid_score": score,
        "text": text[:150] + "..."
    } for i, (idx, score, text) in enumerate(hybrid_scores[:k])]


def rrf_search(query: str, k: int = 5):
    """RRF поиск."""
    rankings = []
    
    # BM25 ranking
    tokenized_query = tokenize(query)
    if tokenized_query:
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_ranking = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
        rankings.append(bm25_ranking)
    
    # Semantic ranking
    if vectorstore:
        vector_results = vectorstore.similarity_search_with_score(query, k=20)
        semantic_ranking = []
        for doc, _ in vector_results:
            try:
                idx = chunks.index(doc.page_content)
                semantic_ranking.append(idx)
            except ValueError:
                continue
        rankings.append(semantic_ranking)
    
    if not rankings:
        return []
    
    # RRF
    rrf_scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            rrf_scores[doc_id] += 1 / (60 + rank)
    
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return [{
        "rank": i+1,
        "chunk_id": doc_id,
        "rrf_score": score,
        "text": chunks[doc_id][:150] + "..."
    } for i, (doc_id, score) in enumerate(sorted_results)]


# ============================================================
# Сравнение на тестовых запросах
# ============================================================
print("="*60)
print("📊 СРАВНЕНИЕ ВСЕХ МЕТОДОВ")
print("="*60)

test_queries = [
    ("волшебная палочка", "Точный запрос"),
    ("магический жезл", "Синоним"),
    ("Гарри Поттер", "Имя персонажа"),
    ("школа для волшебников", "Парафраз"),
]

for query, query_type in test_queries:
    print(f"\n{'='*60}")
    print(f"❓ Запрос: «{query}» ({query_type})")
    print(f"{'='*60}")
    
    # BM25
    print("\n1️⃣ BM25 (Keyword):")
    bm25_results = bm25_search(query, k=2)
    if bm25_results:
        for r in bm25_results:
            print(f"   [{r['rank']}] Score: {r['score']:.4f}")
            print(f"      {r['text']}")
    else:
        print("   ⚠️ Нет результатов")
    
    # Semantic
    print("\n2️⃣ Semantic (Vector):")
    sem_results = semantic_search(query, k=2)
    if sem_results:
        for r in sem_results:
            print(f"   [{r['rank']}] Similarity: {r['similarity']:.4f}")
            print(f"      {r['text']}")
    else:
        print("   ⚠️ Нет результатов (требуется API ключ)")
    
    # Hybrid
    print("\n3️⃣ Hybrid (Weighted, α=0.4):")
    hybrid_results = hybrid_search(query, k=2, alpha=0.4)
    if hybrid_results:
        for r in hybrid_results:
            print(f"   [{r['rank']}] Hybrid Score: {r['hybrid_score']:.4f}")
            print(f"      {r['text']}")
    else:
        print("   ⚠️ Нет результатов")
    
    # RRF
    print("\n4️⃣ RRF (Rank Fusion):")
    rrf_results = rrf_search(query, k=2)
    if rrf_results:
        for r in rrf_results:
            print(f"   [{r['rank']}] RRF Score: {r['rrf_score']:.6f}")
            print(f"      {r['text']}")
    else:
        print("   ⚠️ Нет результатов")


# ============================================================
# Сводная таблица
# ============================================================
print("\n" + "="*60)
print("📋 СВОДНАЯ ТАБЛИЦА МЕТОДОВ")
print("="*60)
print("""
┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Метод            │ Точные совп. │ Синонимы     │ Парафразы    │ Скорость     │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ BM25             │ ⭐⭐⭐⭐⭐     │ ⭐           │ ⭐           │ ⚡⚡⚡⚡⚡     │
│ Semantic         │ ⭐⭐⭐        │ ⭐⭐⭐⭐⭐     │ ⭐⭐⭐⭐⭐     │ ⚡⚡⚡        │
│ Hybrid (α=0.4)   │ ⭐⭐⭐⭐       │ ⭐⭐⭐⭐       │ ⭐⭐⭐⭐       │ ⚡⚡⚡        │
│ RRF              │ ⭐⭐⭐⭐       │ ⭐⭐⭐⭐       │ ⭐⭐⭐⭐       │ ⚡⚡⚡        │
└──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

┌──────────────────┬──────────────┬──────────────┬──────────────┐
│ Метод            │ Сложность    │ Требования   │ Когда использовать │
├──────────────────┼──────────────┼──────────────┼────────────────────┤
│ BM25             │ Низкая       │ rank-bm25    │ Быстрый прототип   │
│ Semantic         │ Средняя      │ Embeddings   │ Синонимы важны     │
│ Hybrid           │ Средняя      │ Оба          │ Баланс качества    │
│ RRF              │ Низкая       │ Оба          │ Много источников   │
└──────────────────┴──────────────┴──────────────┴────────────────────┘
""")


# ============================================================
# Рекомендации
# ============================================================
print("\n" + "="*60)
print("💡 РЕКОМЕНДАЦИИ")
print("="*60)
print("""
1. Начните с BM25:
   - Быстро и бесплатно
   - Отлично для точных совпадений
   - Не требует ML моделей

2. Добавьте Semantic если:
   - Нужны синонимы
   - Пользователи задают вопросы по-разному
   - Есть бюджет на embeddings

3. Используйте Hybrid когда:
   - Нужен баланс точности и понимания
   - Есть оба индекса
   - Важно качество поиска

4. Выберите RRF если:
   - Много источников поиска (>2)
   - Scores несопоставимы
   - Нужна простота объединения

5. Оптимальная конфигурация:
   - BM25 + Semantic + RRF
   - Лучшее качество на практике
   - Устойчивость к выбросам
""")


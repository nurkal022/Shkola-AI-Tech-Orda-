"""
Лекция 10: Reciprocal Rank Fusion (RRF)
========================================
RRF объединяет результаты из разных источников поиска,
используя только ранги (позиции), а не scores.
Решает проблему несопоставимости scores от разных методов.
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

# BM25
def tokenize(text: str) -> list:
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    return [w for w in words if len(w) > 2]

tokenized_chunks = [tokenize(chunk) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
print("   ✅ BM25 индекс создан")

# Vector
if os.getenv("OPENAI_API_KEY"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("   ✅ Vector индекс создан")
else:
    vectorstore = None
    print("   ⚠️ Vector индекс недоступен (нет API ключа)")

print()


# ============================================================
# RRF функция
# ============================================================
def reciprocal_rank_fusion(rankings: list, k: int = 60):
    """
    Reciprocal Rank Fusion - объединяет ранги из разных источников.
    
    Args:
        rankings: Список списков рангов, например:
                 [
                   [doc_id1, doc_id2, doc_id3],  # BM25 ranking
                   [doc_id3, doc_id1, doc_id4],  # Semantic ranking
                 ]
        k: Константа сглаживания (обычно 60)
    
    Returns:
        Список (doc_id, rrf_score) отсортированный по убыванию score
    """
    # Словарь: doc_id -> RRF score
    rrf_scores = defaultdict(float)
    
    # Для каждого источника рангов
    for ranking in rankings:
        # Для каждого документа в ранге
        for rank, doc_id in enumerate(ranking, 1):
            # RRF формула: 1 / (k + rank)
            rrf_scores[doc_id] += 1 / (k + rank)
    
    # Сортируем по убыванию RRF score
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results


# ============================================================
# Функция поиска с RRF
# ============================================================
def rrf_search(query: str, k: int = 5, top_n_per_source: int = 20):
    """
    Поиск с использованием RRF для объединения BM25 и Semantic.
    
    Args:
        query: Поисковый запрос
        k: Количество финальных результатов
        top_n_per_source: Сколько результатов брать из каждого источника
    
    Returns:
        Список результатов с RRF score
    """
    rankings = []
    
    # 1. BM25 ranking
    tokenized_query = tokenize(query)
    if tokenized_query:
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_ranking = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_n_per_source]
        rankings.append(bm25_ranking)
        print(f"   📊 BM25: {len(bm25_ranking)} результатов")
    
    # 2. Semantic ranking
    if vectorstore:
        vector_results = vectorstore.similarity_search_with_score(query, k=top_n_per_source)
        semantic_ranking = []
        for doc, _ in vector_results:
            # Находим индекс чанка
            try:
                idx = chunks.index(doc.page_content)
                semantic_ranking.append(idx)
            except ValueError:
                # Если точного совпадения нет, пропускаем
                continue
        rankings.append(semantic_ranking)
        print(f"   📊 Semantic: {len(semantic_ranking)} результатов")
    
    if not rankings:
        return []
    
    # 3. RRF объединение
    rrf_results = reciprocal_rank_fusion(rankings, k=60)
    
    # 4. Формируем финальные результаты
    results = []
    for doc_id, rrf_score in rrf_results[:k]:
        results.append({
            "rank": len(results) + 1,
            "chunk_id": doc_id,
            "rrf_score": rrf_score,
            "text": chunks[doc_id][:200] + "..." if len(chunks[doc_id]) > 200 else chunks[doc_id],
            "full_text": chunks[doc_id],
        })
    
    return results


# ============================================================
# Демонстрация
# ============================================================
print("="*60)
print("🔍 ДЕМОНСТРАЦИЯ RRF FUSION")
print("="*60)

test_queries = [
    "волшебная палочка",
    "магический жезл",
    "Гарри Поттер",
    "школа магии",
]

for query in test_queries:
    print(f"\n{'─'*60}")
    print(f"❓ Запрос: «{query}»")
    print(f"{'─'*60}")
    
    results = rrf_search(query, k=3)
    
    if not results:
        print("   ⚠️ Результаты не найдены")
        continue
    
    for result in results:
        print(f"\n   [{result['rank']}] RRF Score: {result['rrf_score']:.6f}")
        print(f"   📄 {result['text']}")


# ============================================================
# Сравнение: Hybrid vs RRF
# ============================================================
print("\n" + "="*60)
print("📊 СРАВНЕНИЕ: Hybrid (weighted) vs RRF")
print("="*60)

query = "волшебная палочка"
print(f"\nЗапрос: «{query}»\n")

# Hybrid Search (weighted sum) - локальная реализация для сравнения
def hybrid_search_local(query: str, k: int = 5, alpha: float = 0.4):
    """Локальная реализация hybrid search для сравнения."""
    tokenized_query = tokenize(query)
    if tokenized_query:
        bm25_scores = bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_norm = [s / max_bm25 for s in bm25_scores]
    else:
        bm25_norm = [0] * len(chunks)
    
    if vectorstore:
        vector_results = vectorstore.similarity_search_with_score(query, k=len(chunks))
        vector_scores_dict = {doc.page_content: 1 - score for doc, score in vector_results}
        vector_norm = [vector_scores_dict.get(chunk, 0) for chunk in chunks]
    else:
        vector_norm = [0] * len(chunks)
    
    hybrid_scores = [
        (i, (1 - alpha) * bm25_norm[i] + alpha * vector_norm[i], bm25_norm[i], vector_norm[i])
        for i in range(len(chunks))
    ]
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [{
        "rank": i+1,
        "hybrid_score": score,
        "bm25_score": bm25_score,
        "semantic_score": sem_score,
    } for i, (_, score, bm25_score, sem_score) in enumerate(hybrid_scores[:k])]

hybrid_results = hybrid_search_local(query, k=3, alpha=0.4)
print("1️⃣ Hybrid Search (weighted sum, α=0.4):")
for r in hybrid_results:
    print(f"   [{r['rank']}] Hybrid: {r['hybrid_score']:.4f} | BM25: {r['bm25_score']:.4f} | Semantic: {r['semantic_score']:.4f}")

# RRF
rrf_results = rrf_search(query, k=3)
print("\n2️⃣ RRF Fusion (rank-based):")
for r in rrf_results:
    print(f"   [{r['rank']}] RRF Score: {r['rrf_score']:.6f}")

print("\n💡 RRF не требует нормализации scores - работает только с рангами!")


# ============================================================
# Как работает RRF
# ============================================================
print("\n" + "="*60)
print("📚 КАК РАБОТАЕТ RRF")
print("="*60)
print("""
Проблема: Scores от разных источников несопоставимы
  • BM25: scores могут быть 0-100+
  • Semantic: cosine distance 0-2, similarity 0-1
  • Как их объединить?

Решение RRF: Использовать только РАНГИ, не scores!

Формула RRF:
  RRF_score(d) = Σ 1 / (k + rank_i(d))
  
Где:
  • k = 60 (константа сглаживания)
  • rank_i(d) = позиция документа d в i-м поиске
  • Суммируем по всем источникам

Пример:
  Документ A:
    - BM25: rank = 1  → RRF += 1/(60+1) = 0.0164
    - Semantic: rank = 5 → RRF += 1/(60+5) = 0.0154
    - Итого: RRF(A) = 0.0318
  
  Документ B:
    - BM25: rank = 10 → RRF += 1/(60+10) = 0.0143
    - Semantic: rank = 2 → RRF += 1/(60+2) = 0.0161
    - Итого: RRF(B) = 0.0304
  
  Результат: A > B (A стабильно в топе обоих поисков)

Плюсы RRF:
  ✅ Не требует нормализации scores
  ✅ Работает с любым количеством источников
  ✅ Устойчив к выбросам
  ✅ Простая формула
  ✅ Хорошо работает на практике

Минусы RRF:
  ❌ Игнорирует абсолютные scores (только порядок)
  ❌ Может терять информацию о "качестве" совпадения
  ❌ Требует достаточного количества источников

Когда использовать:
  • Несколько разных источников поиска
  • Scores несопоставимы
  • Нужна простота и надёжность
  • Много источников (>2)
""")


# ============================================================
# Визуализация RRF
# ============================================================
print("\n" + "="*60)
print("📈 ВИЗУАЛИЗАЦИЯ RRF")
print("="*60)

# Пример с конкретными рангами
example_rankings = [
    [0, 1, 2, 3, 4],  # BM25: doc 0 на 1 месте, doc 1 на 2 месте...
    [2, 0, 4, 1, 3],  # Semantic: doc 2 на 1 месте, doc 0 на 2 месте...
]

print("\nПример рангов:")
print(f"  BM25:    {example_rankings[0]}")
print(f"  Semantic: {example_rankings[1]}")

rrf_scores = reciprocal_rank_fusion(example_rankings, k=60)

print("\nRRF Scores:")
for doc_id, score in rrf_scores:
    bm25_rank = example_rankings[0].index(doc_id) + 1 if doc_id in example_rankings[0] else 999
    sem_rank = example_rankings[1].index(doc_id) + 1 if doc_id in example_rankings[1] else 999
    print(f"  Doc {doc_id}: RRF={score:.6f} (BM25 rank={bm25_rank}, Semantic rank={sem_rank})")

print("\n💡 Документ 0 и 2 получили высокие RRF scores,")
print("   потому что они в топе обоих поисков!")


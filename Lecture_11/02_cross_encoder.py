"""
Лекция 11: Cross-Encoder Reranking (Детальный пример)
======================================================
Практический пример использования Cross-Encoder для reranking.
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import time

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

# Векторный индекс
if os.getenv("OPENAI_API_KEY"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("   ✅ Vector индекс создан\n")
else:
    vectorstore = None
    print("   ⚠️ Vector индекс недоступен\n")


# ============================================================
# Установка Cross-Encoder
# ============================================================
print("="*60)
print("🔧 Установка Cross-Encoder")
print("="*60)

try:
    from sentence_transformers import CrossEncoder
    
    # Загружаем модель (скачивается при первом запуске)
    print("   ⏳ Загрузка модели cross-encoder/ms-marco-MiniLM-L-6-v2...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("   ✅ Модель загружена\n")
    
except ImportError:
    print("   ❌ sentence-transformers не установлен")
    print("   Установите: pip install sentence-transformers")
    exit(1)


# ============================================================
# Функция reranking
# ============================================================
def rerank_with_cross_encoder(query: str, documents: list, top_k: int = 5):
    """
    Reranking документов с помощью Cross-Encoder.
    
    Args:
        query: Поисковый запрос
        documents: Список Document объектов
        top_k: Количество возвращаемых документов
    
    Returns:
        Tuple: (reranked_documents, scores)
    """
    if not documents:
        return [], []
    
    # Создаём пары (query, document_content)
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Получаем scores (чем больше, тем лучше)
    scores = reranker.predict(pairs)
    
    # Сортируем по убыванию score
    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Возвращаем top_k
    reranked_docs = [doc for doc, score in ranked[:top_k]]
    reranked_scores = [float(score) for doc, score in ranked[:top_k]]
    
    return reranked_docs, reranked_scores


# ============================================================
# Демонстрация: До и После
# ============================================================
print("="*60)
print("🔍 ДЕМОНСТРАЦИЯ: До и После Reranking")
print("="*60)

test_queries = [
    "Как выглядит шрам Гарри Поттера?",
    "Кто такие Дурсли?",
    "Что такое философский камень?",
]

if not vectorstore:
    print("   ⚠️ Vector store недоступен (нужен OPENAI_API_KEY)")
    exit(1)

for query in test_queries:
    print(f"\n{'─'*60}")
    print(f"❓ Запрос: «{query}»")
    print(f"{'─'*60}")
    
    # 1. Базовый retrieval (top-10)
    start = time.time()
    results = vectorstore.similarity_search_with_score(query, k=10)
    retrieval_time = time.time() - start
    
    documents = [
        Document(
            page_content=doc.page_content,
            metadata={"similarity": 1 - score, "original_rank": i+1}
        )
        for i, (doc, score) in enumerate(results)
    ]
    
    print(f"\n📊 ДО reranking (top-5 по similarity):")
    for i, doc in enumerate(documents[:5], 1):
        similarity = doc.metadata["similarity"]
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"   [{i}] Similarity: {similarity:.4f}")
        print(f"       {preview}...")
    
    # 2. Reranking
    start = time.time()
    reranked, rerank_scores = rerank_with_cross_encoder(query, documents, top_k=5)
    rerank_time = time.time() - start
    
    print(f"\n📊 ПОСЛЕ reranking (top-5 по релевантности):")
    for i, (doc, score) in enumerate(zip(reranked, rerank_scores), 1):
        original_rank = doc.metadata["original_rank"]
        preview = doc.page_content[:80].replace('\n', ' ')
        change = "↑" if original_rank > i else "↓" if original_rank < i else "="
        print(f"   [{i}] Rerank Score: {score:.4f} {change} (был #{original_rank})")
        print(f"       {preview}...")
    
    print(f"\n⏱️ Время: Retrieval={retrieval_time:.3f}s, Reranking={rerank_time:.3f}s")


# ============================================================
# Анализ изменений
# ============================================================
print("\n" + "="*60)
print("📈 АНАЛИЗ: Как изменился порядок")
print("="*60)

query = "Как выглядит шрам Гарри Поттера?"
results = vectorstore.similarity_search_with_score(query, k=10)

documents = [
    Document(
        page_content=doc.page_content,
        metadata={"similarity": 1 - score, "original_rank": i+1}
    )
    for i, (doc, score) in enumerate(results)
]

reranked, rerank_scores = rerank_with_cross_encoder(query, documents, top_k=10)

print("\nСравнение позиций:")
print("─" * 60)
print(f"{'До':<5} {'После':<7} {'Изменение':<12} {'Similarity':<12} {'Rerank Score'}")
print("─" * 60)

for i, (doc, rerank_score) in enumerate(zip(reranked, rerank_scores), 1):
    original_rank = doc.metadata["original_rank"]
    similarity = doc.metadata["similarity"]
    change = original_rank - i
    change_str = f"{change:+d}" if change != 0 else "0"
    
    print(f"{original_rank:<5} {i:<7} {change_str:<12} {similarity:<12.4f} {rerank_score:.4f}")


# ============================================================
# Производительность
# ============================================================
print("\n" + "="*60)
print("⚡ ПРОИЗВОДИТЕЛЬНОСТЬ")
print("="*60)

query = "волшебная палочка"
results = vectorstore.similarity_search_with_score(query, k=20)
documents = [Document(page_content=doc.page_content) for doc, _ in results]

# Тест на разном количестве документов
for k in [5, 10, 20]:
    start = time.time()
    reranked, _ = rerank_with_cross_encoder(query, documents[:k], top_k=k)
    elapsed = time.time() - start
    
    print(f"   {k:2d} документов: {elapsed:.3f}s ({elapsed/k*1000:.1f}ms на документ)")


# ============================================================
# Рекомендации
# ============================================================
print("\n" + "="*60)
print("💡 РЕКОМЕНДАЦИИ")
print("="*60)
print("""
1. Когда использовать Cross-Encoder:
   ✅ Нужна точность ранжирования
   ✅ Есть GPU (ускоряет в 10-100 раз)
   ✅ Бюджет ограничен (бесплатно)
   ✅ Retriever возвращает 10-50 документов

2. Оптимальная стратегия:
   • Retriever: top-20 (быстро, широкий охват)
   • Reranker: top-5 (точно, узкий фокус)
   • LLM: получает только самые релевантные

3. Оптимизация:
   • Используйте GPU если доступен
   • Батчинг для множественных запросов
   • Кешируйте результаты для одинаковых запросов
   • Рассмотрите более быстрые модели для больших объёмов

4. Альтернативные модели:
   • cross-encoder/ms-marco-MiniLM-L-6-v2 (быстрая, хорошая)
   • cross-encoder/ms-marco-MiniLM-L-12-v2 (медленнее, точнее)
   • cross-encoder/ms-marco-electra-base (самая точная, медленная)
""")

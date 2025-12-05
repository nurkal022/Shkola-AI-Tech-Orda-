"""
RAG: Сравнение методов поиска
=============================
Демонстрация разных методов поиска БЕЗ генерации.
Показываем только извлеченные документы.
"""
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================
print("="*60)
print("📚 ПОДГОТОВКА ДАННЫХ")
print("="*60)

# Загружаем первую книгу
file = list(Path("data").glob("*.txt"))[0]
text = file.read_text(encoding='utf-8')

# Разбиваем на чанки
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)[:200]  # 200 чанков для демо

print(f"📖 Загружено: {len(chunks)} чанков")

# Embeddings для векторного поиска
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Создаем FAISS индекс
print("🔄 Создание векторного индекса...")
vectorstore = FAISS.from_texts(chunks, embeddings)
print("✅ Готово!\n")


# ============================================================
# 1. KEYWORD SEARCH (BM25)
# ============================================================
def keyword_search(query: str, k: int = 5):
    """
    BM25 - классический поиск по ключевым словам.
    Ищет точные совпадения слов, учитывает частоту.
    """
    # Токенизация
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Поиск
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Топ результаты
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": idx,
            "score": scores[idx],
            "text": chunks[idx]
        })
    return results


# ============================================================
# 2. VECTOR SEARCH (Semantic)
# ============================================================
def vector_search(query: str, k: int = 5):
    """
    Векторный поиск - семантическое сходство.
    Находит похожие по смыслу, даже без точных слов.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    return [{
        "chunk_id": chunks.index(doc.page_content) if doc.page_content in chunks else -1,
        "score": float(score),
        "text": doc.page_content
    } for doc, score in results]


# ============================================================
# 3. MMR SEARCH (Diversity)
# ============================================================
def mmr_search(query: str, k: int = 5):
    """
    MMR - Maximum Marginal Relevance.
    Баланс между релевантностью и разнообразием.
    """
    results = vectorstore.max_marginal_relevance_search(
        query, k=k, fetch_k=20, lambda_mult=0.5
    )
    
    return [{
        "chunk_id": chunks.index(doc.page_content) if doc.page_content in chunks else -1,
        "text": doc.page_content
    } for doc in results]


# ============================================================
# 4. HYBRID SEARCH (Keyword + Vector)
# ============================================================
def hybrid_search(query: str, k: int = 5, alpha: float = 0.5):
    """
    Гибридный поиск - комбинация BM25 и векторного.
    alpha: 0 = только BM25, 1 = только векторный
    """
    # BM25 scores
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(query.lower().split())
    
    # Нормализация BM25
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_norm = [s / max_bm25 for s in bm25_scores]
    
    # Vector scores
    vector_results = vectorstore.similarity_search_with_score(query, k=len(chunks))
    vector_scores = {doc.page_content: 1 - (score / 2) for doc, score in vector_results}  # Инвертируем
    
    # Комбинируем
    combined = []
    for i, chunk in enumerate(chunks):
        vec_score = vector_scores.get(chunk, 0)
        hybrid_score = (1 - alpha) * bm25_norm[i] + alpha * vec_score
        combined.append((i, hybrid_score, chunk))
    
    # Сортируем
    combined.sort(key=lambda x: x[1], reverse=True)
    
    return [{
        "chunk_id": idx,
        "score": score,
        "text": text
    } for idx, score, text in combined[:k]]


# ============================================================
# ДЕМОНСТРАЦИЯ
# ============================================================
def show_results(results, method_name):
    """Красивый вывод результатов"""
    print(f"\n{'='*60}")
    print(f"🔍 {method_name}")
    print(f"{'='*60}")
    
    for i, r in enumerate(results, 1):
        score_str = f" (score: {r['score']:.3f})" if 'score' in r else ""
        print(f"\n📄 Результат {i}{score_str}")
        print(f"   Chunk ID: {r['chunk_id']}")
        print(f"   Текст: {r['text'][:300]}...")


# ============================================================
# ТЕСТ 1: Точный запрос (есть в тексте)
# ============================================================
print("\n" + "="*60)
print("🧪 ТЕСТ 1: Точный запрос")
print("="*60)

query1 = "Гарри Поттер"
print(f"\n❓ Запрос: '{query1}'")

show_results(keyword_search(query1, k=3), "1️⃣ KEYWORD (BM25)")
show_results(vector_search(query1, k=3), "2️⃣ VECTOR (Semantic)")
show_results(mmr_search(query1, k=3), "3️⃣ MMR (Diversity)")
show_results(hybrid_search(query1, k=3), "4️⃣ HYBRID (BM25 + Vector)")


# ============================================================
# ТЕСТ 2: Семантический запрос (синонимы)
# ============================================================
print("\n\n" + "="*60)
print("🧪 ТЕСТ 2: Семантический запрос (синонимы)")
print("="*60)

query2 = "мальчик волшебник"  # Нет точных слов, но есть смысл
print(f"\n❓ Запрос: '{query2}'")

show_results(keyword_search(query2, k=3), "1️⃣ KEYWORD (BM25)")
show_results(vector_search(query2, k=3), "2️⃣ VECTOR (Semantic)")


# ============================================================
# ТЕСТ 3: Вопрос
# ============================================================
print("\n\n" + "="*60)
print("🧪 ТЕСТ 3: Вопросительный запрос")
print("="*60)

query3 = "Кто такой Дамблдор?"
print(f"\n❓ Запрос: '{query3}'")

show_results(keyword_search(query3, k=3), "1️⃣ KEYWORD (BM25)")
show_results(vector_search(query3, k=3), "2️⃣ VECTOR (Semantic)")
show_results(hybrid_search(query3, k=3, alpha=0.7), "4️⃣ HYBRID (alpha=0.7)")


# ============================================================
# СРАВНЕНИЕ МЕТОДОВ
# ============================================================
print("\n\n" + "="*60)
print("📊 КОГДА КАКОЙ МЕТОД ИСПОЛЬЗОВАТЬ")
print("="*60)
print("""
┌─────────────────┬────────────────────────────────────────┐
│ Метод           │ Лучше всего для                        │
├─────────────────┼────────────────────────────────────────┤
│ KEYWORD (BM25)  │ Точные термины, имена, коды            │
│                 │ "error 404", "Python 3.12"             │
├─────────────────┼────────────────────────────────────────┤
│ VECTOR          │ Смысловой поиск, синонимы              │
│                 │ "как исправить баг" → "debugging"      │
├─────────────────┼────────────────────────────────────────┤
│ MMR             │ Нужны разнообразные результаты         │
│                 │ Избежать дубликатов                    │
├─────────────────┼────────────────────────────────────────┤
│ HYBRID          │ Универсальный, лучший для production   │
│                 │ Комбинирует точность и семантику       │
└─────────────────┴────────────────────────────────────────┘
""")


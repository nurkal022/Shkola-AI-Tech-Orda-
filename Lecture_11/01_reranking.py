"""
Лекция 11: Reranking (Переранжирование)
========================================
Проблема: Retriever возвращает документы по similarity,
но порядок не всегда соответствует релевантности к вопросу.
Решение: Второй проход - reranking для уточнения порядка.
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# Загрузка данных и создание базового retriever
# ============================================================
print("="*60)
print("📖 Подготовка данных")
print("="*60)

text = Path("../Lecture_07/data/Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt").read_text(encoding='utf-8')

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = splitter.split_text(text)

print(f"   Загружено: {len(chunks)} чанков\n")

# Создаём векторный индекс
if os.getenv("OPENAI_API_KEY"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("   ✅ Vector индекс создан\n")
else:
    print("   ⚠️ OPENAI_API_KEY не найден\n")
    vectorstore = None


# ============================================================
# Базовый retrieval (до reranking)
# ============================================================
def retrieve_documents(query: str, k: int = 10):
    """Базовый retrieval без reranking."""
    if not vectorstore:
        return []
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    documents = [Document(page_content=doc.page_content, metadata={"score": score}) for doc, score in results]
    return documents


# ============================================================
# Демонстрация проблемы
# ============================================================
print("="*60)
print("🔍 ПРОБЛЕМА: Порядок по similarity ≠ релевантность")
print("="*60)

query = "Как выглядит шрам Гарри Поттера?"

if vectorstore:
    results = retrieve_documents(query, k=5)
    
    print(f"\nЗапрос: «{query}»\n")
    print("Документы ДО reranking (по similarity):")
    print("─" * 60)
    
    for i, doc in enumerate(results, 1):
        score = doc.metadata.get("score", 0)
        similarity = 1 - score
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"\n[{i}] Similarity: {similarity:.4f}")
        print(f"    {preview}...")
    
    print("\n⚠️ Проблема: Документ #1 может быть не самым релевантным!")
    print("   Нужен reranking для уточнения порядка.")


# ============================================================
# Метод 1: Cross-Encoder Reranking
# ============================================================
print("\n" + "="*60)
print("1️⃣ Cross-Encoder Reranking")
print("="*60)

print("""
Cross-Encoder оценивает пару (query, document) напрямую,
а не через отдельные эмбеддинги.

Установка:
  pip install sentence-transformers

Преимущества:
  ✅ Более точное ранжирование
  ✅ Учитывает взаимодействие query-document
  ✅ Быстрее чем LLM reranking

Недостатки:
  ❌ Медленнее чем простой similarity
  ❌ Требует GPU для больших объёмов
""")

try:
    from sentence_transformers import CrossEncoder
    
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def cross_encoder_rerank(query: str, documents: list, top_k: int = 3):
        """Reranking с помощью Cross-Encoder."""
        # Создаём пары (query, document)
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Получаем scores
        scores = reranker.predict(pairs)
        
        # Сортируем по score (больше = лучше)
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, score in ranked[:top_k]], [score for doc, score in ranked[:top_k]]
    
    if vectorstore:
        print("\n   Демонстрация Cross-Encoder reranking:")
        results = retrieve_documents(query, k=10)
        reranked, scores = cross_encoder_rerank(query, results, top_k=3)
        
        print(f"\n   Запрос: «{query}»")
        print(f"   Top-3 после reranking:\n")
        
        for i, (doc, score) in enumerate(zip(reranked, scores), 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   [{i}] Rerank Score: {score:.4f}")
            print(f"       {preview}...")
    
except ImportError:
    print("\n   ⚠️ sentence-transformers не установлен")
    print("   Установите: pip install sentence-transformers")


# ============================================================
# Метод 2: Cohere Rerank API
# ============================================================
print("\n" + "="*60)
print("2️⃣ Cohere Rerank API")
print("="*60)

print("""
Cohere предоставляет готовый API для reranking.

Установка:
  pip install cohere

Преимущества:
  ✅ Готовое решение (не нужно обучать модель)
  ✅ Поддерживает много языков
  ✅ Хорошее качество

Недостатки:
  ❌ Требует API ключ (платно)
  ❌ Зависимость от внешнего сервиса
""")

try:
    import cohere
    
    if os.getenv("COHERE_API_KEY"):
        co = cohere.Client(os.getenv("COHERE_API_KEY"))
        
        def cohere_rerank(query: str, documents: list, top_k: int = 3):
            """Reranking с помощью Cohere API."""
            results = co.rerank(
                query=query,
                documents=[doc.page_content for doc in documents],
                top_n=top_k,
                model="rerank-multilingual-v3.0"
            )
            
            reranked = [documents[r.index] for r in results.results]
            scores = [r.relevance_score for r in results.results]
            
            return reranked, scores
        
        if vectorstore:
            print("\n   Демонстрация Cohere reranking:")
            results = retrieve_documents(query, k=10)
            reranked, scores = cohere_rerank(query, results, top_k=3)
            
            print(f"\n   Запрос: «{query}»")
            print(f"   Top-3 после reranking:\n")
            
            for i, (doc, score) in enumerate(zip(reranked, scores), 1):
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   [{i}] Relevance: {score:.4f}")
                print(f"       {preview}...")
    else:
        print("\n   ⚠️ COHERE_API_KEY не найден в .env")
        print("   Для демонстрации добавьте ключ в .env файл")
        
except ImportError:
    print("\n   ⚠️ cohere не установлен")
    print("   Установите: pip install cohere")


# ============================================================
# Метод 3: LLM-based Reranking
# ============================================================
print("\n" + "="*60)
print("3️⃣ LLM-based Reranking")
print("="*60)

print("""
Используем LLM для оценки релевантности документов.

Преимущества:
  ✅ Очень точное ранжирование
  ✅ Понимает контекст и нюансы
  ✅ Можно кастомизировать промпт

Недостатки:
  ❌ Медленно (много API вызовов)
  ❌ Дорого
  ❌ Может быть нестабильным
""")

if os.getenv("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def llm_rerank(query: str, documents: list, top_k: int = 3):
        """Reranking с помощью LLM."""
        # Создаём промпт
        docs_text = "\n\n".join([
            f"[{i+1}] {doc.page_content[:300]}..."
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""Отранжируй документы по релевантности к вопросу.

Вопрос: {query}

Документы:
{docs_text}

Верни только номера документов в порядке релевантности (от самого релевантного к наименее релевантному), через запятую.
Например: 3, 1, 5, 2, 4

Номера:"""
        
        response = llm.invoke(prompt)
        
        # Парсим ответ
        try:
            ranked_indices = [int(x.strip()) - 1 for x in response.content.split(',')]
            reranked = [documents[i] for i in ranked_indices[:top_k] if 0 <= i < len(documents)]
            return reranked
        except:
            # Если парсинг не удался, возвращаем оригинальный порядок
            return documents[:top_k]
    
    if vectorstore:
        print("\n   Демонстрация LLM reranking:")
        results = retrieve_documents(query, k=5)  # Меньше для экономии токенов
        
        print(f"\n   Запрос: «{query}»")
        print("   ⏳ Обработка LLM...")
        
        reranked = llm_rerank(query, results, top_k=3)
        
        print(f"   Top-3 после reranking:\n")
        for i, doc in enumerate(reranked, 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   [{i}] {preview}...")
else:
    print("\n   ⚠️ OPENAI_API_KEY не найден")


# ============================================================
# Сравнение методов
# ============================================================
print("\n" + "="*60)
print("📊 СРАВНЕНИЕ МЕТОДОВ RERANKING")
print("="*60)
print("""
┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Метод            │ Точность     │ Скорость     │ Стоимость   │ Сложность    │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Cross-Encoder    │ ⭐⭐⭐⭐      │ ⚡⚡⚡        │ Бесплатно    │ Средняя      │
│ Cohere API       │ ⭐⭐⭐⭐⭐     │ ⚡⚡⚡⚡      │ Платно       │ Низкая       │
│ LLM-based        │ ⭐⭐⭐⭐⭐     │ ⚡           │ Дорого       │ Высокая      │
└──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

Рекомендации:
  • Для продакшена: Cohere Rerank или Cross-Encoder
  • Для экспериментов: LLM-based
  • Для быстрого прототипа: Cross-Encoder (бесплатно)
""")


# ============================================================
# Когда использовать Reranking
# ============================================================
print("\n" + "="*60)
print("💡 КОГДА ИСПОЛЬЗОВАТЬ RERANKING")
print("="*60)
print("""
✅ Используйте reranking когда:
  • Важна точность порядка документов
  • Retriever возвращает много документов (k > 5)
  • Есть бюджет на дополнительную обработку
  • Критична релевантность для LLM

❌ Не нужен reranking когда:
  • Retriever уже очень точный
  • Мало документов (k <= 3)
  • Критична скорость (real-time)
  • Нет бюджета на дополнительную обработку

Оптимальная стратегия:
  1. Retriever возвращает top-20
  2. Reranker отбирает top-5
  3. LLM получает только самые релевантные
""")

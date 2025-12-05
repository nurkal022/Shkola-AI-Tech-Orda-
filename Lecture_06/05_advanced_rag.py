"""
RAG Шаг 5: Продвинутые техники
==============================
Сравнение методов улучшения качества RAG.
"""
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# Загрузка компонентов
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

INDEX_PATH = "./faiss_harry_potter"
if not Path(INDEX_PATH).exists():
    print("❌ Сначала запустите 04_rag_pipeline.py для создания индекса")
    exit()

vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
print("✅ Индекс загружен\n")


# ============================================================
# 1. БАЗОВЫЙ ПОИСК (для сравнения)
# ============================================================
def basic_search(query: str, k: int = 4):
    """Обычный векторный поиск"""
    return vectorstore.similarity_search(query, k=k)


# ============================================================
# 2. MULTI-QUERY: Генерация альтернативных запросов
# ============================================================
def multi_query_search(query: str, k: int = 4):
    """
    Генерируем разные формулировки вопроса → ищем по каждой.
    Улучшает recall (находим больше релевантных документов).
    """
    # Генерируем альтернативные запросы
    prompt = ChatPromptTemplate.from_template(
        "Сгенерируй 3 альтернативные формулировки вопроса. Только запросы, по одному на строку.\n\nВопрос: {q}"
    )
    chain = prompt | llm | StrOutputParser()
    alt_queries = chain.invoke({"q": query}).strip().split('\n')
    
    all_queries = [query] + [q.strip() for q in alt_queries if q.strip()]
    
    # Собираем уникальные документы
    seen = set()
    docs = []
    for q in all_queries:
        for doc in vectorstore.similarity_search(q, k=k):
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                docs.append(doc)
    
    return docs[:k*2], all_queries


# ============================================================
# 3. RERANKING: Переоценка релевантности с LLM
# ============================================================
def rerank_search(query: str, k: int = 4):
    """
    Сначала находим много документов, потом LLM оценивает релевантность.
    Улучшает precision (более точные результаты).
    """
    # Получаем больше документов
    docs = vectorstore.similarity_search(query, k=k*3)
    
    # LLM оценивает каждый документ
    rerank_prompt = ChatPromptTemplate.from_template(
        "Оцени релевантность документа к вопросу от 0 до 10. Верни ТОЛЬКО число.\n\nВопрос: {q}\nДокумент: {doc}\n\nОценка:"
    )
    chain = rerank_prompt | llm | StrOutputParser()
    
    scored = []
    for doc in docs:
        try:
            score = float(chain.invoke({"q": query, "doc": doc.page_content[:500]}))
        except:
            score = 5.0
        scored.append((doc, score))
    
    # Сортируем по оценке
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in scored[:k]], scored[:k]


# ============================================================
# 4. MMR: Maximum Marginal Relevance (разнообразие)
# ============================================================
def mmr_search(query: str, k: int = 4):
    """
    MMR балансирует релевантность и разнообразие.
    Избегает дублирующей информации.
    """
    docs = vectorstore.max_marginal_relevance_search(
        query, 
        k=k,
        fetch_k=20,      # Сначала берем 20
        lambda_mult=0.5  # 0=разнообразие, 1=релевантность
    )
    return docs


# ============================================================
# СРАВНЕНИЕ МЕТОДОВ
# ============================================================
print("="*60)
print("📊 СРАВНЕНИЕ МЕТОДОВ ПОИСКА")
print("="*60)

query = "Как Гарри победил Волдеморта?"
print(f"\n❓ Запрос: {query}\n")

# 1. Базовый поиск
print("-"*60)
print("1️⃣ БАЗОВЫЙ ПОИСК")
basic_docs = basic_search(query)
print(f"   Найдено: {len(basic_docs)} документов")
for doc in basic_docs[:2]:
    print(f"   • [{doc.metadata['title']}] {doc.page_content[:100]}...")

# 2. Multi-Query
print("\n" + "-"*60)
print("2️⃣ MULTI-QUERY (альтернативные формулировки)")
mq_docs, queries = multi_query_search(query)
print(f"   Запросы: {queries}")
print(f"   Найдено: {len(mq_docs)} уникальных документов")

# 3. Reranking
print("\n" + "-"*60)
print("3️⃣ RERANKING (переоценка LLM)")
rr_docs, scores = rerank_search(query)
print(f"   Топ документы по оценке LLM:")
for doc, score in scores[:3]:
    print(f"   • Score {score:.0f}: [{doc.metadata['title']}]")

# 4. MMR
print("\n" + "-"*60)
print("4️⃣ MMR (разнообразие)")
mmr_docs = mmr_search(query)
print(f"   Найдено: {len(mmr_docs)} разнообразных документов")
titles = [doc.metadata['title'] for doc in mmr_docs]
print(f"   Источники: {set(titles)}")


# ============================================================
# РЕКОМЕНДАЦИИ
# ============================================================
print("\n" + "="*60)
print("💡 КОГДА ИСПОЛЬЗОВАТЬ")
print("="*60)
print("""
   БАЗОВЫЙ      → Простые случаи, быстрый поиск
   MULTI-QUERY  → Сложные вопросы, нужен широкий охват
   RERANKING    → Важна точность, есть время на LLM вызовы
   MMR          → Нужны разнообразные источники
   
   🔥 ЛУЧШАЯ КОМБИНАЦИЯ:
   Multi-Query → Reranking → MMR
""")

"""
Лекция 11: Caching (Кеширование)
=================================
Кеширование для ускорения RAG и экономии средств.
Одинаковые/похожие запросы не требуют повторных вызовов LLM.
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from dotenv import load_dotenv
import os
import time
import hashlib

load_dotenv()

# ============================================================
# Загрузка данных
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

if os.getenv("OPENAI_API_KEY"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print("   ✅ Индексы созданы\n")
else:
    vectorstore = None
    llm = None
    print("   ⚠️ OPENAI_API_KEY не найден\n")


# ============================================================
# Метод 1: In-Memory Cache (LangChain)
# ============================================================
print("="*60)
print("1️⃣ In-Memory Cache (LangChain)")
print("="*60)

print("""
Простое кеширование одинаковых запросов в памяти.

Преимущества:
  ✅ Простота использования
  ✅ Нет зависимостей
  ✅ Мгновенный ответ для кешированных запросов

Недостатки:
  ❌ Только точные совпадения
  ❌ Кеш теряется при перезапуске
  ❌ Не масштабируется (один процесс)
""")

# Включаем кеш
set_llm_cache(InMemoryCache())

if llm:
    query = "Кто такой Гарри Поттер?"
    
    print(f"   Запрос: «{query}»\n")
    
    # Первый вызов (miss)
    print("   Первый вызов (cache miss):")
    start = time.time()
    response1 = llm.invoke(query)
    time1 = time.time() - start
    print(f"   ⏱️ Время: {time1:.3f}s")
    print(f"   Ответ: {response1.content[:100]}...\n")
    
    # Второй вызов (hit)
    print("   Второй вызов (cache hit):")
    start = time.time()
    response2 = llm.invoke(query)
    time2 = time.time() - start
    print(f"   ⏱️ Время: {time2:.3f}s")
    print(f"   Ответ: {response2.content[:100]}...")
    
    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"\n   🚀 Ускорение: {speedup:.1f}x")
    print(f"   💰 Экономия: {((time1 - time2) / time1 * 100):.1f}% времени")


# ============================================================
# Метод 2: Semantic Cache (LangChain)
# ============================================================
print("\n" + "="*60)
print("2️⃣ Semantic Cache (LangChain)")
print("="*60)

print("""
Кеширует не только точные совпадения, но и семантически похожие запросы.

Установка:
  pip install redis
  # Или используйте InMemorySemanticCache для тестирования

Преимущества:
  ✅ Кеширует похожие запросы (синонимы, парафразы)
  ✅ Экономит больше вызовов LLM
  ✅ Настраиваемый порог схожести

Недостатки:
  ❌ Требует Redis (или память для InMemory)
  ❌ Нужны эмбеддинги (дополнительная стоимость)
  ❌ Может кешировать слишком разные запросы
""")

try:
    from langchain_community.cache import RedisSemanticCache
    
    if os.getenv("REDIS_URL"):
        # Redis Semantic Cache
        semantic_cache = RedisSemanticCache(
            redis_url=os.getenv("REDIS_URL"),
            embedding=embeddings if embeddings else None,
            score_threshold=0.95  # Порог схожести (0-1)
        )
        set_llm_cache(semantic_cache)
        print("   ✅ Redis Semantic Cache подключен")
        print(f"   📊 Порог схожести: 0.95")
    else:
        print("   ⚠️ REDIS_URL не найден в .env")
        print("   Для демонстрации добавьте: REDIS_URL=redis://localhost:6379")
        
        # Альтернатива: InMemorySemanticCache (для тестирования)
        try:
            from langchain.cache import InMemorySemanticCache
            
            if embeddings:
                semantic_cache = InMemorySemanticCache(
                    embedding=embeddings,
                    score_threshold=0.95
                )
                set_llm_cache(semantic_cache)
                print("   ✅ InMemorySemanticCache используется (для тестирования)")
        except ImportError:
            print("   ⚠️ InMemorySemanticCache недоступен")
            
except ImportError:
    print("   ⚠️ RedisSemanticCache не установлен")
    print("   Установите: pip install redis langchain-community")

# Демонстрация semantic cache
if llm and embeddings:
    queries = [
        "Кто такой Гарри Поттер?",
        "Расскажи про Гарри Поттера",  # Парафраз
        "Что известно о Гарри Поттере?",  # Ещё один парафраз
    ]
    
    print("\n   Демонстрация semantic cache:")
    print("   (Похожие запросы должны использовать кеш)\n")
    
    for i, query in enumerate(queries, 1):
        print(f"   Запрос {i}: «{query}»")
        start = time.time()
        response = llm.invoke(query)
        elapsed = time.time() - start
        print(f"   ⏱️ Время: {elapsed:.3f}s")
        print(f"   Ответ: {response.content[:80]}...\n")


# ============================================================
# Метод 3: RAG Cache (кеширование полного пайплайна)
# ============================================================
print("\n" + "="*60)
print("3️⃣ RAG Cache (кеширование полного пайплайна)")
print("="*60)

print("""
Кешируем не только LLM ответы, но и результаты retrieval.

Стратегия:
  1. Кешируем embeddings запросов
  2. Кешируем результаты retrieval
  3. Кешируем финальные ответы LLM
""")

class SimpleRAGCache:
    """Простой RAG cache для демонстрации."""
    
    def __init__(self):
        self.query_cache = {}  # query_hash -> response
        self.retrieval_cache = {}  # query_hash -> documents
    
    def get_query_hash(self, query: str) -> str:
        """Хеш запроса для кеширования."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get_retrieval(self, query: str):
        """Получить кешированные документы."""
        key = self.get_query_hash(query)
        return self.retrieval_cache.get(key)
    
    def set_retrieval(self, query: str, documents: list):
        """Сохранить документы в кеш."""
        key = self.get_query_hash(query)
        self.retrieval_cache[key] = documents
    
    def get_response(self, query: str):
        """Получить кешированный ответ."""
        key = self.get_query_hash(query)
        return self.query_cache.get(key)
    
    def set_response(self, query: str, response: str):
        """Сохранить ответ в кеш."""
        key = self.get_query_hash(query)
        self.query_cache[key] = response

# Демонстрация
rag_cache = SimpleRAGCache()

def cached_rag(query: str, use_cache: bool = True):
    """RAG с кешированием."""
    # Проверяем кеш ответа
    if use_cache:
        cached_response = rag_cache.get_response(query)
        if cached_response:
            return cached_response, True  # cache hit
    
    # Проверяем кеш retrieval
    if use_cache:
        cached_docs = rag_cache.get_retrieval(query)
        if cached_docs:
            docs = cached_docs
            cache_retrieval = True
        else:
            docs = vectorstore.similarity_search(query, k=3) if vectorstore else []
            rag_cache.set_retrieval(query, docs)
            cache_retrieval = False
    else:
        docs = vectorstore.similarity_search(query, k=3) if vectorstore else []
        cache_retrieval = False
    
    # Генерируем ответ
    if llm and docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Ответь на вопрос, используя контекст:\n\n{context}\n\nВопрос: {query}"
        response = llm.invoke(prompt)
        answer = response.content
        
        # Сохраняем в кеш
        if use_cache:
            rag_cache.set_response(query, answer)
        
        return answer, False  # cache miss
    else:
        return "Не могу ответить", False

if vectorstore and llm:
    query = "Кто такой Гарри Поттер?"
    
    print("   Демонстрация RAG cache:\n")
    
    # Без кеша
    print("   БЕЗ кеша:")
    start = time.time()
    answer1, _ = cached_rag(query, use_cache=False)
    time1 = time.time() - start
    print(f"   ⏱️ Время: {time1:.3f}s")
    print(f"   Ответ: {answer1[:100]}...\n")
    
    # С кешем (первый раз)
    print("   С кешем (первый вызов, cache miss):")
    start = time.time()
    answer2, hit = cached_rag(query, use_cache=True)
    time2 = time.time() - start
    print(f"   ⏱️ Время: {time2:.3f}s ({'hit' if hit else 'miss'})")
    print(f"   Ответ: {answer2[:100]}...\n")
    
    # С кешем (второй раз)
    print("   С кешем (второй вызов, cache hit):")
    start = time.time()
    answer3, hit = cached_rag(query, use_cache=True)
    time3 = time.time() - start
    print(f"   ⏱️ Время: {time3:.3f}s ({'hit' if hit else 'miss'})")
    print(f"   Ответ: {answer3[:100]}...")
    
    speedup = time1 / time3 if time3 > 0 else float('inf')
    print(f"\n   🚀 Ускорение: {speedup:.1f}x")
    print(f"   💰 Экономия: {((time1 - time3) / time1 * 100):.1f}% времени")


# ============================================================
# Benchmark: С кешем vs Без кеша
# ============================================================
print("\n" + "="*60)
print("📊 BENCHMARK: С кешем vs Без кеша")
print("="*60)

if llm:
    test_queries = [
        "Кто такой Гарри Поттер?",
        "Что такое Хогвартс?",
        "Кто такие Дурсли?",
    ]
    
    # Без кеша
    print("\n   БЕЗ кеша:")
    times_no_cache = []
    for query in test_queries:
        start = time.time()
        llm.invoke(query)
        elapsed = time.time() - start
        times_no_cache.append(elapsed)
        print(f"   • {query[:30]:30} {elapsed:.3f}s")
    
    avg_no_cache = sum(times_no_cache) / len(times_no_cache)
    print(f"   Среднее: {avg_no_cache:.3f}s\n")
    
    # С кешем (первый проход - miss)
    print("   С кешем (первый проход - все miss):")
    times_cache_miss = []
    for query in test_queries:
        start = time.time()
        llm.invoke(query)
        elapsed = time.time() - start
        times_cache_miss.append(elapsed)
        print(f"   • {query[:30]:30} {elapsed:.3f}s (miss)")
    
    # С кешем (второй проход - hit)
    print("\n   С кешем (второй проход - все hit):")
    times_cache_hit = []
    for query in test_queries:
        start = time.time()
        llm.invoke(query)
        elapsed = time.time() - start
        times_cache_hit.append(elapsed)
        print(f"   • {query[:30]:30} {elapsed:.3f}s (hit)")
    
    avg_cache_hit = sum(times_cache_hit) / len(times_cache_hit)
    print(f"   Среднее: {avg_cache_hit:.3f}s")
    
    speedup = avg_no_cache / avg_cache_hit if avg_cache_hit > 0 else float('inf')
    print(f"\n   🚀 Ускорение: {speedup:.1f}x")
    print(f"   💰 Экономия времени: {((avg_no_cache - avg_cache_hit) / avg_no_cache * 100):.1f}%")
    print(f"   💰 Экономия API вызовов: {((len(test_queries) - 0) / len(test_queries) * 100):.1f}% на повторных запросах")


# ============================================================
# Рекомендации
# ============================================================
print("\n" + "="*60)
print("💡 РЕКОМЕНДАЦИИ ПО КЕШИРОВАНИЮ")
print("="*60)
print("""
1. Когда использовать кеширование:
   ✅ Много повторяющихся запросов
   ✅ Дорогие API вызовы (GPT-4, embeddings)
   ✅ Критична скорость ответа
   ✅ Есть память/Redis для хранения

2. Стратегии кеширования:
   • LLM Cache: для одинаковых промптов
   • Semantic Cache: для похожих запросов
   • RAG Cache: для полного пайплайна
   • TTL (Time To Live): автоматическая очистка старых записей

3. Настройка TTL:
   • Статические данные: TTL = ∞ (или очень долго)
   • Динамические данные: TTL = 1-24 часа
   • Часто обновляемые: TTL = 5-60 минут

4. Оптимизация:
   • Используйте Redis для продакшена
   • Настройте правильный score_threshold для semantic cache
   • Мониторьте hit rate (должен быть >50%)
   • Очищайте кеш при обновлении данных
""")

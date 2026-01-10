"""
Лекция 11: Полный оптимизированный RAG пайплайн
==============================================
Объединяем все оптимизации:
- Reranking для точности
- Caching для скорости
- Метрики для мониторинга
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import time

load_dotenv()

# ============================================================
# Подготовка данных
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

if not os.getenv("OPENAI_API_KEY"):
    print("   ⚠️ OPENAI_API_KEY не найден")
    exit(1)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(chunks, embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Включаем кеш
set_llm_cache(InMemoryCache())

print("   ✅ Индексы созданы, кеш включён\n")


# ============================================================
# Оптимизированный RAG класс
# ============================================================
print("="*60)
print("🔧 Оптимизированный RAG Pipeline")
print("="*60)

class OptimizedRAG:
    """RAG система с оптимизациями."""
    
    def __init__(
        self,
        vectorstore,
        llm,
        embeddings,
        reranker=None,
        use_cache=True,
        retrieval_k=10,
        rerank_k=5,
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self.reranker = reranker
        self.use_cache = use_cache
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_latency": 0.0,
        }
    
    def rerank_documents(self, query: str, documents: list) -> list:
        """Reranking документов если доступен reranker."""
        if not self.reranker or len(documents) == 0:
            return documents[:self.rerank_k]
        
        try:
            from sentence_transformers import CrossEncoder
            
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.reranker.predict(pairs)
            
            ranked = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [doc for doc, _ in ranked[:self.rerank_k]]
        except:
            return documents[:self.rerank_k]
    
    def invoke(self, query: str) -> dict:
        """Выполняет RAG запрос с оптимизациями."""
        start_time = time.time()
        self.metrics["total_queries"] += 1
        
        # 1. Retrieval (широкий поиск)
        docs = self.vectorstore.similarity_search(query, k=self.retrieval_k)
        
        # 2. Reranking (если доступен)
        if self.reranker:
            docs = self.rerank_documents(query, docs)
        
        # 3. Generation (с кешированием через LangChain)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Ответь на вопрос, используя только предоставленный контекст.

Контекст:
{context}

Вопрос: {query}

Ответ:"""
        
        # LLM вызов (кешируется автоматически если включён)
        response = self.llm.invoke(prompt)
        answer = response.content
        
        latency = time.time() - start_time
        
        # Обновляем метрики
        self.metrics["avg_latency"] = (
            (self.metrics["avg_latency"] * (self.metrics["total_queries"] - 1) + latency) /
            self.metrics["total_queries"]
        )
        
        return {
            "answer": answer,
            "documents": docs,
            "latency": latency,
            "num_docs": len(docs),
        }
    
    def get_metrics(self) -> dict:
        """Возвращает метрики системы."""
        return self.metrics.copy()


# ============================================================
# Создание оптимизированной системы
# ============================================================
print("\n   Создание системы с оптимизациями...")

# Опционально: Cross-Encoder reranker
reranker = None
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("   ✅ Reranker включён")
except ImportError:
    print("   ⚠️ Reranker недоступен (sentence-transformers не установлен)")

rag = OptimizedRAG(
    vectorstore=vectorstore,
    llm=llm,
    embeddings=embeddings,
    reranker=reranker,
    use_cache=True,
    retrieval_k=10,
    rerank_k=5,
)

print("   ✅ Система создана\n")


# ============================================================
# Демонстрация
# ============================================================
print("="*60)
print("🔍 ДЕМОНСТРАЦИЯ ОПТИМИЗИРОВАННОГО RAG")
print("="*60)

test_queries = [
    "Кто такой Гарри Поттер?",
    "Что такое Хогвартс?",
    "Кто такой Гарри Поттер?",  # Повторный запрос (для демонстрации кеша)
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'─'*60}")
    print(f"Запрос {i}: «{query}»")
    print(f"{'─'*60}")
    
    result = rag.invoke(query)
    
    print(f"\n   Ответ: {result['answer'][:150]}...")
    print(f"   ⏱️ Latency: {result['latency']:.3f}s")
    print(f"   📄 Документов использовано: {result['num_docs']}")


# ============================================================
# Сравнение: С оптимизациями vs Без
# ============================================================
print("\n" + "="*60)
print("📊 СРАВНЕНИЕ: С оптимизациями vs Без")
print("="*60)

query = "Кто такой Гарри Поттер?"

# Без оптимизаций
print("\n   1️⃣ БЕЗ оптимизаций:")
set_llm_cache(None)  # Отключаем кеш

start = time.time()
docs = vectorstore.similarity_search(query, k=5)
context = "\n\n".join([doc.page_content for doc in docs])
prompt = f"Ответь на вопрос: {query}\n\nКонтекст:\n{context}\n\nОтвет:"
response = llm.invoke(prompt)
time_no_opt = time.time() - start

print(f"   ⏱️ Время: {time_no_opt:.3f}s")
print(f"   📄 Документов: 5")

# С оптимизациями
print("\n   2️⃣ С оптимизациями:")
set_llm_cache(InMemoryCache())  # Включаем кеш

start = time.time()
result = rag.invoke(query)
time_with_opt = time.time() - start

print(f"   ⏱️ Время: {time_with_opt:.3f}s")
print(f"   📄 Документов: {result['num_docs']} (после reranking)")

# Повторный запрос (демонстрация кеша)
print("\n   3️⃣ Повторный запрос (cache hit):")
start = time.time()
result2 = rag.invoke(query)
time_cached = time.time() - start

print(f"   ⏱️ Время: {time_cached:.3f}s")
print(f"   🚀 Ускорение: {time_no_opt / time_cached:.1f}x")

print(f"\n   💡 Итог:")
print(f"   • Без оптимизаций: {time_no_opt:.3f}s")
print(f"   • С оптимизациями (первый раз): {time_with_opt:.3f}s")
print(f"   • С оптимизациями (кеш): {time_cached:.3f}s")
print(f"   • Экономия времени: {((time_no_opt - time_cached) / time_no_opt * 100):.1f}%")


# ============================================================
# Метрики системы
# ============================================================
print("\n" + "="*60)
print("📈 МЕТРИКИ СИСТЕМЫ")
print("="*60)

metrics = rag.get_metrics()
print(f"\n   Статистика:")
print(f"   • Всего запросов: {metrics['total_queries']}")
print(f"   • Средняя latency: {metrics['avg_latency']:.3f}s")
print(f"   • Cache hit rate: {(metrics['cache_hits'] / metrics['total_queries'] * 100) if metrics['total_queries'] > 0 else 0:.1f}%")


# ============================================================
# Рекомендации
# ============================================================
print("\n" + "="*60)
print("💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ")
print("="*60)
print("""
1. Оптимизации для продакшена:
   ✅ Reranking (Cross-Encoder или Cohere)
   ✅ Semantic Cache (Redis или InMemory)
   ✅ Мониторинг метрик
   ✅ A/B тестирование конфигураций

2. Приоритет оптимизаций:
   1. Caching (максимальный эффект на скорость)
   2. Reranking (максимальный эффект на качество)
   3. Метрики (для понимания системы)
   4. A/B тестирование (для выбора конфигурации)

3. Мониторинг:
   • Latency (время ответа)
   • Cache hit rate (эффективность кеша)
   • Quality metrics (точность, релевантность)
   • Cost (стоимость API вызовов)

4. Масштабирование:
   • Используйте Redis для кеша (распределённый)
   • Рассмотрите async обработку
   • Балансировка нагрузки
   • Горизонтальное масштабирование
""")

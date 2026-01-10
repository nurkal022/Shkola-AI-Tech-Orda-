"""
Лекция 11: Semantic Cache (Детальный пример)
============================================
Кеширование семантически похожих запросов.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from dotenv import load_dotenv
import os
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ============================================================
# Простая реализация Semantic Cache
# ============================================================
print("="*60)
print("🔧 Простая реализация Semantic Cache")
print("="*60)

class SimpleSemanticCache:
    """Простой semantic cache для демонстрации."""
    
    def __init__(self, embeddings, threshold=0.95):
        """
        Args:
            embeddings: Объект для создания эмбеддингов
            threshold: Порог схожести (0-1)
        """
        self.embeddings = embeddings
        self.threshold = threshold
        self.cache = {}  # query_hash -> response
        self.query_vectors = []  # Список векторов запросов
        self.query_keys = []  # Соответствующие ключи
    
    def _get_embedding(self, text: str):
        """Получить эмбеддинг текста."""
        return np.array(self.embeddings.embed_query(text))
    
    def _find_similar(self, query_vec):
        """Найти похожий запрос в кеше."""
        if not self.query_vectors:
            return None
        
        # Вычисляем схожесть со всеми кешированными запросами
        similarities = cosine_similarity(
            query_vec.reshape(1, -1),
            np.array(self.query_vectors)
        )[0]
        
        # Находим максимальную схожесть
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        if max_similarity >= self.threshold:
            return self.query_keys[max_idx], max_similarity
        
        return None
    
    def get(self, query: str):
        """Получить ответ из кеша если есть похожий запрос."""
        query_vec = self._get_embedding(query)
        result = self._find_similar(query_vec)
        
        if result:
            key, similarity = result
            return self.cache[key], similarity
        
        return None, None
    
    def set(self, query: str, response: str):
        """Сохранить запрос и ответ в кеш."""
        query_vec = self._get_embedding(query)
        key = hash(query)
        
        self.cache[key] = response
        self.query_vectors.append(query_vec)
        self.query_keys.append(key)
    
    def stats(self):
        """Статистика кеша."""
        return {
            "cached_queries": len(self.cache),
            "threshold": self.threshold,
        }


# ============================================================
# Демонстрация
# ============================================================
print("\n" + "="*60)
print("🔍 ДЕМОНСТРАЦИЯ Semantic Cache")
print("="*60)

if not os.getenv("OPENAI_API_KEY"):
    print("   ⚠️ OPENAI_API_KEY не найден")
    exit(1)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Создаём semantic cache
semantic_cache = SimpleSemanticCache(embeddings, threshold=0.90)

# Тестовые запросы (похожие по смыслу)
test_queries = [
    "Кто такой Гарри Поттер?",
    "Расскажи про Гарри Поттера",  # Парафраз
    "Что известно о Гарри Поттере?",  # Ещё один парафраз
    "Кто такой Дамблдор?",  # Другой вопрос
    "Расскажи про Дамблдора",  # Парафраз предыдущего
]

print("\n   Тестируем semantic cache на похожих запросах:\n")

for i, query in enumerate(test_queries, 1):
    print(f"   [{i}] Запрос: «{query}»")
    
    # Проверяем кеш
    cached_response, similarity = semantic_cache.get(query)
    
    if cached_response:
        print(f"       ✅ Cache HIT (similarity: {similarity:.3f})")
        print(f"       Ответ: {cached_response[:80]}...")
    else:
        print(f"       ❌ Cache MISS")
        # Генерируем ответ
        start = time.time()
        response = llm.invoke(query)
        elapsed = time.time() - start
        answer = response.content
        
        # Сохраняем в кеш
        semantic_cache.set(query, answer)
        
        print(f"       ⏱️ Время генерации: {elapsed:.3f}s")
        print(f"       Ответ: {answer[:80]}...")
    
    print()


# ============================================================
# Анализ эффективности
# ============================================================
print("="*60)
print("📊 АНАЛИЗ ЭФФЕКТИВНОСТИ")
print("="*60)

stats = semantic_cache.stats()
print(f"\n   Статистика кеша:")
print(f"   • Кешированных запросов: {stats['cached_queries']}")
print(f"   • Порог схожести: {stats['threshold']}")

# Тест на разных порогах
print("\n   Влияние порога схожести:")
thresholds = [0.85, 0.90, 0.95, 0.98]

for threshold in thresholds:
    cache = SimpleSemanticCache(embeddings, threshold=threshold)
    
    # Симулируем запросы
    hits = 0
    for query in test_queries:
        cached, sim = cache.get(query)
        if cached:
            hits += 1
        else:
            cache.set(query, f"Ответ на: {query}")
    
    hit_rate = hits / len(test_queries) * 100
    print(f"   • Threshold {threshold}: Hit rate = {hit_rate:.1f}%")


# ============================================================
# Сравнение: InMemory vs Semantic Cache
# ============================================================
print("\n" + "="*60)
print("📊 СРАВНЕНИЕ: InMemory vs Semantic Cache")
print("="*60)

# InMemory Cache (точные совпадения)
inmemory_cache = InMemoryCache()
set_llm_cache(inmemory_cache)

similar_queries = [
    "Кто такой Гарри Поттер?",
    "Расскажи про Гарри Поттера",  # Парафраз
]

print("\n   InMemory Cache (точные совпадения):")
for query in similar_queries:
    start = time.time()
    response = llm.invoke(query)
    elapsed = time.time() - start
    print(f"   • «{query}»: {elapsed:.3f}s")

print("\n   Semantic Cache (похожие запросы):")
semantic_cache2 = SimpleSemanticCache(embeddings, threshold=0.90)
for query in similar_queries:
    cached, sim = semantic_cache2.get(query)
    if cached:
        print(f"   • «{query}»: Cache HIT (similarity: {sim:.3f})")
    else:
        start = time.time()
        response = llm.invoke(query)
        elapsed = time.time() - start
        answer = response.content
        semantic_cache2.set(query, answer)
        print(f"   • «{query}»: {elapsed:.3f}s (MISS, сохранено в кеш)")


# ============================================================
# Рекомендации
# ============================================================
print("\n" + "="*60)
print("💡 РЕКОМЕНДАЦИИ")
print("="*60)
print("""
1. Выбор порога схожести:
   • 0.95-0.98: Строгий (только очень похожие запросы)
   • 0.90-0.95: Баланс (рекомендуется)
   • 0.85-0.90: Мягкий (больше cache hits, но может быть неточным)

2. Когда использовать Semantic Cache:
   ✅ Пользователи задают вопросы по-разному
   ✅ Много синонимов и парафразов
   ✅ Важна экономия API вызовов
   ✅ Есть бюджет на embeddings

3. Оптимизация:
   • Используйте быстрые embedding модели для кеша
   • Рассмотрите локальные embeddings (экономия)
   • Настройте TTL для автоматической очистки
   • Мониторьте hit rate и качество ответов

4. Альтернативы:
   • RedisSemanticCache (LangChain) - для продакшена
   • GPTCache - специализированная библиотека
   • Custom решение с векторной БД (FAISS, Qdrant)
""")

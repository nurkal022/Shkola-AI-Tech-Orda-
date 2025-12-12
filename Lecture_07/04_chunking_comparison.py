"""
Лекция 7: Практическое сравнение размеров чанков
================================================
Визуализация trade-offs разных параметров чанкинга.
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import time

# Загружаем текст
text = Path("data/Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt").read_text(encoding='utf-8')
print(f"📖 Текст: {len(text):,} символов\n")


# ============================================================
# Эксперимент 1: Влияние размера чанка
# ============================================================
print("="*60)
print("📊 Эксперимент 1: Размер чанка")
print("="*60)

results = []
for chunk_size in [256, 512, 1024, 1536, 2048]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5,  # 20% overlap
    )
    chunks = splitter.split_text(text)
    
    avg_size = sum(len(c) for c in chunks) // len(chunks)
    results.append({
        'chunk_size': chunk_size,
        'num_chunks': len(chunks),
        'avg_size': avg_size,
        'total_chars': sum(len(c) for c in chunks),
    })

print("\n   chunk_size │ Чанков │ Avg размер │ Всего символов │ Overhead")
print("   " + "─"*65)
for r in results:
    overhead = (r['total_chars'] / len(text) - 1) * 100
    print(f"   {r['chunk_size']:>10} │ {r['num_chunks']:>6} │ {r['avg_size']:>10} │ {r['total_chars']:>14,} │ {overhead:>6.1f}%")


# ============================================================
# Эксперимент 2: Влияние overlap
# ============================================================
print("\n" + "="*60)
print("📊 Эксперимент 2: Влияние overlap")
print("="*60)

chunk_size = 1000
print(f"\n   Фиксированный chunk_size = {chunk_size}")
print("\n   overlap │ % overlap │ Чанков │ Overhead │ Граничный контекст")
print("   " + "─"*65)

for overlap in [0, 100, 200, 300, 400, 500]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_text(text)
    total_chars = sum(len(c) for c in chunks)
    overhead = (total_chars / len(text) - 1) * 100
    
    # Оценка качества границ (сколько предложений сохранено целыми)
    broken_sentences = sum(1 for c in chunks if not c.rstrip().endswith(('.', '!', '?', '"', '»')))
    intact_ratio = (len(chunks) - broken_sentences) / len(chunks) * 100
    
    print(f"   {overlap:>7} │ {overlap/chunk_size*100:>8.0f}% │ {len(chunks):>6} │ {overhead:>7.1f}% │ {intact_ratio:>5.1f}% целых")


# ============================================================
# Эксперимент 3: Качество retrieval при разных размерах
# ============================================================
print("\n" + "="*60)
print("📊 Эксперимент 3: Retrieval качество")
print("="*60)

# Тестовые вопросы
test_queries = [
    "Как выглядит шрам Гарри Поттера?",
    "Кто такой Дурсль?",
    "Что такое философский камень?",
]

print("\n   Создаём индексы для разных размеров чанков...")
embeddings = OpenAIEmbeddings()

retrieval_results = {}
for chunk_size in [512, 1024, 2048]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5,
    )
    chunks = splitter.split_text(text)
    
    # Создаём векторный индекс
    start = time.time()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    index_time = time.time() - start
    
    retrieval_results[chunk_size] = {
        'vectorstore': vectorstore,
        'chunks': chunks,
        'index_time': index_time,
    }
    print(f"   chunk_size={chunk_size}: {len(chunks)} чанков, индексация {index_time:.1f}с")

print("\n   Тестируем поиск...")
print("   " + "─"*70)

for query in test_queries:
    print(f"\n   Запрос: «{query}»")
    
    for chunk_size, data in retrieval_results.items():
        # Ищем топ-1 релевантный чанк
        docs = data['vectorstore'].similarity_search_with_score(query, k=1)
        doc, score = docs[0]
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   [{chunk_size:>4}] score={score:.3f} | {preview}...")


# ============================================================
# Эксперимент 4: Скорость индексации и поиска
# ============================================================
print("\n" + "="*60)
print("📊 Эксперимент 4: Производительность")
print("="*60)

# Используем уже созданные индексы
print("\n   chunk_size │ Чанков │ Индексация │ Поиск (avg)")
print("   " + "─"*55)

for chunk_size, data in retrieval_results.items():
    # Время поиска (среднее по нескольким запросам)
    search_times = []
    for query in test_queries * 3:  # 9 запросов
        start = time.time()
        data['vectorstore'].similarity_search(query, k=3)
        search_times.append(time.time() - start)
    
    avg_search = sum(search_times) / len(search_times) * 1000  # в мс
    
    print(f"   {chunk_size:>10} │ {len(data['chunks']):>6} │ {data['index_time']:>9.2f}с │ {avg_search:>8.1f}мс")


# ============================================================
# TRADE-OFFS ВИЗУАЛИЗАЦИЯ
# ============================================================
print("\n" + "="*60)
print("📈 TRADE-OFFS: Размер чанка")
print("="*60)
print("""
    Маленькие чанки (256-512)        │  Большие чанки (1500-2048)
    ─────────────────────────────────┼─────────────────────────────────
    ✅ Точный поиск                  │  ✅ Много контекста
    ✅ Меньше токенов на запрос      │  ✅ Меньше API вызовов (embeddings)
    ✅ Быстрый поиск                 │  ✅ Сохранность смысла
    ❌ Потеря контекста              │  ❌ Шум в результатах
    ❌ Больше чанков = дороже        │  ❌ Дольше поиск
    ❌ Разрыв логических блоков      │  ❌ Превышение контекста LLM
    
    РЕКОМЕНДАЦИЯ: 1000-1500 символов = оптимальный баланс
""")


# ============================================================
# TRADE-OFFS: Overlap
# ============================================================
print("\n" + "="*60)
print("📈 TRADE-OFFS: Overlap (перекрытие)")
print("="*60)
print("""
    Маленький overlap (0-10%)        │  Большой overlap (30-50%)
    ─────────────────────────────────┼─────────────────────────────────
    ✅ Меньше дубликатов             │  ✅ Контекст сохраняется
    ✅ Меньше чанков                 │  ✅ Границы не теряют смысл
    ✅ Дешевле хранить               │  ✅ Лучше для сложных запросов
    ❌ Потеря контекста на границах  │  ❌ Дублирование информации
    ❌ Разрыв предложений            │  ❌ Дороже (больше чанков)
    
    РЕКОМЕНДАЦИЯ: 15-20% = хороший баланс (150-200 для chunk_size=1000)
""")


# ============================================================
# ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ
# ============================================================
print("\n" + "="*60)
print("🎯 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ")
print("="*60)
print("""
   1. Для Q&A / FAQ систем:
      chunk_size=500-800, overlap=100
      → Точные ответы на конкретные вопросы
   
   2. Для документации / мануалов:
      chunk_size=1000-1500, overlap=200
      → Баланс точности и контекста
   
   3. Для книг / длинных текстов:
      chunk_size=1500-2000, overlap=300
      → Сохранение нарратива
   
   4. Для кода:
      Используйте code-aware splitter
      chunk_size=1000-1500, overlap=100
      → Не разрывать функции/классы
   
   5. Общее правило:
      Начните с chunk_size=1000, overlap=200
      Измерьте качество на тестовых запросах
      Итеративно подбирайте параметры
""")


"""
Лекция 10: BM25 Search (Keyword-based)
======================================
BM25 - классический поиск по ключевым словам.
Ищет точные совпадения, учитывает частоту терминов.
"""

from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# ============================================================
# Загрузка и подготовка данных
# ============================================================
print("="*60)
print("📖 Загрузка данных")
print("="*60)

text = Path("../Lecture_07/data/Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt").read_text(encoding='utf-8')

# Чанкинг
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = splitter.split_text(text)

print(f"   Загружено: {len(chunks)} чанков")
print(f"   Средний размер: {sum(len(c) for c in chunks) // len(chunks)} символов\n")


# ============================================================
# Подготовка для BM25
# ============================================================
print("="*60)
print("🔧 Подготовка BM25 индекса")
print("="*60)

def tokenize(text: str) -> list:
    """Простая токенизация для русского языка."""
    # Убираем пунктуацию, приводим к нижнему регистру
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Разбиваем на слова
    words = text.split()
    # Убираем очень короткие слова
    words = [w for w in words if len(w) > 2]
    return words

# Токенизируем все чанки
tokenized_chunks = [tokenize(chunk) for chunk in chunks]

# Создаём BM25 индекс
bm25 = BM25Okapi(tokenized_chunks)

print(f"   ✅ Индекс создан для {len(tokenized_chunks)} документов")
print(f"   📊 Среднее количество токенов на чанк: {sum(len(t) for t in tokenized_chunks) // len(tokenized_chunks)}\n")


# ============================================================
# Функция BM25 поиска
# ============================================================
def bm25_search(query: str, k: int = 5):
    """
    BM25 поиск по ключевым словам.
    
    Args:
        query: Поисковый запрос
        k: Количество результатов
    
    Returns:
        Список результатов с score и текстом
    """
    # Токенизируем запрос
    tokenized_query = tokenize(query)
    
    if not tokenized_query:
        return []
    
    # Получаем scores для всех документов
    scores = bm25.get_scores(tokenized_query)
    
    # Находим топ-K результатов
    top_indices = sorted(
        range(len(scores)), 
        key=lambda i: scores[i], 
        reverse=True
    )[:k]
    
    # Формируем результаты
    results = []
    for idx in top_indices:
        results.append({
            "rank": len(results) + 1,
            "chunk_id": idx,
            "score": float(scores[idx]),
            "text": chunks[idx][:200] + "..." if len(chunks[idx]) > 200 else chunks[idx],
            "full_text": chunks[idx],
        })
    
    return results


# ============================================================
# Демонстрация
# ============================================================
print("="*60)
print("🔍 ДЕМОНСТРАЦИЯ BM25 ПОИСКА")
print("="*60)

test_queries = [
    "волшебная палочка",
    "Гарри Поттер",
    "Хогвартс",
    "философский камень",
    "Дурсли",
]

for query in test_queries:
    print(f"\n{'─'*60}")
    print(f"❓ Запрос: «{query}»")
    print(f"{'─'*60}")
    
    results = bm25_search(query, k=3)
    
    if not results:
        print("   ⚠️ Результаты не найдены")
        continue
    
    for result in results:
        print(f"\n   [{result['rank']}] Score: {result['score']:.4f}")
        print(f"   📄 {result['text']}")


# ============================================================
# Как работает BM25
# ============================================================
print("\n" + "="*60)
print("📚 КАК РАБОТАЕТ BM25")
print("="*60)
print("""
BM25 (Best Matching 25) - улучшенная версия TF-IDF:

score(D, Q) = Σ IDF(qi) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |D|/avgdl))

Где:
  • TF (Term Frequency) - частота термина в документе
  • IDF (Inverse Document Frequency) - обратная частота в коллекции
    → Редкие слова важнее частых!
  • k1 = 1.2-2.0 (насыщение TF)
  • b = 0.75 (нормализация длины документа)

Плюсы BM25:
  ✅ Быстрый (не требует ML моделей)
  ✅ Точные совпадения
  ✅ Учитывает редкость слов
  ✅ Работает без обучения

Минусы BM25:
  ❌ Не понимает синонимы ("волшебный" ≠ "магический")
  ❌ Не понимает контекст
  ❌ Требует точного совпадения токенов
  ❌ Не работает с парафразами
""")


# ============================================================
# Сравнение с разными запросами
# ============================================================
print("\n" + "="*60)
print("📊 СРАВНЕНИЕ: Точные vs Неточные запросы")
print("="*60)

print("\n1️⃣ Точный запрос (есть в тексте):")
query1 = "волшебная палочка"
results1 = bm25_search(query1, k=1)
if results1:
    print(f"   Запрос: «{query1}»")
    print(f"   Top score: {results1[0]['score']:.4f}")
    print(f"   Текст: {results1[0]['text'][:100]}...")

print("\n2️⃣ Синоним (нет точного совпадения):")
query2 = "магический жезл"  # Синоним, но не в тексте
results2 = bm25_search(query2, k=1)
if results2:
    print(f"   Запрос: «{query2}»")
    print(f"   Top score: {results2[0]['score']:.4f}")
    print(f"   Текст: {results2[0]['text'][:100]}...")
    print("   ⚠️ BM25 не нашёл релевантный текст из-за синонимов!")

print("\n3️⃣ Парафраз:")
query3 = "мальчик который выжил"  # Парафраз имени Гарри
results3 = bm25_search(query3, k=1)
if results3:
    print(f"   Запрос: «{query3}»")
    print(f"   Top score: {results3[0]['score']:.4f}")
    print(f"   Текст: {results3[0]['text'][:100]}...")

print("\n💡 Вывод: BM25 отлично работает с точными совпадениями,")
print("   но не справляется с синонимами и парафразами.")
print("   Для этого нужен Semantic Search!")


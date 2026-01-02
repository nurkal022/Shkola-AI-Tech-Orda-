"""
Лекция 10: Semantic Search (Vector-based)
=========================================
Семантический поиск использует эмбеддинги для понимания смысла.
Понимает синонимы, парафразы и контекст.
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

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
# Создание векторного индекса
# ============================================================
print("="*60)
print("🔧 Создание векторного индекса")
print("="*60)

# Проверяем наличие API ключа
if not os.getenv("OPENAI_API_KEY"):
    print("   ⚠️ OPENAI_API_KEY не найден!")
    print("   Создайте .env файл с OPENAI_API_KEY=your_key")
    print("   Или установите переменную окружения")
    exit(1)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("   🔄 Создание эмбеддингов...")
vectorstore = FAISS.from_texts(chunks, embeddings)

print(f"   ✅ Векторный индекс создан")
print(f"   📊 Размерность эмбеддингов: 1536\n")


# ============================================================
# Функция Semantic Search
# ============================================================
def semantic_search(query: str, k: int = 5):
    """
    Семантический поиск с использованием векторных эмбеддингов.
    
    Args:
        query: Поисковый запрос
        k: Количество результатов
    
    Returns:
        Список результатов с score и текстом
    """
    # Поиск с scores (косинусное расстояние)
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    # Формируем результаты
    formatted_results = []
    for rank, (doc, score) in enumerate(results, 1):
        # score - это расстояние, конвертируем в similarity (1 - distance)
        similarity = 1 - score
        
        formatted_results.append({
            "rank": rank,
            "score": float(score),  # Расстояние (меньше = лучше)
            "similarity": float(similarity),  # Схожесть (больше = лучше)
            "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "full_text": doc.page_content,
        })
    
    return formatted_results


# ============================================================
# Демонстрация
# ============================================================
print("="*60)
print("🔍 ДЕМОНСТРАЦИЯ SEMANTIC SEARCH")
print("="*60)

test_queries = [
    "волшебная палочка",
    "магический жезл",  # Синоним!
    "школа для волшебников",
    "мальчик который выжил",  # Парафраз
    "семья которая не любит магию",
]

for query in test_queries:
    print(f"\n{'─'*60}")
    print(f"❓ Запрос: «{query}»")
    print(f"{'─'*60}")
    
    results = semantic_search(query, k=3)
    
    for result in results:
        print(f"\n   [{result['rank']}] Distance: {result['score']:.4f} | Similarity: {result['similarity']:.4f}")
        print(f"   📄 {result['text']}")


# ============================================================
# Как работает Semantic Search
# ============================================================
print("\n" + "="*60)
print("📚 КАК РАБОТАЕТ SEMANTIC SEARCH")
print("="*60)
print("""
1. Текст → Эмбеддинг (вектор 1536 размерности)
   - Каждое слово/фраза представлена как точка в многомерном пространстве
   - Семантически близкие слова находятся рядом

2. Запрос → Эмбеддинг
   - Тот же процесс для поискового запроса

3. Поиск по косинусному сходству:
   similarity = cos(θ) = (A · B) / (||A|| × ||B||)
   
   Где:
   • A = вектор документа
   • B = вектор запроса
   • θ = угол между векторами

Плюсы Semantic Search:
  ✅ Понимает синонимы ("волшебный" ≈ "магический")
  ✅ Понимает контекст
  ✅ Работает с парафразами
  ✅ Находит семантически близкие концепции

Минусы Semantic Search:
  ❌ Может "улетать" по смыслу (слишком общие результаты)
  ❌ Требует ML модель (дороже, медленнее)
  ❌ Может пропускать точные совпадения
  ❌ Зависит от качества эмбеддингов
""")


# ============================================================
# Сравнение: Точные vs Синонимы
# ============================================================
print("\n" + "="*60)
print("📊 СРАВНЕНИЕ: Точные vs Синонимы")
print("="*60)

print("\n1️⃣ Точный запрос:")
query1 = "волшебная палочка"
results1 = semantic_search(query1, k=1)
if results1:
    print(f"   Запрос: «{query1}»")
    print(f"   Similarity: {results1[0]['similarity']:.4f}")
    print(f"   Текст: {results1[0]['text'][:100]}...")

print("\n2️⃣ Синоним (Semantic понимает!):")
query2 = "магический жезл"
results2 = semantic_search(query2, k=1)
if results2:
    print(f"   Запрос: «{query2}»")
    print(f"   Similarity: {results2[0]['similarity']:.4f}")
    print(f"   Текст: {results2[0]['text'][:100]}...")
    print("   ✅ Semantic Search нашёл релевантный текст!")

print("\n3️⃣ Парафраз:")
query3 = "мальчик который выжил"
results3 = semantic_search(query3, k=1)
if results3:
    print(f"   Запрос: «{query3}»")
    print(f"   Similarity: {results3[0]['similarity']:.4f}")
    print(f"   Текст: {results3[0]['text'][:100]}...")
    print("   ✅ Semantic понимает парафразы!")

print("\n💡 Вывод: Semantic Search отлично работает с синонимами,")
print("   но может пропускать точные совпадения.")
print("   Идеальное решение: комбинировать BM25 + Semantic = Hybrid Search!")


"""
RAG Шаг 3: Embeddings и Vector Store
====================================
Создание векторов и хранилища для семантического поиска.
"""
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

# ============================================================
# 1. EMBEDDING МОДЕЛИ
# ============================================================
print("="*60)
print("1️⃣ EMBEDDING МОДЕЛИ OpenAI")
print("="*60)
print("""
   text-embedding-3-small  - быстрее, дешевле ($0.02/1M токенов)
   text-embedding-3-large  - точнее, дороже ($0.13/1M токенов)
   text-embedding-ada-002  - legacy модель
""")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("✅ Используем: text-embedding-3-small")

# Пример embedding
test_text = "Гарри Поттер - мальчик который выжил"
vector = embeddings.embed_query(test_text)
print(f"\n📊 Размерность вектора: {len(vector)}")
print(f"   Первые 5 значений: {vector[:5]}")


# ============================================================
# 2. ПОДГОТОВКА ДОКУМЕНТОВ (только 1 книга для демо)
# ============================================================
print("\n" + "="*60)
print("2️⃣ ПОДГОТОВКА ДОКУМЕНТОВ")
print("="*60)

documents = []
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Берем только первую книгу для быстрого демо
file = list(Path("data").glob("*.txt"))[0]
name = file.stem.split(']_')[-1].replace('_', ' ')
text = file.read_text(encoding='utf-8')
chunks = splitter.split_text(text)

# Берем первые 100 чанков для демо (избегаем лимита токенов)
for i, chunk in enumerate(chunks[:100]):
    documents.append(Document(
        page_content=chunk,
        metadata={"title": name, "chunk_id": i}
    ))

print(f"📖 {name}: {len(chunks)} чанков всего")
print(f"✅ Используем для демо: {len(documents)} чанков")


# ============================================================
# 3. СОЗДАНИЕ FAISS ИНДЕКСА
# ============================================================
print("\n" + "="*60)
print("3️⃣ FAISS - Создание индекса")
print("="*60)

print("🔄 Создание FAISS индекса...")
faiss_store = FAISS.from_documents(documents, embeddings)
print("✅ FAISS индекс создан!")

# Сохранение
faiss_store.save_local("./faiss_demo")
print("💾 Сохранено: ./faiss_demo")


# ============================================================
# 4. ПОИСК БЕЗ ГЕНЕРАЦИИ - просто извлечение документов
# ============================================================
print("\n" + "="*60)
print("4️⃣ ПОИСК ДОКУМЕНТОВ (без LLM)")
print("="*60)

queries = [
    "Кто такой Волдеморт?",
    "Хогвартс",
    "Дамблдор",
]

for query in queries:
    print(f"\n❓ Запрос: {query}")
    print("-"*50)
    
    # Простой поиск - возвращает документы
    docs = faiss_store.similarity_search(query, k=3)
    
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 Документ {i}:")
        print(f"   Источник: {doc.metadata['title']}, chunk #{doc.metadata['chunk_id']}")
        print(f"   Текст: {doc.page_content[:200]}...")


# ============================================================
# 5. СРАВНЕНИЕ МЕТОДОВ ПОИСКА
# ============================================================
print("\n" + "="*60)
print("5️⃣ СРАВНЕНИЕ МЕТОДОВ ПОИСКА")
print("="*60)

query = "Гарри Поттер волшебник"

# Метод 1: similarity_search - базовый поиск
print("\n🔍 1. similarity_search (базовый)")
docs1 = faiss_store.similarity_search(query, k=3)
for doc in docs1:
    print(f"   • chunk #{doc.metadata['chunk_id']}: {doc.page_content[:80]}...")

# Метод 2: similarity_search_with_score - с оценкой расстояния
print("\n🔍 2. similarity_search_with_score (с оценкой)")
docs2 = faiss_store.similarity_search_with_score(query, k=3)
for doc, score in docs2:
    print(f"   • score={score:.3f}, chunk #{doc.metadata['chunk_id']}: {doc.page_content[:60]}...")

# Метод 3: max_marginal_relevance_search (MMR) - разнообразие
print("\n🔍 3. max_marginal_relevance_search (MMR - разнообразие)")
docs3 = faiss_store.max_marginal_relevance_search(query, k=3, fetch_k=10)
for doc in docs3:
    print(f"   • chunk #{doc.metadata['chunk_id']}: {doc.page_content[:80]}...")

# Метод 4: similarity_search_with_relevance_scores - нормализованные оценки
print("\n🔍 4. similarity_search_with_relevance_scores (0-1 оценка)")
docs4 = faiss_store.similarity_search_with_relevance_scores(query, k=3)
for doc, score in docs4:
    print(f"   • relevance={score:.3f}, chunk #{doc.metadata['chunk_id']}: {doc.page_content[:60]}...")


# ============================================================
# 6. ПОИСК ПО ВЕКТОРУ НАПРЯМУЮ
# ============================================================
print("\n" + "="*60)
print("6️⃣ ПОИСК ПО ВЕКТОРУ НАПРЯМУЮ")
print("="*60)

# Создаем embedding запроса вручную
query_vector = embeddings.embed_query("магия и волшебство")
print(f"📊 Вектор запроса: {len(query_vector)} измерений")

# Поиск по вектору
docs5 = faiss_store.similarity_search_by_vector(query_vector, k=2)
for doc in docs5:
    print(f"   • {doc.page_content[:100]}...")


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "="*60)
print("📊 СРАВНЕНИЕ МЕТОДОВ")
print("="*60)
print("""
   similarity_search              → Базовый, быстрый
   similarity_search_with_score   → + расстояние (меньше = лучше)
   similarity_search_with_relevance_scores → + релевантность 0-1
   max_marginal_relevance_search  → + разнообразие результатов
   similarity_search_by_vector    → Поиск по готовому вектору
""")

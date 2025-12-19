"""
Пример 4: ChromaDB — простая векторная БД на SQLite

ChromaDB:
- Хранит данные в SQLite (персистентность из коробки)
- Встроенная поддержка метаданных и фильтрации
- Простой API
- Можно использовать встроенные эмбеддинги или свои
"""

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

print("=== ChromaDB — векторная база данных ===\n")

# =============================================
# Шаг 1: Создаём клиент с персистентностью
# =============================================
print("1. Создаём клиент ChromaDB...")

# Данные сохраняются в папку ./chroma_db
client = chromadb.PersistentClient(path="./chroma_db")
print("   Данные будут храниться в ./chroma_db")

# =============================================
# Шаг 2: Настраиваем эмбеддинги (OpenAI)
# =============================================
print("\n2. Настраиваем OpenAI эмбеддинги...")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small"
)

# =============================================
# Шаг 3: Создаём коллекцию
# =============================================
print("\n3. Создаём коллекцию...")

# Удаляем если существует (для чистого запуска)
try:
    client.delete_collection("my_documents")
except:
    pass

collection = client.create_collection(
    name="my_documents",
    embedding_function=openai_ef,
    metadata={"description": "Демо коллекция"}
)
print(f"   Коллекция '{collection.name}' создана")

# =============================================
# Шаг 4: Загружаем и добавляем документы из книг
# =============================================
print("\n4. Загружаем документы из книг...")

from utils import get_all_books_chunks

chunks = get_all_books_chunks("data", chunk_size=500, max_chunks=200)
print(f"   Загружено {len(chunks)} чанков")

# Подготавливаем данные
documents = [chunk["text"] for chunk in chunks]
metadatas = [
    {
        "book": chunk["book"],
        "chunk_id": chunk["chunk_id"]
    }
    for chunk in chunks
]
ids = [f"{chunk['book']}_{chunk['chunk_id']}" for chunk in chunks]

# Добавляем батчами (ChromaDB может обработать много, но лучше батчами)
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_meta = metadatas[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    
    collection.add(
        documents=batch_docs,
        metadatas=batch_meta,
        ids=batch_ids
    )
    print(f"   Добавлено: {min(i+batch_size, len(documents))}/{len(documents)}")

print(f"\n   Всего в коллекции: {collection.count()} документов")

# =============================================
# Шаг 5: Поиск
# =============================================
print("\n5. Поиск...")

def search(query: str, n_results: int = 3, where: dict = None):
    """Поиск с опциональной фильтрацией"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where  # Фильтр по метаданным
    )
    return results


# Простой поиск
print("\n🔍 Запрос: 'кто такой Гарри Поттер?'")
results = search("кто такой Гарри Поттер?")
for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
    preview = doc[:80] + "..." if len(doc) > 80 else doc
    print(f"   [{1-dist:.3f}] [{meta['book']}] {preview}")

# Поиск с фильтром по книге
print("\n🔍 Запрос: 'Хогвартс' (только первая книга)")
# Получаем первую книгу из коллекции
all_books = collection.get()
if all_books['metadatas']:
    first_book = all_books['metadatas'][0]['book']
    results = search("Хогвартс", where={"book": first_book})
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        preview = doc[:80] + "..." if len(doc) > 80 else doc
        print(f"   [{1-dist:.3f}] [{meta['book']}] {preview}")

# Ещё один запрос
print("\n🔍 Запрос: 'кто такой Хагрид?'")
results = search("кто такой Хагрид?")
for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
    preview = doc[:80] + "..." if len(doc) > 80 else doc
    print(f"   [{1-dist:.3f}] [{meta['book']}] {preview}")


# =============================================
# Дополнительные операции
# =============================================
print("\n\n=== Дополнительные операции ===")

# Получить документ по ID
print("\nПолучение по ID 'doc_0':")
result = collection.get(ids=["doc_0"])
print(f"   {result['documents'][0]}")

# Обновить документ
print("\nОбновление doc_0...")
collection.update(
    ids=["doc_0"],
    documents=["Python — самый популярный язык для ML"],
    metadatas=[{"category": "ml", "difficulty": "beginner"}]
)

# Проверяем обновление
result = collection.get(ids=["doc_0"])
print(f"   Новый текст: {result['documents'][0]}")
print(f"   Новые метаданные: {result['metadatas'][0]}")

# Удалить документ
print("\nУдаление doc_7...")
collection.delete(ids=["doc_7"])
print(f"   Осталось документов: {collection.count()}")


"""
Пример 2: Загрузка документов в Supabase

1. Читаем Конституцию Республики Казахстан
2. Разбиваем на чанки
3. Создаём эмбеддинги через OpenAI
4. Загружаем в Supabase с векторами
"""

from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
from utils import get_all_books_chunks
import os
import time

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI
openai_client = OpenAI()

print("=== Загрузка документов в Supabase ===\n")

# =============================================
# Шаг 1: Загружаем данные из конституции
# =============================================
print("1. Загружаем данные из конституции...")

from utils import get_book_chunks

# Загружаем конституцию
# Используем рекурсивный чанкинг с параметрами для ~100 документов
constitution_path = "data/конституция.txt"
chunks = get_book_chunks(constitution_path, chunk_size=1500, chunk_overlap=200)
print(f"\nЗагружено {len(chunks)} чанков из конституции (рекурсивный чанкинг)\n")

# =============================================
# Шаг 2: Очищаем старые данные (опционально)
# =============================================
print("2. Очищаем старые данные...")
try:
    supabase.table("documents").delete().neq("id", 0).execute()  # Удаляем все
    print("   ✅ Старые данные удалены")
except Exception as e:
    print(f"   ⚠️  {e}")

# =============================================
# Шаг 3: Создаём эмбеддинги и загружаем
# =============================================
print("\n3. Создаём эмбеддинги и загружаем в Supabase...")
print("   (это может занять время...)")

def get_embeddings_batch(texts: list[str], batch_size: int = 100):
    """Получить эмбеддинги батчами"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"   Обработано эмбеддингов: {min(i+batch_size, len(texts))}/{len(texts)}")
    return all_embeddings

# Получаем эмбеддинги
texts = [chunk["text"] for chunk in chunks]
embeddings = get_embeddings_batch(texts, batch_size=100)

# Загружаем в Supabase батчами
batch_size = 20
total_loaded = 0

for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size]
    
    # Подготавливаем данные для вставки
    records = []
    for chunk, emb in zip(batch_chunks, batch_embeddings):
        records.append({
            "content": chunk["text"],
            "book": chunk["book"],
            "chunk_id": chunk["chunk_id"],
            "embedding": emb  # Supabase автоматически конвертирует в VECTOR
        })
    
    # Вставляем батч
    try:
        supabase.table("documents").insert(records).execute()
        total_loaded += len(records)
        print(f"   Загружено в Supabase: {total_loaded}/{len(chunks)}")
    except Exception as e:
        print(f"   ⚠️  Ошибка при загрузке батча: {e}")
        break
    
    # Небольшая задержка чтобы не перегрузить API
    time.sleep(0.5)

print(f"\n✅ Загружено {total_loaded} документов в Supabase!")

# Проверяем количество
try:
    count = supabase.table("documents").select("id", count="exact").execute()
    print(f"📊 Всего документов в БД: {count.count if hasattr(count, 'count') else 'N/A'}")
except:
    pass


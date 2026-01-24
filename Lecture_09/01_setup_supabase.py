"""
Пример 1: Настройка Supabase с pgvector

Шаги:
1. Создать проект в Supabase (https://supabase.com)
2. Включить расширение pgvector
3. Создать таблицу для документов
4. Создать индекс для быстрого поиска
"""

from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

# Получаем credentials из .env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Service role key для SQL

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️  Ошибка: SUPABASE_URL и SUPABASE_SERVICE_ROLE_KEY должны быть в .env")
    print("\nПример .env:")
    print("SUPABASE_URL=https://your-project.supabase.co")
    print("SUPABASE_SERVICE_ROLE_KEY=your-service-role-key")
    exit(1)

# Создаём клиент
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("=== Настройка Supabase с pgvector ===\n")

# =============================================
# Шаг 1: Включить расширение pgvector
# =============================================
print("1. Включаем расширение pgvector...")

# Выполняем SQL через Supabase REST API (нужен service role key)
try:
    # Используем rpc для выполнения SQL
    # В Supabase это делается через SQL Editor или через прямые SQL запросы
    print("   ⚠️  Включите pgvector вручную через SQL Editor в Supabase:")
    print("   SQL: CREATE EXTENSION IF NOT EXISTS vector;")
    print("   Или выполните через psql/pgAdmin")
except Exception as e:
    print(f"   Примечание: {e}")

# =============================================
# Шаг 2: Создать таблицу
# =============================================
print("\n2. Создаём таблицу documents...")

create_table_sql = """
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  book TEXT,
  chunk_id INTEGER,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT NOW()
);
"""

print("   ⚠️  Выполните SQL в Supabase SQL Editor:")
print(create_table_sql)

# =============================================
# Шаг 3: Создать индекс для векторного поиска
# =============================================
print("\n3. Создаём индекс для векторного поиска...")

create_index_sql = """
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
"""

print("   ⚠️  Выполните SQL в Supabase SQL Editor:")
print(create_index_sql)

# =============================================
# Проверка подключения
# =============================================
print("\n4. Проверяем подключение...")

try:
    # Пробуем получить список таблиц
    result = supabase.table("documents").select("id").limit(1).execute()
    print("   ✅ Подключение к Supabase успешно!")
    print(f"   Таблица 'documents' существует")
except Exception as e:
    print(f"   ⚠️  Таблица ещё не создана или ошибка подключения: {e}")
    print("   Создайте таблицу через SQL Editor в Supabase Dashboard")

print("\n" + "="*50)
print("📝 ИНСТРУКЦИЯ:")
print("="*50)
print("""
1. Откройте Supabase Dashboard → SQL Editor
2. Выполните следующие SQL команды:

-- Включить pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Создать таблицу
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  book TEXT,
  chunk_id INTEGER,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT NOW()
);

-- Создать индекс
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

3. После этого запустите 02_load_documents.py
""")


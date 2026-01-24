# 🚀 Быстрый старт

## 1. Получить Service Role Key (2 минуты)

1. https://supabase.com/dashboard → выберите проект `TechOrdaTest`
2. **Settings** (⚙️) → **API**
3. Найдите **`service_role`** → 👁️ показать → 📋 скопировать
4. Вставьте в `.env`: `SUPABASE_SERVICE_ROLE_KEY=ваш-ключ`

## 2. Выполнить SQL (5 минут)

**Supabase Dashboard → SQL Editor → New query**

### Шаг 1: Включить pgvector
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Шаг 2: Создать таблицу
```sql
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  book TEXT,
  chunk_id INTEGER,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT NOW()
);
```

### Шаг 3: Создать индекс
```sql
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Шаг 4: Создать функцию поиска
```sql
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding VECTOR(1536),
  match_count INT DEFAULT 5,
  filter_book TEXT DEFAULT NULL
)
RETURNS TABLE (
  id INT,
  content TEXT,
  book TEXT,
  chunk_id INT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    documents.id,
    documents.content,
    documents.book,
    documents.chunk_id,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE (filter_book IS NULL OR documents.book = filter_book)
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

## 3. Проверить подключение

```bash
python 00_test_connection.py
```

Должно показать: ✅ Подключение работает!

## 4. Загрузить данные

```bash
python 02_load_documents.py
```

## 5. Запустить RAG

```bash
python 04_rag_pipeline.py
# или
python 05_langchain_supabase.py
```

---

**Ваш проект:**
- URL: https://qdhswcgeqbkkhdstiiqh.supabase.co
- Dashboard: https://supabase.com/dashboard/project/qdhswcgeqbkkhdstiiqh

**Подробные инструкции:** см. `SETUP_INSTRUCTIONS.md`


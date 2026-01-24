# Лекция 9: RAG с pgvector и Supabase

Практическое руководство по созданию RAG-пайплайна с использованием pgvector в Supabase.

## Что такое Supabase?

Supabase = PostgreSQL + Auth + Storage + API
- Бесплатный tier для разработки
- Автоматический REST API
- Встроенная поддержка pgvector

## Установка

```bash
pip install -r requirements.txt
```

## Настройка

### 1. Создайте проект в Supabase

1. Зайдите на https://supabase.com
2. Создайте новый проект (бесплатно)
3. Дождитесь завершения настройки (~2 минуты)

### 2. Получите Service Role Key

**Ваш проект:**
- **Project URL**: https://qdhswcgeqbkkhdstiiqh.supabase.co ✅ (уже в .env)
- **Project ID**: qdhswcgeqbkkhdstiiqh

**Где найти Service Role Key:**
1. Supabase Dashboard → **Settings** (⚙️ в левом меню)
2. Выберите **API**
3. В секции **Project API keys** найдите **`service_role`** (секретный ключ)
4. Нажмите 👁️ чтобы показать, затем 📋 чтобы скопировать
5. Добавьте в `.env`: `SUPABASE_SERVICE_ROLE_KEY=ваш-ключ`

⚠️ **ВНИМАНИЕ**: Service Role Key имеет полный доступ к БД! Не публикуйте его!

### 3. Создайте файл `.env`

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
OPENAI_API_KEY=sk-proj-...
```

### 4. Включите pgvector

**Способ 1 (через SQL Editor):**
1. Supabase Dashboard → **SQL Editor** (иконка базы данных в левом меню)
2. Нажмите **New query**
3. Выполните:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Способ 2 (через UI):**
1. Supabase Dashboard → **Database** → **Extensions**
2. Найдите `vector` в списке
3. Нажмите **Enable**

### 5. Создайте таблицу

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

### 6. Создайте индекс

```sql
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### 7. Создайте функцию для поиска

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

## Примеры

### 0. Проверка подключения
```bash
python 00_test_connection.py
```
Быстрая проверка подключения к Supabase.

### 1. Настройка Supabase
```bash
python 01_setup_supabase.py
```
Проверяет подключение и показывает инструкции.

### 2. Загрузка документов
```bash
python 02_load_documents.py
```
Загружает книги о Гарри Поттере в Supabase с эмбеддингами.

### 3. Векторный поиск
```bash
python 03_vector_search.py
```
Демонстрирует SQL операторы для векторного поиска.

### 4. RAG пайплайн
```bash
python 04_rag_pipeline.py
```
Полный цикл: Retrieval → Augmentation → Generation.

### 5. LangChain интеграция
```bash
python 05_langchain_supabase.py
```
Упрощённая работа через LangChain.

## Преимущества Supabase + pgvector

✅ **Всё в одной БД** — SQL + векторы  
✅ **Фильтрация** — WHERE условия + векторный поиск  
✅ **JOIN'ы** — можно соединять с другими таблицами  
✅ **Транзакции** — ACID гарантии  
✅ **Бесплатный tier** — для разработки и обучения  

## Данные

В папке `data/` находятся книги о Гарри Поттере (7 книг).

Примеры автоматически загружают и обрабатывают эти данные.

## Примечания

- Первая загрузка может занять время (создание эмбеддингов)
- Используется ограниченное количество чанков (30-50) для демо
- В production увеличьте `lists` в индексе для больших датасетов


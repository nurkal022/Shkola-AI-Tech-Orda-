# 🎯 Следующие шаги - конкретные команды

## ✅ Шаг 1: Service Role Key добавлен в .env

## 📝 Шаг 2: Выполнить SQL в Supabase

### Откройте Supabase SQL Editor:

1. Перейдите: https://supabase.com/dashboard/project/qdhswcgeqbkkhdstiiqh
2. В левом меню нажмите **SQL Editor** (иконка базы данных)
3. Нажмите **New query**

### Выполните ВСЕ 4 SQL команды (скопируйте и вставьте):

```sql
-- 1. Включить pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Создать таблицу
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  book TEXT,
  chunk_id INTEGER,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT NOW()
);

-- 3. Создать индекс
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 4. Создать функцию поиска
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

**Нажмите Run (или Ctrl+Enter)**

Должно появиться: ✅ "Success. No rows returned" для каждого запроса

---

## ✅ Шаг 3: Проверить подключение

```bash
cd /Users/nurlykhan/TechOrda/Lecture_09
python 00_test_connection.py
```

**Ожидаемый результат:**
```
✅ Клиент создан
✅ Таблица 'documents' существует
📊 Документов в БД: 0
✅ Подключение работает!
```

---

## 📦 Шаг 4: Установить зависимости (если ещё не установлены)

```bash
cd /Users/nurlykhan/TechOrda/Lecture_09
pip install -r requirements.txt
```

---

## 📚 Шаг 5: Загрузить данные из книг

```bash
cd /Users/nurlykhan/TechOrda/Lecture_09
python 02_load_documents.py
```

**Это займёт 2-5 минут** (создание эмбеддингов через OpenAI API)

**Ожидаемый результат:**
```
✅ Загружено 50 документов в Supabase!
📊 Всего документов в БД: 50
```

---

## 🤖 Шаг 6: Запустить RAG пайплайн

### Вариант A: Базовый RAG

```bash
python 04_rag_pipeline.py
```

### Вариант B: RAG с LangChain (рекомендуется)

```bash
python 05_langchain_supabase.py
```

**Ожидаемый результат:**
```
❓ Вопрос: Кто такой Гарри Поттер?
📚 Найденные документы:
   1. [книга] Одиннадцатилетний мальчик-сирота Гарри Поттер...
💬 Ответ: Гарри Поттер - это одиннадцатилетний мальчик-сирота...
```

---

## 🔍 Шаг 7: Протестировать векторный поиск

```bash
python 03_vector_search.py
```

---

## 📋 Чеклист

- [ ] Service Role Key добавлен в .env ✅
- [ ] SQL команды выполнены в Supabase SQL Editor
- [ ] `python 00_test_connection.py` работает
- [ ] `python 02_load_documents.py` загрузил данные
- [ ] `python 04_rag_pipeline.py` или `05_langchain_supabase.py` работает

---

## 🆘 Если что-то не работает:

### Ошибка подключения:
```bash
# Проверьте .env файл
cat .env | grep SUPABASE
```

### Таблица не найдена:
- Убедитесь что выполнили SQL из Шага 2
- Проверьте в Supabase Dashboard → Table Editor → должна быть таблица `documents`

### Ошибка при загрузке:
- Проверьте OpenAI API ключ в .env
- Убедитесь что есть интернет

### Функция match_documents не найдена:
- Выполните SQL команду #4 из Шага 2
- Проверьте в Database → Functions


# Инструкция по настройке Supabase

## Ваши данные проекта

- **Project URL**: https://qdhswcgeqbkkhdstiiqh.supabase.co
- **Project ID**: qdhswcgeqbkkhdstiiqh
- **Publishable Key**: sb_publishable_h4XzAFWp_jHAeik70AOHjw_OfDc8S_S

## ⚠️ ВАЖНО: Нужен Service Role Key

Для работы примеров нужен **Service Role Key** (не Publishable Key).

### 📍 Где найти Service Role Key (пошаговая инструкция):

1. **Зайдите в Supabase Dashboard**
   - Откройте: https://supabase.com/dashboard
   - Войдите в свой аккаунт

2. **Выберите проект**
   - Найдите проект `TechOrdaTest` (или проект с ID: qdhswcgeqbkkhdstiiqh)
   - Нажмите на него

3. **Откройте настройки API**
   - В левом боковом меню найдите иконку **⚙️ Settings** (Настройки)
   - Нажмите на **Settings**
   - В подменю выберите **API**

4. **Найдите Service Role Key**
   - На странице API вы увидите секцию **Project API keys**
   - Там будет два ключа:
     - **`anon` `public`** — это Publishable Key (для клиентских приложений)
     - **`service_role` `secret`** — это Service Role Key (⚠️ нужен нам!)
   - Нажмите на иконку **👁️** (глаз) рядом с `service_role` чтобы показать ключ
   - Нажмите на иконку **📋** (копировать) чтобы скопировать ключ
   - ⚠️ **ВНИМАНИЕ**: Этот ключ имеет полный доступ к БД! Не публикуйте его!

5. **Обновите .env файл**
   - Откройте файл `.env` в папке `Lecture_09`
   - Замените `your-service-role-key-here` на скопированный ключ:
   ```bash
   SUPABASE_URL=https://qdhswcgeqbkkhdstiiqh.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=<вставьте-сюда-ваш-ключ>
   ```

## Шаги настройки

### 1. Включите расширение pgvector

**Где:** Supabase Dashboard → **SQL Editor**

**Как:**
1. В левом боковом меню найдите **SQL Editor** (иконка базы данных)
2. Нажмите на **SQL Editor**
3. Нажмите кнопку **New query** (Новый запрос)
4. Вставьте следующий SQL:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
5. Нажмите **Run** (или Ctrl+Enter)
6. Должно появиться сообщение об успехе: "Success. No rows returned"

### 2. Создайте таблицу для документов

**Где:** Supabase Dashboard → **SQL Editor**

**SQL запрос:**
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

**Проверка:**
- После выполнения перейдите в **Table Editor** (Редактор таблиц)
- Должна появиться таблица `documents` с колонками: id, content, book, chunk_id, embedding, created_at

### 3. Создайте индекс для быстрого векторного поиска

**Где:** Supabase Dashboard → **SQL Editor**

**SQL запрос:**
```sql
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Что делает:**
- Создаёт индекс для ускорения векторного поиска
- `ivfflat` — тип индекса для приближённого поиска
- `vector_cosine_ops` — операторы для косинусного расстояния
- `lists = 100` — параметр для баланса скорости/точности

### 4. Создайте функцию для векторного поиска

**Где:** Supabase Dashboard → **SQL Editor**

**SQL запрос:**
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

**Что делает:**
- Создаёт функцию `match_documents` для поиска похожих документов
- Использует оператор `<=>` для косинусного расстояния
- Возвращает топ-K наиболее похожих документов
- Поддерживает фильтрацию по книге

**Проверка:**
- После выполнения функция появится в **Database** → **Functions**

## Проверка подключения

После настройки запустите:

```bash
python 00_test_connection.py
```

**Ожидаемый результат:**
```
✅ Клиент создан
✅ Таблица 'documents' существует
📊 Документов в БД: 0
✅ Подключение работает!
```

Если видите ошибки:
- ❌ "SUPABASE_SERVICE_ROLE_KEY не настроен" → проверьте .env файл
- ❌ "Таблица 'documents' не найдена" → выполните SQL из шага 2
- ❌ "Ошибка подключения" → проверьте URL и ключ

## Безопасность

⚠️ **НИКОГДА** не коммитьте Service Role Key в git!
- Он имеет полный доступ к вашей БД (обходит Row Level Security)
- Используйте только для серверных приложений
- Для клиентских приложений используйте Publishable Key (anon/public)
- Добавьте `.env` в `.gitignore`

## Полезные ссылки

- [Официальная документация Supabase](https://supabase.com/docs)
- [Supabase + LangChain интеграция](https://supabase.com/docs/guides/ai/langchain)
- [pgvector документация](https://github.com/pgvector/pgvector)

## Альтернативный способ: через Database → Extensions

Вместо SQL можно включить pgvector через UI:

1. **Database** → **Extensions**
2. Найдите `vector` в списке
3. Нажмите **Enable** (Включить)

Но для создания таблицы и функции всё равно нужен SQL Editor.


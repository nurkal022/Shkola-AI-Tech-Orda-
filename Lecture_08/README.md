# Лекция 8: Введение в векторные базы данных

Примеры работы с векторными БД: FAISS и ChromaDB

## Установка

```bash
pip install -r requirements.txt
```

## Настройка

Создайте файл `.env` с вашим OpenAI API ключом:

```
OPENAI_API_KEY=sk-proj-...
```

## Примеры

### 1. Основы эмбеддингов
```bash
python 01_embeddings_basics.py
```
Демонстрирует что такое эмбеддинги и косинусное сходство.

### 2. In-memory поиск
```bash
python 02_simple_search.py
```
Простой векторный поиск без БД на данных из книг о Гарри Поттере.

### 3. FAISS
```bash
python 03_faiss_example.py
```
Быстрый векторный поиск с FAISS от Meta.

### 4. ChromaDB
```bash
python 04_chroma_example.py
```
Векторная БД на SQLite с метаданными и фильтрацией.

### 5. Сравнение
```bash
python 05_comparison.py
```
Сравнение FAISS vs ChromaDB по скорости и возможностям.

### 6. LangChain интеграция
```bash
python 06_langchain_vectorstore.py
```
Единый интерфейс LangChain для разных векторных БД + RAG пример.

## Данные

В папке `data/` находятся книги о Гарри Поттере (7 книг).

Примеры автоматически загружают и обрабатывают эти данные.

## Примечания

- Первый запуск может занять время (создание эмбеддингов)
- Для тестирования используется ограниченное количество чанков (50-200)
- FAISS индексы сохраняются в `faiss_index.bin`
- ChromaDB данные сохраняются в `chroma_db/` и `chroma_langchain_db/`


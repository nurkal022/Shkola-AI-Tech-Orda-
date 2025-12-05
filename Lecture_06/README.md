# Лекция 6: RAG (Retrieval-Augmented Generation) Architecture

## 📋 Описание
Полное руководство по созданию RAG системы - от загрузки данных до продвинутых техник поиска и генерации. На примере книг о Гарри Поттере.

## 🎯 Цели
- Понять архитектуру RAG систем
- Научиться загружать и обрабатывать текстовые данные
- Освоить стратегии chunking (разбиения текста)
- Создавать и использовать векторные хранилища
- Строить полноценные RAG pipelines
- Изучить продвинутые техники улучшения качества

## 📁 Структура материалов

```
Lecture_06/
├── data/                           # Датасет: 7 книг Гарри Поттера
│   ├── Garri_Potter_i_Filosofskiy_kamen.txt
│   ├── Garri_Potter_i_Taynaya_komnata.txt
│   └── ... (все 7 книг)
│
├── 01_data_loading.py              # Шаг 1: Загрузка данных
├── 02_text_chunking.py             # Шаг 2: Разбиение на чанки
├── 03_embeddings_vectorstore.py    # Шаг 3: Embeddings и Vector Store
├── 04_rag_pipeline.py              # Шаг 4: Базовый RAG Pipeline
├── 05_advanced_rag.py              # Шаг 5: Продвинутые техники
├── 06_interactive_chat.py          # Шаг 6: Интерактивный чат
│
├── requirements.txt                # Зависимости
└── README.md                       # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
cd Lecture_06
pip install -r requirements.txt
```

### 2. Настройка окружения
Создайте файл `.env` в корне проекта:
```env
OPENAI_API_KEY=your_api_key_here
```

### 3. Последовательный запуск
```bash
# Шаг 1: Проверка загрузки данных
python 01_data_loading.py

# Шаг 2: Анализ стратегий chunking
python 02_text_chunking.py

# Шаг 3: Создание embeddings и vector store
python 03_embeddings_vectorstore.py

# Шаг 4: Запуск базового RAG (создаст индекс)
python 04_rag_pipeline.py

# Шаг 5: Продвинутые техники RAG
python 05_advanced_rag.py

# Шаг 6: Интерактивный чат
python 06_interactive_chat.py
```

## 📚 Описание модулей

### 01_data_loading.py
- Загрузка текстовых файлов
- Извлечение метаданных из имен файлов
- Анализ корпуса текстов

### 02_text_chunking.py
- **RecursiveCharacterTextSplitter** - рекурсивное разбиение
- **CharacterTextSplitter** - простое разбиение
- Сравнение стратегий с разными размерами чанков
- Анализ качества разбиения

### 03_embeddings_vectorstore.py
- **OpenAI Embeddings** (text-embedding-3-small/large)
- **FAISS** - быстрый локальный vector store
- **Chroma** - персистентное хранилище
- Семантический поиск с оценкой релевантности

### 04_rag_pipeline.py
- Полный RAG pipeline с GPT-4.1-mini
- Retrieval → Context → Generation
- Сохранение и загрузка индексов
- Поддержка LCEL chains

### 05_advanced_rag.py
Продвинутые техники:
- **Multi-Query Retrieval** - генерация альтернативных запросов
- **Contextual Compression** - сжатие документов
- **Reranking** - переранжирование с помощью LLM
- **Parent Document Retrieval** - расширение контекста

### 06_interactive_chat.py
- Интерактивный чат-бот
- История диалога (ConversationBufferWindowMemory)
- Стриминг ответов
- Команды управления (/clear, /history, /quit)

## 🏗️ Архитектура RAG

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Query   │───▶│ Embedding│───▶│  Vector  │              │
│  │          │    │  Model   │    │  Search  │              │
│  └──────────┘    └──────────┘    └────┬─────┘              │
│                                       │                      │
│                                       ▼                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Answer  │◀───│   LLM    │◀───│ Context  │              │
│  │          │    │(GPT-4.1) │    │ Builder  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Используемые модели

| Компонент | Модель | Описание |
|-----------|--------|----------|
| LLM | gpt-4.1-mini | Генерация ответов |
| Embeddings | text-embedding-3-small | Векторизация текста |
| Vector Store | FAISS | Быстрый локальный поиск |

## 📊 Метрики датасета

| Книга | Символов | Чанков (1000) |
|-------|----------|---------------|
| Философский камень | ~500K | ~600 |
| Тайная комната | ~550K | ~650 |
| Узник Азкабана | ~650K | ~750 |
| Кубок огня | ~1.1M | ~1300 |
| Орден Феникса | ~1.5M | ~1800 |
| Принц-Полукровка | ~950K | ~1100 |
| Дары Смерти | ~1.1M | ~1300 |
| **ВСЕГО** | **~6.3M** | **~7500** |

## 💡 Tips & Best Practices

### Chunking
- Маленькие чанки (500) → точный поиск, меньше контекста
- Большие чанки (2000) → больше контекста, менее точный поиск
- Оптимально: 1000-1500 с overlap 200-300

### Embeddings
- `text-embedding-3-small` - быстрее, дешевле
- `text-embedding-3-large` - точнее для сложных задач

### Retrieval
- k=4-6 документов обычно достаточно
- Multi-query улучшает recall
- Reranking улучшает precision

## 🏆 Проект

Создайте RAG систему для своего датасета:
1. Подготовьте данные (документы, статьи, книги)
2. Реализуйте chunking с оптимальными параметрами
3. Создайте vector store
4. Настройте RAG pipeline
5. Добавьте интерактивный интерфейс

## 📖 Дополнительные ресурсы

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [FAISS Documentation](https://faiss.ai/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

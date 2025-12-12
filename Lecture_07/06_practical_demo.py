"""
Лекция 7: Практическая демонстрация
===================================
Полный пайплайн: загрузка → очистка → чанкинг → метаданные → индекс
"""

from pathlib import Path
from typing import List
import re
import unicodedata
from datetime import datetime

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
class Config:
    DATA_DIR = Path("data")
    INDEX_DIR = Path("./faiss_lecture7")
    
    # Параметры чанкинга
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Параметры очистки
    MIN_CHUNK_LENGTH = 100
    REMOVE_EXTRA_WHITESPACE = True


# ============================================================
# ШАГ 1: Загрузка документов
# ============================================================
print("="*60)
print("📥 ШАГ 1: Загрузка документов")
print("="*60)

def load_documents(data_dir: Path) -> List[Document]:
    """Загружает все текстовые файлы из директории."""
    documents = []
    
    # Загружаем только книги Гарри Поттера (игнорируем sample.txt и cleaned файлы)
    for file_path in sorted(data_dir.glob("Rouling_Djoann_*.txt")):
        text = file_path.read_text(encoding='utf-8')
        
        # Извлекаем метаданные из имени файла
        filename = file_path.stem
        book_match = re.search(r'#(\d+)', filename)
        book_num = int(book_match.group(1)) if book_match else 0
        book_title = filename.split(']_')[-1].replace('_', ' ')
        
        doc = Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "book_number": book_num,
                "book_title": book_title,
                "author": "J.K. Rowling",
                "series": "Harry Potter",
                "original_length": len(text),
                "loaded_at": datetime.now().isoformat(),
            }
        )
        documents.append(doc)
        print(f"   📖 {book_title}: {len(text):,} символов")
    
    return documents

documents = load_documents(Config.DATA_DIR)
print(f"\n   ✅ Загружено: {len(documents)} документов")
print(f"   📊 Всего: {sum(len(d.page_content) for d in documents):,} символов")


# ============================================================
# ШАГ 2: Очистка текста
# ============================================================
print("\n" + "="*60)
print("🧹 ШАГ 2: Очистка текста")
print("="*60)

def clean_text(text: str) -> str:
    """Очищает текст от мусора."""
    # Нормализация Unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Убираем множественные пробелы
    text = re.sub(r' +', ' ', text)
    
    # Убираем множественные переносы строк (больше 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Убираем пробелы в начале/конце строк
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    # Убираем спецсимволы (оставляем базовую пунктуацию)
    text = re.sub(r'[^\w\s.,!?;:\-—–\'"«»()\[\]\n]', '', text)
    
    return text.strip()

def clean_documents(docs: List[Document]) -> List[Document]:
    """Применяет очистку ко всем документам."""
    cleaned = []
    total_removed = 0
    
    for doc in docs:
        original_len = len(doc.page_content)
        cleaned_text = clean_text(doc.page_content)
        removed = original_len - len(cleaned_text)
        total_removed += removed
        
        cleaned_doc = Document(
            page_content=cleaned_text,
            metadata={
                **doc.metadata,
                "cleaned_length": len(cleaned_text),
                "chars_removed": removed,
            }
        )
        cleaned.append(cleaned_doc)
    
    return cleaned, total_removed

documents, total_removed = clean_documents(documents)
print(f"   🧹 Удалено мусора: {total_removed:,} символов")
print(f"   📊 После очистки: {sum(len(d.page_content) for d in documents):,} символов")


# ============================================================
# ШАГ 3: Чанкинг
# ============================================================
print("\n" + "="*60)
print("✂️ ШАГ 3: Чанкинг")
print("="*60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
)

chunks = splitter.split_documents(documents)

print(f"   ⚙️ Параметры:")
print(f"      - chunk_size: {Config.CHUNK_SIZE}")
print(f"      - chunk_overlap: {Config.CHUNK_OVERLAP}")
print(f"\n   📊 Результат:")
print(f"      - Документов: {len(documents)}")
print(f"      - Чанков: {len(chunks)}")

# Статистика по размерам
sizes = [len(c.page_content) for c in chunks]
print(f"      - Мин. размер: {min(sizes)}")
print(f"      - Макс. размер: {max(sizes)}")
print(f"      - Средний: {sum(sizes)//len(sizes)}")


# ============================================================
# ШАГ 4: Обогащение метаданными
# ============================================================
print("\n" + "="*60)
print("🏷️ ШАГ 4: Обогащение метаданными")
print("="*60)

def enrich_chunk(chunk: Document, chunk_id: int, total_chunks: int) -> Document:
    """Добавляет метаданные к чанку."""
    text = chunk.page_content
    
    # Позиционные метаданные
    chunk.metadata['chunk_id'] = chunk_id
    chunk.metadata['chunk_total'] = total_chunks
    chunk.metadata['position'] = round(chunk_id / total_chunks, 2)
    
    # Контентные метаданные
    chunk.metadata['char_count'] = len(text)
    chunk.metadata['word_count'] = len(text.split())
    
    # Первое предложение как preview
    first_sentence = text.split('.')[0][:100] if '.' in text else text[:100]
    chunk.metadata['preview'] = first_sentence
    
    # Определяем тип контента
    chunk.metadata['has_dialog'] = any(c in text for c in ['—', '«', '"'])
    
    return chunk

# Группируем чанки по книгам для правильной нумерации
chunks_by_book = {}
for chunk in chunks:
    book = chunk.metadata['book_number']
    if book not in chunks_by_book:
        chunks_by_book[book] = []
    chunks_by_book[book].append(chunk)

# Обогащаем
enriched_chunks = []
for book_num, book_chunks in sorted(chunks_by_book.items()):
    for i, chunk in enumerate(book_chunks):
        enriched = enrich_chunk(chunk, i, len(book_chunks))
        enriched_chunks.append(enriched)

print(f"   ✅ Обогащено: {len(enriched_chunks)} чанков")
print(f"\n   📋 Пример метаданных:")
sample_meta = enriched_chunks[100].metadata
for k, v in sample_meta.items():
    val = str(v)[:40] + "..." if len(str(v)) > 40 else v
    print(f"      - {k}: {val}")


# ============================================================
# ШАГ 5: Фильтрация плохих чанков
# ============================================================
print("\n" + "="*60)
print("🔍 ШАГ 5: Фильтрация плохих чанков")
print("="*60)

def is_valid_chunk(chunk: Document, min_length: int = 100) -> bool:
    """Проверяет валидность чанка."""
    text = chunk.page_content
    
    # Слишком короткий
    if len(text) < min_length:
        return False
    
    # Слишком много повторов (признак мусора)
    words = text.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # Менее 30% уникальных слов
            return False
    
    return True

before_count = len(enriched_chunks)
filtered_chunks = [c for c in enriched_chunks if is_valid_chunk(c, Config.MIN_CHUNK_LENGTH)]
removed_count = before_count - len(filtered_chunks)

print(f"   🗑️ Удалено плохих чанков: {removed_count}")
print(f"   ✅ Осталось: {len(filtered_chunks)} чанков")


# ============================================================
# ШАГ 6: Создание векторного индекса
# ============================================================
print("\n" + "="*60)
print("📦 ШАГ 6: Создание векторного индекса")
print("="*60)

print("   🔄 Создание эмбеддингов...")
embeddings = OpenAIEmbeddings()

# Создаём FAISS индекс
vectorstore = FAISS.from_documents(filtered_chunks, embeddings)

# Сохраняем
Config.INDEX_DIR.mkdir(exist_ok=True)
vectorstore.save_local(str(Config.INDEX_DIR))

print(f"   ✅ Индекс создан и сохранён в: {Config.INDEX_DIR}")
print(f"   📊 Размер индекса: {len(filtered_chunks)} векторов")


# ============================================================
# ШАГ 7: Тестирование
# ============================================================
print("\n" + "="*60)
print("🧪 ШАГ 7: Тестирование")
print("="*60)

test_queries = [
    "Как выглядит шрам Гарри Поттера?",
    "Кто такие Дурсли?",
    "Что делает философский камень?",
]

for query in test_queries:
    print(f"\n   ❓ {query}")
    
    results = vectorstore.similarity_search_with_score(query, k=2)
    
    for doc, score in results:
        book = doc.metadata.get('book_title', 'N/A')
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"      📖 [{book[:20]}] (score: {score:.3f})")
        print(f"         {preview}...")


# ============================================================
# ШАГ 8: Тестирование с фильтрами
# ============================================================
print("\n" + "="*60)
print("🎯 ШАГ 8: Поиск с фильтрами")
print("="*60)

# Поиск только в первой книге
query = "волшебная палочка"
print(f"\n   ❓ '{query}' (только книга 1)")

# FAISS поддерживает фильтрацию через filter
results = vectorstore.similarity_search(
    query, 
    k=3,
    filter={"book_number": 1}
)

for doc in results:
    book = doc.metadata.get('book_title', 'N/A')
    print(f"      📖 [{book[:30]}]")
    print(f"         {doc.page_content[:80]}...")


# ============================================================
# ИТОГОВАЯ СТАТИСТИКА
# ============================================================
print("\n" + "="*60)
print("📊 ИТОГОВАЯ СТАТИСТИКА ПАЙПЛАЙНА")
print("="*60)
print(f"""
   Документов загружено:     {len(documents)}
   Символов до очистки:      {sum(d.metadata['original_length'] for d in documents):,}
   Символов после очистки:   {sum(len(d.page_content) for d in documents):,}
   Чанков создано:           {len(chunks)}
   Чанков после фильтрации:  {len(filtered_chunks)}
   Векторов в индексе:       {len(filtered_chunks)}
   
   Параметры:
   - chunk_size:    {Config.CHUNK_SIZE}
   - chunk_overlap: {Config.CHUNK_OVERLAP}
   - min_length:    {Config.MIN_CHUNK_LENGTH}
""")

print("\n   ✅ Пайплайн завершён! Индекс готов к использованию.")


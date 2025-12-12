"""
Лекция 7: Обогащение чанков метаданными
=======================================
Метаданные улучшают поиск и фильтрацию.
"""

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import hashlib
from datetime import datetime


# ============================================================
# 1. Базовые метаданные
# ============================================================
print("="*60)
print("1️⃣ Базовые метаданные")
print("="*60)

# Загружаем текст
data_dir = Path("data")
files = list(data_dir.glob("Rouling_Djoann_*.txt"))  # Только книги Гарри Поттера

# Создаём документы с метаданными
documents = []
for file_path in files[:3]:  # Первые 3 книги для демо
    text = file_path.read_text(encoding='utf-8')
    
    # Извлекаем информацию из имени файла
    # Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt
    filename = file_path.stem
    book_num = int(re.search(r'#(\d+)', filename).group(1))
    book_title = filename.split(']_')[-1].replace('_', ' ')
    
    doc = Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "filename": filename,
            "book_number": book_num,
            "book_title": book_title,
            "author": "J.K. Rowling",
            "series": "Harry Potter",
            "language": "ru",
            "char_count": len(text),
            "word_count": len(text.split()),
            "loaded_at": datetime.now().isoformat(),
        }
    )
    documents.append(doc)
    print(f"   📖 {book_title}")
    print(f"      Метаданные: {list(doc.metadata.keys())}")


# ============================================================
# 2. Наследование метаданных при чанкинге
# ============================================================
print("\n" + "="*60)
print("2️⃣ Наследование метаданных при чанкинге")
print("="*60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# LangChain автоматически наследует метаданные
chunks = splitter.split_documents(documents[:1])  # Только первая книга

print(f"   Документов: {len(documents[:1])}")
print(f"   После чанкинга: {len(chunks)} чанков")
print(f"\n   Пример метаданных чанка:")
for k, v in chunks[5].metadata.items():
    val = str(v)[:50] + "..." if len(str(v)) > 50 else v
    print(f"   - {k}: {val}")


# ============================================================
# 3. Добавление позиционных метаданных
# ============================================================
print("\n" + "="*60)
print("3️⃣ Позиционные метаданные")
print("="*60)

def add_positional_metadata(chunks: List[Document]) -> List[Document]:
    """Добавляет позиционную информацию к чанкам."""
    total = len(chunks)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "chunk_total": total,
            "chunk_position": round(i / total, 2),  # 0.0 - начало, 1.0 - конец
            "is_first": i == 0,
            "is_last": i == total - 1,
        })
    
    return chunks

chunks = add_positional_metadata(chunks)

print("   Добавлены:")
print("   - chunk_id: порядковый номер")
print("   - chunk_total: общее количество")
print("   - chunk_position: позиция (0.0 - 1.0)")
print("   - is_first / is_last: флаги")

print(f"\n   Пример (чанк 10):")
print(f"   - chunk_id: {chunks[10].metadata['chunk_id']}")
print(f"   - chunk_position: {chunks[10].metadata['chunk_position']}")


# ============================================================
# 4. Извлечение контентных метаданных
# ============================================================
print("\n" + "="*60)
print("4️⃣ Контентные метаданные (извлечённые из текста)")
print("="*60)

def extract_content_metadata(chunk: Document) -> Document:
    """Извлекает метаданные из содержимого чанка."""
    text = chunk.page_content
    
    # Первое предложение как summary
    first_sentence = text.split('.')[0][:150] if '.' in text else text[:150]
    
    # Ключевые слова (простой вариант - частотный анализ)
    words = re.findall(r'\b[А-Яа-яЁё]{4,}\b', text.lower())
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:5]
    keywords = [w[0] for w in top_words]
    
    # Определяем тип контента
    has_dialog = '—' in text or '"' in text or '«' in text
    has_numbers = bool(re.search(r'\d+', text))
    
    # Обнаружение имён (простой паттерн - слова с заглавной буквы)
    names = set(re.findall(r'\b[A-ZА-ЯЁ][a-zа-яё]+\b', text))
    # Фильтруем начала предложений (упрощённо)
    
    chunk.metadata.update({
        "first_sentence": first_sentence,
        "keywords": keywords,
        "has_dialog": has_dialog,
        "has_numbers": has_numbers,
        "potential_names": list(names)[:10],
        "avg_sentence_length": len(text) / (text.count('.') + 1),
    })
    
    return chunk

# Применяем к нескольким чанкам
for i in [0, 10, 50]:
    chunks[i] = extract_content_metadata(chunks[i])

print("   Пример извлечённых метаданных (чанк 10):")
print(f"   - first_sentence: {chunks[10].metadata['first_sentence'][:60]}...")
print(f"   - keywords: {chunks[10].metadata['keywords']}")
print(f"   - has_dialog: {chunks[10].metadata['has_dialog']}")
print(f"   - potential_names: {chunks[10].metadata['potential_names'][:5]}")


# ============================================================
# 5. Структурные метаданные (главы, разделы)
# ============================================================
print("\n" + "="*60)
print("5️⃣ Структурные метаданные")
print("="*60)

def detect_chapter_structure(text: str) -> List[Dict[str, Any]]:
    """Определяет структуру глав в тексте."""
    # Паттерн для глав
    chapter_pattern = r'(Глава\s+\d+[^\n]*|ГЛАВА\s+\d+[^\n]*)'
    
    chapters = []
    matches = list(re.finditer(chapter_pattern, text))
    
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        
        chapters.append({
            "chapter_title": match.group().strip(),
            "chapter_num": i + 1,
            "start_pos": start,
            "end_pos": end,
            "length": end - start,
        })
    
    return chapters

# Демонстрация
sample_text = documents[0].page_content
chapters = detect_chapter_structure(sample_text)
print(f"   Найдено глав: {len(chapters)}")
for ch in chapters[:3]:
    print(f"   - {ch['chapter_title'][:40]}... ({ch['length']:,} символов)")


def add_chapter_metadata(chunks: List[Document], chapters: List[Dict]) -> List[Document]:
    """Добавляет информацию о главе к каждому чанку."""
    # Определяем позицию каждого чанка в исходном тексте (упрощённо)
    # В реальности нужно отслеживать offset при чанкинге
    
    for chunk in chunks:
        text_preview = chunk.page_content[:100]
        
        # Ищем, в какую главу попадает начало чанка
        for ch in chapters:
            if ch['chapter_title'][:20] in text_preview:
                chunk.metadata['chapter'] = ch['chapter_title']
                chunk.metadata['chapter_num'] = ch['chapter_num']
                break
    
    return chunks

print("\n   💡 Для точного определения главы нужно:")
print("   - Отслеживать offset при чанкинге")
print("   - Или использовать MarkdownHeaderTextSplitter для структурированных документов")


# ============================================================
# 6. Хеширование для дедупликации
# ============================================================
print("\n" + "="*60)
print("6️⃣ Хеширование для дедупликации")
print("="*60)

def add_hash_metadata(chunks: List[Document]) -> List[Document]:
    """Добавляет хеш содержимого для дедупликации."""
    for chunk in chunks:
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
        chunk.metadata['content_hash'] = content_hash
        chunk.metadata['content_hash_short'] = content_hash[:8]
    
    return chunks

chunks = add_hash_metadata(chunks)

print("   Хеш полезен для:")
print("   - Дедупликации при обновлении индекса")
print("   - Отслеживания изменений")
print("   - Идентификации одинаковых чанков")

print(f"\n   Пример: {chunks[0].metadata['content_hash_short']}")


# ============================================================
# 7. Гипотетические вопросы (HyDE-style)
# ============================================================
print("\n" + "="*60)
print("7️⃣ Гипотетические вопросы к чанку")
print("="*60)

print("""
   Идея: Генерировать вопросы, на которые чанк может ответить.
   Это улучшает retrieval, когда запрос пользователя - вопрос.
   
   Реализация с LLM:
   
   from langchain_openai import ChatOpenAI
   
   llm = ChatOpenAI(model="gpt-4o-mini")
   
   def generate_questions(chunk: Document) -> Document:
       prompt = f'''
       На основе текста сгенерируй 3 вопроса, на которые этот текст отвечает.
       
       Текст: {chunk.page_content[:500]}
       
       Вопросы (по одному на строку):
       '''
       
       response = llm.invoke(prompt)
       questions = response.content.strip().split('\\n')
       
       chunk.metadata['hypothetical_questions'] = questions
       return chunk
   
   Плюсы:
   + Значительно улучшает retrieval для вопросов
   + Расширяет семантическое покрытие чанка
   
   Минусы:
   - Дорого (нужен LLM для каждого чанка)
   - Время на индексацию
""")


# ============================================================
# 8. Полный пайплайн обогащения
# ============================================================
print("\n" + "="*60)
print("8️⃣ Полный пайплайн обогащения")
print("="*60)

@dataclass
class ChunkEnricher:
    """Пайплайн обогащения чанков метаданными."""
    
    add_positional: bool = True
    add_content: bool = True
    add_hash: bool = True
    extract_keywords: bool = True
    
    def enrich(self, chunks: List[Document]) -> List[Document]:
        """Применяет все этапы обогащения."""
        if self.add_positional:
            chunks = add_positional_metadata(chunks)
        
        if self.add_content:
            chunks = [extract_content_metadata(c) for c in chunks]
        
        if self.add_hash:
            chunks = add_hash_metadata(chunks)
        
        return chunks

# Использование
enricher = ChunkEnricher(
    add_positional=True,
    add_content=True,
    add_hash=True,
)

# enriched_chunks = enricher.enrich(chunks)

print("""
   Пример использования:
   
   enricher = ChunkEnricher(
       add_positional=True,
       add_content=True,
       add_hash=True,
   )
   
   enriched_chunks = enricher.enrich(raw_chunks)
   
   # Теперь каждый чанк содержит богатые метаданные
   # для фильтрации и улучшенного поиска
""")


# ============================================================
# 9. Использование метаданных при поиске
# ============================================================
print("\n" + "="*60)
print("9️⃣ Использование метаданных при поиске")
print("="*60)

print("""
   Метаданные позволяют:
   
   1. Фильтрация перед поиском:
      results = vectorstore.similarity_search(
          query,
          filter={"book_number": 1}  # Только первая книга
      )
   
   2. Пост-фильтрация:
      results = vectorstore.similarity_search(query, k=20)
      results = [r for r in results if r.metadata['has_dialog']]
   
   3. Ре-ранжирование:
      def rerank(results, query):
          for r in results:
              # Повышаем score если chunk_position ближе к началу
              if r.metadata['chunk_position'] < 0.3:
                  r.score *= 1.2
          return sorted(results, key=lambda x: x.score)
   
   4. Группировка результатов:
      from itertools import groupby
      
      results = vectorstore.similarity_search(query, k=10)
      grouped = groupby(results, key=lambda x: x.metadata['book_title'])
""")


# ============================================================
# РЕКОМЕНДАЦИИ
# ============================================================
print("\n" + "="*60)
print("💡 РЕКОМЕНДАЦИИ ПО МЕТАДАННЫМ")
print("="*60)
print("""
   1. Обязательные метаданные:
      - source (файл/URL источника)
      - chunk_id (идентификатор)
      - created_at (время создания)
   
   2. Полезные для фильтрации:
      - document_type (тип документа)
      - category / tags (категории)
      - language (язык)
      - date (дата документа)
   
   3. Полезные для качества:
      - content_hash (дедупликация)
      - chunk_position (контекст)
      - hypothetical_questions (retrieval)
   
   4. Не перегружайте:
      - Храните только нужные поля
      - Избегайте дублирования
      - Используйте индексы для фильтрации
   
   5. Планируйте заранее:
      - Какие фильтры понадобятся?
      - Нужна ли группировка?
      - Как будет обновляться индекс?
""")


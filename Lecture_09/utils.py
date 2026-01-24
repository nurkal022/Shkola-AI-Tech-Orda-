"""
Утилиты для работы с документами
Использует рекурсивный чанкинг из LangChain
"""

import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_book(filepath: str) -> str:
    """Прочитать документ из файла"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def split_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """
    Разбить текст на чанки используя рекурсивный чанкинг
    
    RecursiveCharacterTextSplitter пытается разбить текст по границам:
    1. Параграфы (\n\n)
    2. Предложения (. ! ?)
    3. Слова (пробелы)
    4. Символы (если ничего не подошло)
    
    Args:
        text: Исходный текст
        chunk_size: Целевой размер чанка в символах
        chunk_overlap: Перекрытие между чанками
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Порядок важен!
    )
    
    chunks = splitter.split_text(text)
    return chunks


def get_book_chunks(book_path: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[dict]:
    """
    Получить чанки из документа с метаданными используя рекурсивный чанкинг
    
    Args:
        book_path: Путь к файлу
        chunk_size: Размер чанка (по умолчанию 1500 для ~100 чанков)
        chunk_overlap: Перекрытие между чанками
    
    Returns:
        List[dict] с ключами: text, book, chunk_id
    """
    book_name = os.path.basename(book_path)
    text = read_book(book_path)
    chunks = split_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "text": chunk,
            "book": book_name,
            "chunk_id": i
        })
    
    return result


def get_all_books_chunks(data_dir: str = "data", chunk_size: int = 1500, chunk_overlap: int = 200, max_chunks: int = None) -> List[dict]:
    """
    Получить чанки из всех документов в директории используя рекурсивный чанкинг
    
    Args:
        data_dir: Путь к директории с документами
        chunk_size: Размер чанка (по умолчанию 1500 для ~100 чанков)
        chunk_overlap: Перекрытие между чанками
        max_chunks: Максимальное количество чанков (для тестирования)
    """
    all_chunks = []
    
    # Находим все txt файлы
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    txt_files.sort()  # Сортируем для воспроизводимости
    
    print(f"Найдено документов: {len(txt_files)}")
    
    for txt_file in txt_files:
        filepath = os.path.join(data_dir, txt_file)
        print(f"  Обрабатываем: {txt_file}...")
        chunks = get_book_chunks(filepath, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
        print(f"    Добавлено чанков: {len(chunks)}")
        
        if max_chunks and len(all_chunks) >= max_chunks:
            all_chunks = all_chunks[:max_chunks]
            break
    
    print(f"\nВсего чанков: {len(all_chunks)}")
    return all_chunks


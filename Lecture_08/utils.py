"""
Утилиты для работы с данными из книг о Гарри Поттере
"""

import os
import re
from typing import List


def read_book(filepath: str) -> str:
    """Прочитать книгу из файла"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Разбить текст на чанки с перекрытием
    
    Args:
        text: Исходный текст
        chunk_size: Размер чанка в символах
        overlap: Перекрытие между чанками
    """
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Пытаемся закончить на границе предложения
        if end < len(text):
            last_period = chunk.rfind('.')
            last_exclamation = chunk.rfind('!')
            last_question = chunk.rfind('?')
            last_newline = chunk.rfind('\n')
            
            last_break = max(last_period, last_exclamation, last_question, last_newline)
            if last_break > chunk_size * 0.5:  # Если нашли разумную границу
                chunk = chunk[:last_break + 1]
                end = start + last_break + 1
        
        chunks.append(chunk.strip())
        start = end - overlap  # Перекрытие
    
    return chunks


def get_book_chunks(book_path: str, chunk_size: int = 500) -> List[dict]:
    """
    Получить чанки из книги с метаданными
    
    Returns:
        List[dict] с ключами: text, book, chunk_id
    """
    book_name = os.path.basename(book_path)
    text = read_book(book_path)
    chunks = split_into_chunks(text, chunk_size=chunk_size)
    
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "text": chunk,
            "book": book_name,
            "chunk_id": i
        })
    
    return result


def get_all_books_chunks(data_dir: str = "data", chunk_size: int = 500, max_chunks: int = None) -> List[dict]:
    """
    Получить чанки из всех книг в директории
    
    Args:
        data_dir: Путь к директории с книгами
        max_chunks: Максимальное количество чанков (для тестирования)
    """
    all_chunks = []
    
    # Находим все txt файлы
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    txt_files.sort()  # Сортируем для воспроизводимости
    
    print(f"Найдено книг: {len(txt_files)}")
    
    for txt_file in txt_files:
        filepath = os.path.join(data_dir, txt_file)
        print(f"  Обрабатываем: {txt_file}...")
        chunks = get_book_chunks(filepath, chunk_size=chunk_size)
        all_chunks.extend(chunks)
        print(f"    Добавлено чанков: {len(chunks)}")
        
        if max_chunks and len(all_chunks) >= max_chunks:
            all_chunks = all_chunks[:max_chunks]
            break
    
    print(f"\nВсего чанков: {len(all_chunks)}")
    return all_chunks


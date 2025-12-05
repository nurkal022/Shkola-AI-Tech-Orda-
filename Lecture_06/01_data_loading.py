"""
Шаг 1: Загрузка и предобработка данных
=====================================
Этот модуль демонстрирует различные способы загрузки текстовых данных
для RAG системы.
"""

import os
from pathlib import Path
from typing import List, Dict


def load_single_file(file_path: str) -> str:
    """Загрузка одного текстового файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_all_books(data_dir: str = "data") -> Dict[str, str]:
    """
    Загрузка всех книг из директории
    
    Returns:
        Dict с названием книги как ключ и текстом как значение
    """
    books = {}
    data_path = Path(data_dir)
    
    for file_path in sorted(data_path.glob("*.txt")):
        # Извлекаем номер и название книги из имени файла
        filename = file_path.stem
        # Парсим название: Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen
        parts = filename.split(']_')
        if len(parts) > 1:
            book_name = parts[1].replace('_', ' ')
        else:
            book_name = filename
            
        books[book_name] = load_single_file(str(file_path))
        print(f"✅ Загружена: {book_name} ({len(books[book_name]):,} символов)")
    
    return books


def get_book_metadata(filename: str) -> Dict:
    """Извлечение метаданных из имени файла"""
    # Пример: Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt
    metadata = {
        "author": "Джоан Роулинг",
        "series": "Гарри Поттер",
        "filename": filename
    }
    
    # Извлекаем номер книги
    if '#' in filename:
        try:
            book_num = filename.split('#')[1].split(']')[0]
            metadata["book_number"] = int(book_num)
        except:
            metadata["book_number"] = 0
    
    # Извлекаем название
    parts = filename.split(']_')
    if len(parts) > 1:
        metadata["title"] = parts[1].replace('_', ' ').replace('.txt', '')
    
    return metadata


def analyze_corpus(books: Dict[str, str]) -> None:
    """Анализ загруженного корпуса"""
    print("\n" + "="*60)
    print("📊 АНАЛИЗ КОРПУСА")
    print("="*60)
    
    total_chars = 0
    total_words = 0
    
    for name, text in books.items():
        chars = len(text)
        words = len(text.split())
        total_chars += chars
        total_words += words
        print(f"📖 {name}")
        print(f"   Символов: {chars:,} | Слов: {words:,}")
    
    print("-"*60)
    print(f"📚 ВСЕГО: {len(books)} книг")
    print(f"   Символов: {total_chars:,}")
    print(f"   Слов: {total_words:,}")
    print(f"   Среднее слов/книга: {total_words // len(books):,}")


if __name__ == "__main__":
    # Демонстрация загрузки данных
    print("🚀 Загрузка данных для RAG системы\n")
    
    books = load_all_books("data")
    analyze_corpus(books)
    
    # Пример получения метаданных
    print("\n" + "="*60)
    print("📋 МЕТАДАННЫЕ ПЕРВОЙ КНИГИ")
    print("="*60)
    
    first_file = list(Path("data").glob("*.txt"))[0]
    metadata = get_book_metadata(first_file.name)
    for key, value in metadata.items():
        print(f"   {key}: {value}")


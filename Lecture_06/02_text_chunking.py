"""
Шаг 2: Разбиение текста на чанки (Chunking)
==========================================
Критический этап для RAG - правильное разбиение текста влияет на качество поиска.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain.schema import Document


@dataclass
class ChunkConfig:
    """Конфигурация для chunking"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            # Русскоязычные разделители
            self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


def create_recursive_splitter(config: ChunkConfig) -> RecursiveCharacterTextSplitter:
    """
    Рекурсивный сплиттер - лучший выбор для большинства случаев.
    Пытается разбить по более крупным разделителям, затем по мелким.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
        length_function=len,
    )


def create_character_splitter(config: ChunkConfig) -> CharacterTextSplitter:
    """
    Простой сплиттер по символам.
    Разбивает строго по указанному разделителю.
    """
    return CharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separator="\n\n",
    )


def chunk_text(text: str, splitter) -> List[str]:
    """Разбиение текста на чанки"""
    return splitter.split_text(text)


def chunk_with_metadata(
    text: str, 
    metadata: Dict[str, Any],
    splitter
) -> List[Document]:
    """
    Разбиение текста с сохранением метаданных.
    Каждый чанк получает метаданные + номер чанка.
    """
    chunks = splitter.split_text(text)
    documents = []
    
    for i, chunk in enumerate(chunks):
        doc_metadata = metadata.copy()
        doc_metadata["chunk_id"] = i
        doc_metadata["chunk_total"] = len(chunks)
        
        documents.append(Document(
            page_content=chunk,
            metadata=doc_metadata
        ))
    
    return documents


def compare_chunking_strategies(text: str) -> Dict[str, List[str]]:
    """
    Сравнение разных стратегий chunking.
    Полезно для выбора оптимальной стратегии.
    """
    results = {}
    
    # Разные размеры чанков
    sizes = [500, 1000, 1500, 2000]
    
    for size in sizes:
        config = ChunkConfig(chunk_size=size, chunk_overlap=size // 5)
        splitter = create_recursive_splitter(config)
        chunks = chunk_text(text, splitter)
        results[f"recursive_{size}"] = chunks
        
    return results


def analyze_chunks(chunks: List[str], name: str = "chunks") -> None:
    """Анализ качества разбиения"""
    if not chunks:
        print(f"❌ {name}: пустой список чанков")
        return
        
    lengths = [len(c) for c in chunks]
    
    print(f"\n📊 Анализ '{name}':")
    print(f"   Количество чанков: {len(chunks)}")
    print(f"   Мин. размер: {min(lengths):,} символов")
    print(f"   Макс. размер: {max(lengths):,} символов")
    print(f"   Средний размер: {sum(lengths) // len(lengths):,} символов")
    print(f"   Общий размер: {sum(lengths):,} символов")


def demo_chunking():
    """Демонстрация различных стратегий chunking"""
    from pathlib import Path
    
    # Загружаем первую книгу для демонстрации
    data_path = Path("data")
    first_book = list(data_path.glob("*.txt"))[0]
    
    with open(first_book, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("="*60)
    print("🔪 ДЕМОНСТРАЦИЯ CHUNKING СТРАТЕГИЙ")
    print("="*60)
    print(f"📖 Файл: {first_book.name}")
    print(f"📏 Размер: {len(text):,} символов")
    
    # Стратегия 1: Маленькие чанки (для точного поиска)
    config_small = ChunkConfig(chunk_size=500, chunk_overlap=100)
    splitter_small = create_recursive_splitter(config_small)
    chunks_small = chunk_text(text, splitter_small)
    analyze_chunks(chunks_small, "Маленькие чанки (500)")
    
    # Стратегия 2: Средние чанки (баланс)
    config_medium = ChunkConfig(chunk_size=1000, chunk_overlap=200)
    splitter_medium = create_recursive_splitter(config_medium)
    chunks_medium = chunk_text(text, splitter_medium)
    analyze_chunks(chunks_medium, "Средние чанки (1000)")
    
    # Стратегия 3: Большие чанки (больше контекста)
    config_large = ChunkConfig(chunk_size=2000, chunk_overlap=400)
    splitter_large = create_recursive_splitter(config_large)
    chunks_large = chunk_text(text, splitter_large)
    analyze_chunks(chunks_large, "Большие чанки (2000)")
    
    # Показываем пример чанка
    print("\n" + "="*60)
    print("📝 ПРИМЕР ЧАНКА (средний размер):")
    print("="*60)
    print(chunks_medium[10][:500] + "...")
    
    return chunks_medium


if __name__ == "__main__":
    demo_chunking()


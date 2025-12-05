"""
RAG Шаг 1: Загрузка данных
"""
from pathlib import Path

# === Загрузка всех книг ===
def load_books(data_dir="data"):
    books = {}
    for file in sorted(Path(data_dir).glob("*.txt")):
        # Извлекаем название из имени файла
        name = file.stem.split(']_')[-1].replace('_', ' ')
        books[name] = file.read_text(encoding='utf-8')
        print(f"📖 {name}: {len(books[name]):,} символов")
    return books


if __name__ == "__main__":
    books = load_books()
    
    print(f"\n{'='*50}")
    print(f"📚 Загружено: {len(books)} книг")
    print(f"📊 Всего символов: {sum(len(t) for t in books.values()):,}")
    print(f"📊 Всего слов: {sum(len(t.split()) for t in books.values()):,}")

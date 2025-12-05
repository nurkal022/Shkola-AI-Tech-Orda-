"""
RAG Шаг 2: Сравнение стратегий Chunking
=======================================
Сравниваем популярные методы разбиения текста на чанки.
"""
from pathlib import Path
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# Загрузка тестового текста (первая книга)
text = list(Path("data").glob("*.txt"))[0].read_text(encoding='utf-8')
print(f"📖 Тестовый текст: {len(text):,} символов\n")

# ============================================================
# 1. CharacterTextSplitter - Простое разбиение по символам
# ============================================================
print("="*60)
print("1️⃣ CharacterTextSplitter")
print("   Разбивает по указанному разделителю (по умолчанию \\n\\n)")
print("="*60)

splitter1 = CharacterTextSplitter(
    separator="\n\n",      # Разделитель
    chunk_size=1000,       # Макс размер чанка
    chunk_overlap=200,     # Перекрытие между чанками
)
chunks1 = splitter1.split_text(text)
print(f"   Чанков: {len(chunks1)}")
print(f"   Размеры: min={min(len(c) for c in chunks1)}, max={max(len(c) for c in chunks1)}")


# ============================================================
# 2. RecursiveCharacterTextSplitter - Рекурсивное разбиение
# ============================================================
print("\n" + "="*60)
print("2️⃣ RecursiveCharacterTextSplitter (РЕКОМЕНДУЕТСЯ)")
print("   Пробует разбить по списку разделителей по очереди")
print("="*60)

splitter2 = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
    chunk_size=1000,
    chunk_overlap=200,
)
chunks2 = splitter2.split_text(text)
print(f"   Чанков: {len(chunks2)}")
print(f"   Размеры: min={min(len(c) for c in chunks2)}, max={max(len(c) for c in chunks2)}")


# ============================================================
# 3. TokenTextSplitter - Разбиение по токенам
# ============================================================
print("\n" + "="*60)
print("3️⃣ TokenTextSplitter")
print("   Разбивает по токенам (важно для LLM с лимитом токенов)")
print("="*60)

splitter3 = TokenTextSplitter(
    chunk_size=256,        # Токенов, не символов!
    chunk_overlap=50,
)
chunks3 = splitter3.split_text(text)
print(f"   Чанков: {len(chunks3)}")
print(f"   Размеры (символы): min={min(len(c) for c in chunks3)}, max={max(len(c) for c in chunks3)}")


# ============================================================
# СРАВНЕНИЕ РАЗМЕРОВ ЧАНКОВ
# ============================================================
print("\n" + "="*60)
print("📊 СРАВНЕНИЕ: Разные размеры чанков")
print("="*60)

for chunk_size in [500, 1000, 1500, 2000]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5,  # 20% overlap
    )
    chunks = splitter.split_text(text)
    avg_size = sum(len(c) for c in chunks) // len(chunks)
    print(f"   chunk_size={chunk_size}: {len(chunks)} чанков, avg={avg_size} символов")


# ============================================================
# СРАВНЕНИЕ OVERLAP
# ============================================================
print("\n" + "="*60)
print("📊 СРАВНЕНИЕ: Разный overlap")
print("="*60)

for overlap in [0, 100, 200, 300]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_text(text)
    print(f"   overlap={overlap}: {len(chunks)} чанков")


# ============================================================
# ПРИМЕР ЧАНКОВ
# ============================================================
print("\n" + "="*60)
print("📝 ПРИМЕР ЧАНКА (RecursiveCharacterTextSplitter)")
print("="*60)
print(chunks2[5][:400] + "...")


# ============================================================
# РЕКОМЕНДАЦИИ
# ============================================================
print("\n" + "="*60)
print("💡 РЕКОМЕНДАЦИИ")
print("="*60)
print("""
   • RecursiveCharacterTextSplitter - лучший выбор для большинства случаев
   • chunk_size=1000-1500 - хороший баланс контекст/точность
   • chunk_overlap=200-300 - сохраняет контекст между чанками
   
   Маленькие чанки (500):  + точный поиск, - мало контекста
   Большие чанки (2000):   + много контекста, - менее точный поиск
""")

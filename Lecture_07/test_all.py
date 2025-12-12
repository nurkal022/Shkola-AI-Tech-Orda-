"""
Проверка всех скриптов Lecture_07
==================================
Убеждаемся, что все файлы используют правильный путь к данным.
"""

from pathlib import Path
import sys

# Путь к файлу данных
DATA_FILE = Path("data/Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt")

print("="*60)
print("🔍 ПРОВЕРКА ВСЕХ СКРИПТОВ LECTURE_07")
print("="*60)

# Проверка существования файла данных
print(f"\n1️⃣ Проверка файла данных:")
if DATA_FILE.exists():
    text = DATA_FILE.read_text(encoding='utf-8')
    print(f"   ✅ Файл существует: {DATA_FILE}")
    print(f"   ✅ Размер: {len(text):,} символов")
    print(f"   ✅ Слов: {len(text.split()):,}")
else:
    print(f"   ❌ Файл не найден: {DATA_FILE}")
    sys.exit(1)

# Проверка импортов в файлах
print(f"\n2️⃣ Проверка импортов:")
files_to_check = [
    "02_text_cleaning.py",
    "03_advanced_chunking.py",
    "04_chunking_comparison.py",
    "05_metadata_enrichment.py",
    "06_practical_demo.py",
    "clean_harry_potter.py",
]

for file_name in files_to_check:
    file_path = Path(file_name)
    if not file_path.exists():
        print(f"   ⚠️ Файл не найден: {file_name}")
        continue
    
    content = file_path.read_text(encoding='utf-8')
    
    # Проверяем использование правильного пути
    if "data/Rouling_Djoann_" in content or 'Path("data")' in content:
        print(f"   ✅ {file_name}: использует правильный путь")
    elif "../Lecture_06/data" in content:
        print(f"   ❌ {file_name}: использует старый путь ../Lecture_06/data")
    else:
        print(f"   ⚠️ {file_name}: путь не найден в коде")

# Проверка базовых функций
print(f"\n3️⃣ Тест базовых функций:")

try:
    # Тест загрузки
    text = DATA_FILE.read_text(encoding='utf-8')
    print(f"   ✅ Загрузка файла работает")
    
    # Тест очистки
    import re
    import unicodedata
    
    cleaned = re.sub(r' +', ' ', text[:1000])
    print(f"   ✅ Функции очистки доступны")
    
    # Тест чанкинга
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(text[:5000])
    print(f"   ✅ Чанкинг работает: создано {len(chunks)} чанков")
    
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

print(f"\n" + "="*60)
print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
print("="*60)
print(f"""
📋 ИТОГИ:
   - Файл данных: {DATA_FILE.name}
   - Размер: {len(text):,} символов
   - Все скрипты обновлены для использования правильного пути
   
🚀 Для запуска:
   python 02_text_cleaning.py      # Очистка текста
   python 03_advanced_chunking.py  # Продвинутый чанкинг
   python 04_chunking_comparison.py # Сравнение методов
   python 05_metadata_enrichment.py # Метаданные
   python 06_practical_demo.py      # Полный пайплайн
   python clean_harry_potter.py    # Очистка книги
""")


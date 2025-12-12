"""
Очистка текста книги Гарри Поттер
=================================
Использует функции из 02_text_cleaning.py
"""

import re
import unicodedata
from pathlib import Path


# ============================================================
# Функции очистки (из 02_text_cleaning.py)
# ============================================================

def basic_clean(text: str) -> str:
    """Базовая очистка: пробелы, переносы строк."""
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Нормализует Unicode символы."""
    return unicodedata.normalize('NFKC', text)


def remove_special_chars(text: str) -> str:
    """Удаляет специальные символы, оставляя пунктуацию."""
    return re.sub(r'[^\w\s.,!?;:\-—–\'\"«»()\[\]]', '', text)


def analyze_text_quality(text: str) -> dict:
    """Анализирует качество текста."""
    lines = text.split('\n')
    words = text.split()
    
    return {
        'total_chars': len(text),
        'total_words': len(words),
        'total_lines': len(lines),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'empty_lines_ratio': sum(1 for l in lines if not l.strip()) / len(lines) if lines else 0,
    }


# ============================================================
# Основной код очистки
# ============================================================

# Путь к файлу
input_file = Path("data/Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt")
output_file = Path("data/harry_potter_cleaned.txt")

# Загружаем текст
print("📖 Загрузка файла...")
text = input_file.read_text(encoding='utf-8')

# Анализ ДО очистки
print("\n📊 АНАЛИЗ ДО ОЧИСТКИ:")
quality_before = analyze_text_quality(text)
for k, v in quality_before.items():
    print(f"   {k}: {v:.2f}" if isinstance(v, float) else f"   {k}: {v:,}")

# ============================================================
# ПАЙПЛАЙН ОЧИСТКИ
# ============================================================
print("\n🧹 Выполняем очистку...")

# Шаг 1: Нормализация Unicode
text = normalize_unicode(text)
print("   ✓ Нормализация Unicode")

# Шаг 2: Удаление специальных символов
text = remove_special_chars(text)
print("   ✓ Удаление спецсимволов")

# Шаг 3: Базовая очистка (пробелы, переносы)
text = basic_clean(text)
print("   ✓ Очистка пробелов и переносов")

# Анализ ПОСЛЕ очистки
print("\n📊 АНАЛИЗ ПОСЛЕ ОЧИСТКИ:")
quality_after = analyze_text_quality(text)
for k, v in quality_after.items():
    print(f"   {k}: {v:.2f}" if isinstance(v, float) else f"   {k}: {v:,}")

# Сравнение
chars_removed = quality_before['total_chars'] - quality_after['total_chars']
percent_removed = (chars_removed / quality_before['total_chars']) * 100
print(f"\n🔍 ИТОГО УДАЛЕНО: {chars_removed:,} символов ({percent_removed:.1f}%)")

# ============================================================
# Сохранение результата
# ============================================================
output_file.write_text(text, encoding='utf-8')
print(f"\n✅ Очищенный текст сохранён в: {output_file}")

# Показываем пример
print("\n" + "="*60)
print("📝 ПРИМЕР ОЧИЩЕННОГО ТЕКСТА (первые 500 символов):")
print("="*60)
print(text[:500])


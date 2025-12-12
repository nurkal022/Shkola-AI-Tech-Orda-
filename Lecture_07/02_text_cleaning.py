"""
Лекция 7: Очистка и предобработка текста
========================================
Качество RAG напрямую зависит от качества данных.
Грязные данные = плохой поиск = плохие ответы.
"""

import re
from pathlib import Path

# Загрузим тестовый текст
text = Path("data/Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt").read_text(encoding='utf-8')
sample = text[:10000]  # Небольшой кусок для демонстрации


# ============================================================
# 1. Базовая очистка
# ============================================================
print("="*60)
print("1️⃣ Базовая очистка текста")
print("="*60)

def basic_clean(text: str) -> str:
    """Базовая очистка: пробелы, переносы строк."""     
    # Убираем множественные пробелы
    text = re.sub(r' +', ' ', text)
    # Убираем множественные переносы строк
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Убираем пробелы в начале/конце строк
    text = '\n'.join(line.strip() for line in text.split('\n'))
    # Убираем пробелы в начале и конце
    text = text.strip()
    return text

cleaned = basic_clean(sample)
print(f"   До: {len(sample)} символов")
print(f"   После: {len(cleaned)} символов")
print(f"   Удалено: {len(sample) - len(cleaned)} символов")


# ============================================================
# 2. Удаление специальных символов
# ============================================================
print("\n" + "="*60)
print("2️⃣ Удаление специальных символов")
print("="*60)

def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    """Удаляет специальные символы."""
    if keep_punctuation:
        # Оставляем буквы, цифры, базовую пунктуацию
        text = re.sub(r'[^\w\s.,!?;:\-—–\'\"«»()\[\]]', '', text)
    else:
        # Только буквы, цифры, пробелы
        text = re.sub(r'[^\w\s]', '', text)
    return text

# Пример с "грязным" текстом
dirty = "Привет! ™ Это текст © с разными ® символами § и ¶ мусором..."
clean = remove_special_chars(cleaned)
print(f"   После: {len(clean)} символов")

# ============================================================
# 3. Нормализация Unicode
# ============================================================
print("\n" + "="*60)
print("3️⃣ Нормализация Unicode")
print("="*60)

import unicodedata

def normalize_unicode(text: str) -> str:
    """Нормализует Unicode символы."""
    # NFKC - совместимость и композиция
    text = unicodedata.normalize('NFKC', text)
    return text

# Пример
weird = "Ｈｅｌｌｏ ½ ﬁ"  # Полноширинные символы
normal = normalize_unicode(weird)
print(f"   До:    '{weird}'")
print(f"   После: '{normal}'")


# ============================================================
# 4. Удаление HTML/XML тегов
# ============================================================
print("\n" + "="*60)
print("4️⃣ Удаление HTML/XML тегов")
print("="*60)

def remove_html_tags(text: str) -> str:
    """Удаляет HTML теги из текста."""
    # Удаляем теги
    text = re.sub(r'<[^>]+>', '', text)
    # Декодируем HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    return text

html_text = "<p>Привет, <b>мир</b>!</p>&nbsp;Как&amp;дела?"
clean = remove_html_tags(html_text)
print(f"   До:    '{html_text}'")
print(f"   После: '{clean}'")

# Более мощный вариант с BeautifulSoup
print("\n   💡 Для сложного HTML используйте BeautifulSoup:")
print("""
   from bs4 import BeautifulSoup
   
   def extract_text_from_html(html: str) -> str:
       soup = BeautifulSoup(html, 'html.parser')
       # Удаляем script и style
       for tag in soup(['script', 'style', 'nav', 'footer']):
           tag.decompose()
       return soup.get_text(separator=' ', strip=True)
""")


# ============================================================
# 5. Очистка от URL и email
# ============================================================
print("\n" + "="*60)
print("5️⃣ Удаление URL и email")
print("="*60)

def remove_urls_emails(text: str, replace_with: str = '') -> str:
    """Удаляет или заменяет URL и email."""
    # URL
    text = re.sub(r'https?://\S+|www\.\S+', replace_with, text)
    # Email
    text = re.sub(r'\S+@\S+\.\S+', replace_with, text)
    return text

text_with_urls = "Перейдите на https://example.com или напишите mail@test.com"
clean = remove_urls_emails(text_with_urls, replace_with='[ССЫЛКА]')
print(f"   До:    '{text_with_urls}'")
print(f"   После: '{clean}'")


# ============================================================
# 6. Удаление стоп-слов (опционально для некоторых задач)
# ============================================================
print("\n" + "="*60)
print("6️⃣ Удаление стоп-слов")
print("="*60)

# Базовый список русских стоп-слов
RUSSIAN_STOP_WORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все',
    'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по',
    'только', 'её', 'мне', 'было', 'вот', 'от', 'меня', 'ещё', 'нет', 'о', 'из', 'ему',
    'теперь', 'когда', 'уже', 'вам', 'ни', 'быть', 'был', 'ли', 'при', 'для', 'до'
}

def remove_stop_words(text: str, stop_words: set = RUSSIAN_STOP_WORDS) -> str:
    """Удаляет стоп-слова (осторожно - может изменить смысл!)."""
    words = text.split()
    filtered = [w for w in words if w.lower() not in stop_words]
    return ' '.join(filtered)

text = "Он был в доме и смотрел на меня"
cleaned = remove_stop_words(cleaned)
print(f"   До:    '{len(cleaned)}'")
print(f"   После: '{len(cleaned)}'")


# ============================================================
# 7. Комплексный пайплайн очистки
# ============================================================
print("\n" + "="*60)
print("7️⃣ Комплексный пайплайн очистки")
print("="*60)

class TextCleaner:
    """Конфигурируемый пайплайн очистки текста."""
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_extra_whitespace: bool = True,
        lowercase: bool = False,
        min_line_length: int = 0,
    ):
        self.normalize_unicode = normalize_unicode
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_extra_whitespace = remove_extra_whitespace
        self.lowercase = lowercase
        self.min_line_length = min_line_length
    
    def clean(self, text: str) -> str:
        """Применяет все этапы очистки."""
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        if self.remove_emails:
            text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        if self.remove_extra_whitespace:
            text = re.sub(r' +', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = '\n'.join(line.strip() for line in text.split('\n'))
        
        if self.lowercase:
            text = text.lower()
        
        if self.min_line_length > 0:
            lines = text.split('\n')
            lines = [l for l in lines if len(l.strip()) >= self.min_line_length]
            text = '\n'.join(lines)
        
        return text.strip()

# Использование
cleaner = TextCleaner(
    normalize_unicode=True,
    remove_html=True,
    remove_urls=True,
    min_line_length=10,  # Удалить короткие строки
)

dirty_text = """
<p>Привет!</p>
ok
Это нормальный текст с https://example.com ссылкой.
a
Ещё один абзац.
"""

clean = cleaner.clean(dirty_text)
print(f"   До ({len(dirty_text)} симв.):\n{dirty_text}")
print(f"   После ({len(clean)} симв.):\n{clean}")


# ============================================================
# 8. Очистка специфичная для формата
# ============================================================
print("\n" + "="*60)
print("8️⃣ Очистка для разных форматов")
print("="*60)

# PDF специфичная очистка
def clean_pdf_text(text: str) -> str:
    """Очистка текста извлечённого из PDF."""
    # Убираем номера страниц
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Убираем разрыв слов при переносе
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Восстанавливаем параграфы
    text = re.sub(r'(?<=[.!?])\n(?=[A-ZА-ЯЁ])', '\n\n', text)
    return text

# OCR специфичная очистка
def clean_ocr_text(text: str) -> str:
    """Очистка текста после OCR."""
    # Частые ошибки OCR
    replacements = {
        '|': 'l',  # Вертикальная черта → l
        '0': 'O',  # Может быть как ноль, так и O (контекстно)
        'rn': 'm',  # rn часто OCR'ится вместо m
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Убираем "мусорные" символы
    text = re.sub(r'[^\w\s.,!?;:\-—–\'"«»()\[\]]+', '', text)
    return text

print("""
   clean_pdf_text(text) - для PDF:
   - Убирает номера страниц
   - Склеивает переносы слов
   
   clean_ocr_text(text) - для OCR:
   - Исправляет частые ошибки распознавания
   - Убирает мусорные символы
""")


# ============================================================
# 9. Валидация качества текста
# ============================================================
print("\n" + "="*60)
print("9️⃣ Валидация качества текста")
print("="*60)

def analyze_text_quality(text: str) -> dict:
    """Анализирует качество текста."""
    lines = text.split('\n')
    words = text.split()
    
    return {
        'total_chars': len(text),
        'total_words': len(words),
        'total_lines': len(lines),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'avg_line_length': sum(len(l) for l in lines) / len(lines) if lines else 0,
        'empty_lines_ratio': sum(1 for l in lines if not l.strip()) / len(lines) if lines else 0,
        'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0,
    }

quality = analyze_text_quality(sample)
print("   Метрики качества тестового текста:")
for k, v in quality.items():
    print(f"   - {k}: {v:.2f}" if isinstance(v, float) else f"   - {k}: {v}")


# ============================================================
# РЕКОМЕНДАЦИИ
# ============================================================
print("\n" + "="*60)
print("💡 РЕКОМЕНДАЦИИ ПО ОЧИСТКЕ")
print("="*60)
print("""
   1. Минимальная очистка для RAG:
      - Нормализация Unicode
      - Удаление лишних пробелов/переносов
      - НЕ удаляйте стоп-слова
      - НЕ делайте lowercase (теряется информация)
   
   2. Для поиска важно сохранить:
      - Оригинальный регистр
      - Пунктуацию (несёт смысл)
      - Числа и даты
   
   3. Удаляйте только явный мусор:
      - HTML теги
      - Номера страниц
      - Повторяющиеся разделители
   
   4. Специфичная очистка по формату:
      - PDF: переносы слов, номера страниц
      - HTML: теги, скрипты, стили
      - OCR: исправление ошибок распознавания
   
   5. Всегда проверяйте результат очистки вручную!
""")


"""
Лекция 7: Парсеры документов
============================
Разные типы документов требуют разных парсеров.
Демонстрация основных парсеров LangChain.
"""

from pathlib import Path

# ============================================================
# 1. TextLoader - Обычные текстовые файлы
# ============================================================
print("="*60)
print("1️⃣ TextLoader - Текстовые файлы (.txt)")
print("="*60)

from langchain_community.document_loaders import TextLoader

# Загрузка одного файла
loader = TextLoader("data/sample.txt", encoding="utf-8")
docs = loader.load()
print(f"   Загружено документов: {len(docs)}")
print(f"   Символов: {len(docs[0].page_content):,}")
print(f"   Метаданные: {docs[0].metadata}")
print(f"   Содержимое (первые 100 символов):")
print(f"   '{docs[0].page_content[:100]}...'")


# ============================================================
# 2. DirectoryLoader - Загрузка папки
# ============================================================
print("\n" + "="*60)
print("2️⃣ DirectoryLoader - Все файлы из папки")
print("="*60)

from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "data/",
    glob="*.txt",  # Только .txt файлы
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True,
)
docs = loader.load()
print(f"   Загружено документов: {len(docs)}")
print(f"   Всего символов: {sum(len(d.page_content) for d in docs):,}")


# ============================================================
# 3. PyPDFLoader - PDF файлы
# ============================================================
print("\n" + "="*60)
print("3️⃣ PyPDFLoader - PDF документы")
print("="*60)

print("""
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
pages = loader.load()  # Каждая страница - отдельный Document

# Или загрузить и сразу разбить:
pages = loader.load_and_split()

Особенности:
- Каждая страница = отдельный Document
- Метаданные содержат номер страницы
- Требует: pip install pypdf
""")


# ============================================================
# 4. Docx2txtLoader - Word документы
# ============================================================
print("\n" + "="*60)
print("4️⃣ Docx2txtLoader - Word документы (.docx)")
print("="*60)

print("""
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("document.docx")
docs = loader.load()

Особенности:
- Извлекает только текст (без форматирования)
- Требует: pip install docx2txt
""")


# ============================================================
# 5. UnstructuredHTMLLoader - HTML страницы
# ============================================================
print("\n" + "="*60)
print("5️⃣ UnstructuredHTMLLoader - HTML файлы")
print("="*60)

# ДЕМОНСТРАЦИЯ: HTML
print("   Демонстрация UnstructuredHTMLLoader:")

try:
    from langchain_community.document_loaders import UnstructuredHTMLLoader

    loader = UnstructuredHTMLLoader("data/sample.html")
    docs = loader.load()
    print(f"   ✅ Загружено: {len(docs)} документ")
    print(f"   📄 Содержимое (HTML теги удалены):")
    print(f"   '{docs[0].page_content[:150]}...'")

except ImportError:
    print("   ⚠️ UnstructuredHTMLLoader не установлен (pip install unstructured)")

print("""
Особенности:
- Убирает HTML теги
- Сохраняет структуру текста
- Требует: pip install unstructured
""")


# ============================================================
# 6. UnstructuredMarkdownLoader - Markdown
# ============================================================
print("\n" + "="*60)
print("6️⃣ UnstructuredMarkdownLoader - Markdown файлы")
print("="*60)

# ДЕМОНСТРАЦИЯ: Markdown
print("   Демонстрация UnstructuredMarkdownLoader:")

try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader

    loader = UnstructuredMarkdownLoader("data/sample.md", mode="single")
    docs = loader.load()
    print(f"   ✅ Загружено: {len(docs)} документ")
    print(f"   📄 Содержимое (первые 150 символов):")
    print(f"   '{docs[0].page_content[:150]}...'")

    # Режим elements для структурного разбиения
    loader_elements = UnstructuredMarkdownLoader("data/sample.md", mode="elements")
    docs_elements = loader_elements.load()
    print(f"   📄 В режиме elements: {len(docs_elements)} элементов")

except ImportError:
    print("   ⚠️ UnstructuredMarkdownLoader не установлен (pip install unstructured)")

print("""
Особенности:
- Распознает заголовки, списки, код
- Режим elements для структурного разбиения
""")


# ============================================================
# 7. CSVLoader - CSV файлы
# ============================================================
print("\n" + "="*60)
print("7️⃣ CSVLoader - CSV/табличные данные")
print("="*60)

# ДЕМОНСТРАЦИЯ: CSV
print("   Демонстрация CSVLoader:")

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("data/sample.csv")
docs = loader.load()
print(f"   ✅ Загружено: {len(docs)} строк (документов)")
print(f"   📄 Первая строка:")
print(f"      Content: {docs[0].page_content}")
print(f"      Metadata: {docs[0].metadata}")
print(f"   📄 Вторая строка:")
print(f"      Content: {docs[1].page_content}")

print("""
Особенности:
- Каждая строка CSV = отдельный Document
- Все колонки попадают в метаданные
""")


# ============================================================
# 8. JSONLoader - JSON файлы
# ============================================================
print("\n" + "="*60)
print("8️⃣ JSONLoader - JSON данные")
print("="*60)

# ДЕМОНСТРАЦИЯ: JSON
print("   Демонстрация JSONLoader:")

try:
    from langchain_community.document_loaders import JSONLoader

    # Извлечение отдельных сообщений
    loader = JSONLoader(
        "data/sample.json",
        jq_schema=".messages[]",  # Каждый элемент массива messages
        text_content=False,
    )
    docs = loader.load()
    print(f"   ✅ Загружено: {len(docs)} сообщений")
    print(f"   📄 Первое сообщение:")
    print(f"      Content: {docs[0].page_content}")
    print(f"      Metadata: {docs[0].metadata}")

except ImportError:
    print("   ⚠️ JSONLoader не установлен (pip install jq)")

print("""
Особенности:
- Использует jq синтаксис для навигации
- Требует: pip install jq
""")


# ============================================================
# 9. UnstructuredExcelLoader - Excel файлы
# ============================================================
print("\n" + "="*60)
print("9️⃣ UnstructuredExcelLoader - Excel файлы")
print("="*60)

print("""
from langchain_community.document_loaders import UnstructuredExcelLoader

loader = UnstructuredExcelLoader("data.xlsx", mode="elements")
docs = loader.load()

Особенности:
- Поддерживает .xlsx и .xls
- Режим elements разбивает по таблицам/листам
- Требует: pip install unstructured openpyxl
""")


# ============================================================
# 10. UnstructuredPowerPointLoader - PowerPoint
# ============================================================
print("\n" + "="*60)
print("🔟 UnstructuredPowerPointLoader - Презентации")
print("="*60)

print("""
from langchain_community.document_loaders import UnstructuredPowerPointLoader

loader = UnstructuredPowerPointLoader("presentation.pptx")
docs = loader.load()

Особенности:
- Извлекает текст со всех слайдов
- Режим elements для разбиения по слайдам
- Требует: pip install unstructured python-pptx
""")


# ============================================================
# ТАБЛИЦА СРАВНЕНИЯ
# ============================================================
print("\n" + "="*60)
print("📊 ТАБЛИЦА ПАРСЕРОВ")
print("="*60)
print("""
┌─────────────────────┬────────────────────┬──────────────────────────────┐
│ Формат              │ Loader             │ pip install                  │
├─────────────────────┼────────────────────┼──────────────────────────────┤
│ .txt                │ TextLoader         │ (встроен)                    │
│ .pdf                │ PyPDFLoader        │ pypdf                        │
│ .docx               │ Docx2txtLoader     │ docx2txt                     │
│ .html               │ UnstructuredHTML   │ unstructured                 │
│ .md                 │ UnstructuredMD     │ unstructured                 │
│ .csv                │ CSVLoader          │ (встроен)                    │
│ .json               │ JSONLoader         │ jq                           │
│ .xlsx               │ UnstructuredExcel  │ unstructured openpyxl        │
│ .pptx               │ UnstructuredPPT    │ unstructured python-pptx     │
│ URL                 │ WebBaseLoader      │ beautifulsoup4               │
│ YouTube             │ YoutubeLoader      │ youtube-transcript-api       │
│ Notion              │ NotionDBLoader     │ notion-client                │
└─────────────────────┴────────────────────┴──────────────────────────────┘
""")


# ============================================================
# УНИВЕРСАЛЬНЫЙ ПАРСЕР
# ============================================================
print("\n" + "="*60)
print("🔧 UnstructuredFileLoader - Универсальный парсер")
print("="*60)

print("""
from langchain_community.document_loaders import UnstructuredFileLoader

# Автоматически определяет тип файла
loader = UnstructuredFileLoader("any_file.pdf")  # или .docx, .html, .md...
docs = loader.load()

# Требует: pip install unstructured[all-docs]

Плюсы:
+ Один парсер для всех форматов
+ Автоопределение типа файла

Минусы:
- Большая зависимость (много библиотек)
- Менее точный для специфичных форматов
""")


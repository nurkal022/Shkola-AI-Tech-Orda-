"""
🇰🇿 LIVE ДЕМОНСТРАЦИЯ: RAG с Supabase и Конституцией РК
===========================================================

Этот файл содержит все этапы работы с RAG в одном месте.
Используйте для демонстрации студентам в реальном времени.

Порядок демонстрации:
1. Подключение к Supabase
2. Загрузка и чанкинг документа
3. Создание эмбеддингов
4. Загрузка в базу данных
5. Векторный поиск
6. Полный RAG пайплайн

💡 СОВЕТ: Если данные уже загружены, закомментируйте этапы 2-4
   и переходите сразу к этапам 5-6 (поиск и RAG).
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# ЭТАП 1: ПОДКЛЮЧЕНИЕ К СЕРВИСАМ
# ============================================================
print("="*60)
print("ЭТАП 1: ПОДКЛЮЧЕНИЕ К СЕРВИСАМ")
print("="*60)

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Ошибка: SUPABASE_URL и SUPABASE_SERVICE_ROLE_KEY должны быть в .env")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print(f"✅ Подключено к Supabase: {SUPABASE_URL[:30]}...")

# OpenAI
openai_client = OpenAI()
print("✅ Подключено к OpenAI API")

print()


# ============================================================
# ЭТАП 2: ЗАГРУЗКА И ЧАНКИНГ ДОКУМЕНТА
# ============================================================
# 💡 ЗАКОММЕНТИРУЙТЕ ЭТОТ БЛОК, ЕСЛИ ДАННЫЕ УЖЕ ЗАГРУЖЕНЫ
# ============================================================
print("="*60)
print("ЭТАП 2: ЗАГРУЗКА И ЧАНКИНГ ДОКУМЕНТА")
print("="*60)

# Читаем документ
document_path = "data/конституция.txt"
print(f"📄 Читаем документ: {document_path}")

with open(document_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"   Размер документа: {len(text):,} символов")

# Рекурсивный чанкинг
print("\n🔪 Разбиваем на чанки (рекурсивный метод)...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Размер чанка
    chunk_overlap=150,     # Перекрытие
    length_function=len,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Порядок важен!
)

chunks = splitter.split_text(text)
print(f"   ✅ Создано {len(chunks)} чанков")
print(f"   Средний размер чанка: {sum(len(c) for c in chunks) // len(chunks)} символов")

# Показываем пример первого чанка
print(f"\n   📝 Пример первого чанка ({len(chunks[0])} символов):")
print(f"   {chunks[0][:200]}...")

print()


# ============================================================
# ЭТАП 3: СОЗДАНИЕ ЭМБЕДДИНГОВ
# ============================================================
# 💡 ЗАКОММЕНТИРУЙТЕ ЭТОТ БЛОК, ЕСЛИ ДАННЫЕ УЖЕ ЗАГРУЖЕНЫ
# ============================================================
print("="*60)
print("ЭТАП 3: СОЗДАНИЕ ЭМБЕДДИНГОВ")
print("="*60)

print("🔮 Создаём эмбеддинги через OpenAI...")
print("   (это может занять время...)")

def get_embeddings(texts: list[str], batch_size: int = 100):
    """Создать эмбеддинги батчами"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"   Обработано: {min(i+batch_size, len(texts))}/{len(texts)}")
    return all_embeddings

embeddings = get_embeddings(chunks, batch_size=100)
print(f"   ✅ Создано {len(embeddings)} эмбеддингов")
print(f"   Размер вектора: {len(embeddings[0])} чисел")

print()


# ============================================================
# ЭТАП 4: ЗАГРУЗКА В SUPABASE
# ============================================================
# 💡 ЗАКОММЕНТИРУЙТЕ ЭТОТ БЛОК, ЕСЛИ ДАННЫЕ УЖЕ ЗАГРУЖЕНЫ
# ============================================================
print("="*60)
print("ЭТАП 4: ЗАГРУЗКА В SUPABASE")
print("="*60)

# Очищаем старые данные (опционально)
print("🗑️  Очищаем старые данные...")
try:
    supabase.table("documents").delete().neq("id", 0).execute()
    print("   ✅ Старые данные удалены")
except Exception as e:
    print(f"   ⚠️  {e}")

# Загружаем в базу батчами
print("\n📤 Загружаем чанки в Supabase...")

batch_size = 20
total_loaded = 0
book_name = os.path.basename(document_path)

for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size]
    
    records = []
    for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
        records.append({
            "content": chunk,
            "book": book_name,
            "chunk_id": i + j,
            "embedding": emb
        })
    
    try:
        supabase.table("documents").insert(records).execute()
        total_loaded += len(records)
        print(f"   Загружено: {total_loaded}/{len(chunks)}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        break

print(f"\n   ✅ Загружено {total_loaded} документов в Supabase!")

# Проверяем количество
try:
    count = supabase.table("documents").select("id", count="exact").execute()
    print(f"   📊 Всего в БД: {count.count if hasattr(count, 'count') else 'N/A'}")
except:
    pass

print()


# ============================================================
# ЭТАП 5: ВЕКТОРНЫЙ ПОИСК
# ============================================================
print("="*60)
print("ЭТАП 5: ВЕКТОРНЫЙ ПОИСК")
print("="*60)

def search_documents(query: str, top_k: int = 3):
    """
    Поиск похожих документов по запросу
    
    Args:
        query: Текстовый запрос
        top_k: Количество результатов
    """
    print(f"\n🔍 Запрос: '{query}'")
    
    # 1. Создаём эмбеддинг запроса
    print("   1️⃣  Создаём эмбеддинг запроса...")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    print(f"      Размер вектора: {len(query_embedding)}")
    
    # 2. Ищем в базе через RPC функцию
    print("   2️⃣  Ищем похожие документы в Supabase...")
    try:
        result = supabase.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_count': top_k,
            'filter_book': None
        }).execute()
        
        if not result.data:
            print("      ❌ Ничего не найдено")
            return []
        
        print(f"      ✅ Найдено {len(result.data)} результатов:\n")
        
        # 3. Показываем результаты
        for i, doc in enumerate(result.data, 1):
            similarity = doc['similarity']
            content = doc['content']
            
            # Индикация релевантности
            if similarity > 0.6:
                icon = "🟢"
            elif similarity > 0.5:
                icon = "🟡"
            else:
                icon = "🔴"
            
            print(f"      {icon} Результат {i} | Similarity: {similarity:.3f}")
            print(f"         {content[:150]}...")
            print()
        
        return result.data
        
    except Exception as e:
        print(f"      ❌ Ошибка поиска: {e}")
        print("      💡 Убедитесь что функция match_documents создана в Supabase")
        return []

# Демонстрация поиска
demo_queries = [
    "Какие права имеет гражданин?",
    "Какой язык является государственным?",
]

for query in demo_queries:
    search_documents(query, top_k=3)
    print()

print()


# ============================================================
# ЭТАП 6: ПОЛНЫЙ RAG ПАЙПЛАЙН
# ============================================================
print("="*60)
print("ЭТАП 6: ПОЛНЫЙ RAG ПАЙПЛАЙН")
print("="*60)

def rag_answer(question: str, top_k: int = 3):
    """
    Полный RAG пайплайн: Retrieval → Augmentation → Generation
    
    Args:
        question: Вопрос пользователя
        top_k: Количество релевантных чанков
    """
    print(f"\n❓ ВОПРОС: {question}")
    print("-" * 60)
    
    # ШАГ 1: RETRIEVAL - Извлечение контекста
    print("\n1️⃣  RETRIEVAL (Извлечение контекста)...")
    context_docs = search_documents(question, top_k=top_k)
    
    if not context_docs:
        return "Не удалось найти релевантную информацию."
    
    # Формируем контекст
    context_text = "\n\n---\n\n".join([
        f"[{doc['book']}]\n{doc['content']}"
        for doc in context_docs
    ])
    
    # ШАГ 2: AUGMENTATION - Обогащение промпта
    print("\n2️⃣  AUGMENTATION (Обогащение промпта)...")
    prompt = f"""Ответь на вопрос, используя ТОЛЬКО предоставленный контекст из Конституции Республики Казахстан.

КОНТЕКСТ:
{context_text}

ВОПРОС: {question}

ОТВЕТ (кратко, 2-3 предложения):"""
    
    print("   ✅ Промпт сформирован с контекстом")
    
    # ШАГ 3: GENERATION - Генерация ответа
    print("\n3️⃣  GENERATION (Генерация ответа через GPT-4o-mini)...")
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Ты эксперт по Конституции Республики Казахстан. Отвечай только на основе предоставленного контекста."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )
    
    answer = response.choices[0].message.content
    
    print("\n💬 ОТВЕТ:")
    print(f"   {answer}")
    
    print("\n📚 ИСТОЧНИКИ:")
    for i, doc in enumerate(context_docs, 1):
        print(f"   {i}. [{doc['book']}] similarity: {doc['similarity']:.3f}")
    
    return answer

# Демонстрация RAG
print("\n" + "="*60)
print("ДЕМОНСТРАЦИЯ RAG ПАЙПЛАЙНА")
print("="*60)

demo_questions = [
    "Какие права и свободы гарантирует Конституция гражданам?",
    "Какой язык является государственным в Казахстане?",
    "Из каких палат состоит Парламент?",
]

for question in demo_questions:
    rag_answer(question, top_k=3)
    print("\n" + "="*60 + "\n")

print()
print("="*60)
print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
print("="*60)

# ============================================================
# БОНУС: ИНТЕРАКТИВНЫЙ РЕЖИМ
# ============================================================
print("\n" + "="*60)
print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
print("="*60)
print("💡 Введите свой вопрос или 'выход' для завершения\n")

while True:
    question = input("❓ Ваш вопрос: ").strip()
    if question.lower() in ['выход', 'exit', 'q', 'quit']:
        print("\n👋 До свидания!")
        break
    if question:
        print()
        rag_answer(question, top_k=3)
        print("\n" + "="*60 + "\n")


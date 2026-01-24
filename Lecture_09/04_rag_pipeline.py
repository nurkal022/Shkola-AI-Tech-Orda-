"""
Пример 4: Полный RAG пайплайн с Supabase

RAG = Retrieval-Augmented Generation
1. Получаем вопрос пользователя
2. Создаём эмбеддинг вопроса
3. Ищем релевантные чанки в Supabase
4. Формируем промпт с контекстом
5. Отправляем в LLM → получаем ответ
"""

from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

openai_client = OpenAI()

print("=== RAG пайплайн с Supabase ===\n")


def get_embedding(text: str) -> list[float]:
    """Получить эмбеддинг текста"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def retrieve_context(query: str, top_k: int = 3) -> list[dict]:
    """
    Извлечение релевантного контекста из Supabase
    
    Returns:
        List[dict] с ключами: content, book, similarity
    """
    query_embedding = get_embedding(query)
    
    # Используем RPC функцию (должна быть создана в SQL)
    try:
        result = supabase.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_count': top_k,
            'filter_book': None
        }).execute()
        
        return result.data
    except Exception as e:
        print(f"⚠️  Ошибка при поиске: {e}")
        print("   Убедитесь что функция match_documents создана (см. 03_vector_search.py)")
        return []


def generate_answer(question: str, context: list[dict]) -> str:
    """
    Генерация ответа на основе контекста
    
    Args:
        question: Вопрос пользователя
        context: Релевантные чанки из БД
    """
    # Формируем контекст из найденных чанков
    context_text = "\n\n".join([
        f"[Из книги: {chunk['book']}]\n{chunk['content']}"
        for chunk in context
    ])
    
    # Промпт для LLM
    prompt = f"""Ответь на вопрос, используя только предоставленный контекст из Конституции Республики Казахстан.

Контекст:
{context_text}

Вопрос: {question}

Ответ (кратко, 2-3 предложения):"""

    # Отправляем в LLM
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты помощник, отвечающий на вопросы по Конституции Республики Казахстан."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content


def rag_pipeline(question: str, top_k: int = 3):
    """
    Полный RAG пайплайн
    
    Returns:
        dict с answer и sources
    """
    print(f"❓ Вопрос: {question}\n")
    
    # Шаг 1: Retrieval - извлекаем контекст
    print("1️⃣  Извлечение контекста...")
    context = retrieve_context(question, top_k=top_k)
    
    if not context:
        return {
            "answer": "Не удалось найти релевантную информацию в базе данных.",
            "sources": []
        }
    
    print(f"   Найдено {len(context)} релевантных чанков:")
    for i, chunk in enumerate(context, 1):
        preview = chunk['content'][:60] + "..." if len(chunk['content']) > 60 else chunk['content']
        print(f"   [{i}] [{chunk['similarity']:.3f}] {preview}")
    
    # Шаг 2: Augmentation - формируем промпт с контекстом
    print("\n2️⃣  Генерация ответа...")
    
    # Шаг 3: Generation - генерируем ответ
    answer = generate_answer(question, context)
    
    return {
        "answer": answer,
        "sources": [
            {
                "book": chunk['book'],
                "content": chunk['content'][:200] + "...",
                "similarity": chunk['similarity']
            }
            for chunk in context
        ]
    }


# =============================================
# Демонстрация RAG
# =============================================
questions = [
    "Какие права и свободы гарантирует Конституция гражданам?",
    "Каковы полномочия Президента Республики Казахстан?",
    "Как формируется Парламент?",
    "Что такое Конституционный Суд?",
]

for question in questions:
    print("="*60)
    result = rag_pipeline(question, top_k=3)
    
    print(f"\n💬 Ответ: {result['answer']}")
    print(f"\n📚 Источники:")
    for i, source in enumerate(result['sources'], 1):
        print(f"   {i}. [{source['book']}] similarity: {source['similarity']:.3f}")
    
    print("\n")


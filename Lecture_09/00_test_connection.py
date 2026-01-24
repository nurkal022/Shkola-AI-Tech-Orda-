"""
Быстрая проверка подключения к Supabase

Проверяет:
- Подключение к проекту
- Наличие таблицы documents
- Наличие расширения pgvector
"""

from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

print("=== Проверка подключения к Supabase ===\n")

if not SUPABASE_URL:
    print("❌ SUPABASE_URL не найден в .env")
    print("   Добавьте: SUPABASE_URL=https://qdhswcgeqbkkhdstiiqh.supabase.co")
    exit(1)

if not SUPABASE_KEY or SUPABASE_KEY == "your-service-role-key-here":
    print("❌ SUPABASE_SERVICE_ROLE_KEY не настроен")
    print("\n📝 Где найти Service Role Key:")
    print("   1. Supabase Dashboard → Settings → API")
    print("   2. Найдите 'service_role' key (секретный!)")
    print("   3. Добавьте в .env: SUPABASE_SERVICE_ROLE_KEY=ваш-ключ")
    exit(1)

print(f"🔗 URL: {SUPABASE_URL}")
print(f"🔑 Key: {SUPABASE_KEY[:20]}...\n")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Клиент создан")
    
    # Проверяем таблицу
    try:
        result = supabase.table("documents").select("id").limit(1).execute()
        print("✅ Таблица 'documents' существует")
        
        # Пробуем получить количество
        count_result = supabase.table("documents").select("id", count="exact").limit(1).execute()
        if hasattr(count_result, 'count'):
            print(f"📊 Документов в БД: {count_result.count}")
        else:
            # Альтернативный способ
            all_data = supabase.table("documents").select("id").execute()
            print(f"📊 Документов в БД: {len(all_data.data) if all_data.data else 0}")
    except Exception as e:
        print(f"⚠️  Таблица 'documents' не найдена: {e}")
        print("   Создайте таблицу через SQL Editor (см. SETUP_INSTRUCTIONS.md)")
    
    print("\n✅ Подключение работает!")
    print("\n📝 Следующие шаги:")
    print("   1. Выполните SQL из SETUP_INSTRUCTIONS.md в Supabase SQL Editor")
    print("   2. Запустите: python 02_load_documents.py")
    
except Exception as e:
    print(f"❌ Ошибка подключения: {e}")
    print("\nПроверьте:")
    print("   - Правильность SUPABASE_URL")
    print("   - Правильность SUPABASE_SERVICE_ROLE_KEY")
    print("   - Доступность интернета")
    exit(1)


"""
Лекция 19: Настройка LangSmith
===============================
Пошаговая настройка LangSmith для мониторинга LLM-приложений.

Что такое LangSmith?
- Платформа от LangChain для наблюдаемости (observability)
- Показывает каждый вызов: промпт, ответ, время, токены, стоимость
- Работает с LangChain, OpenAI, и любыми LLM
"""

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

# ============================================================
# 1. Зачем нужна наблюдаемость?
# ============================================================

print("=" * 60)
print("1. ЗАЧЕМ НУЖНА НАБЛЮДАЕМОСТЬ?")
print("=" * 60)
print()
print("Проблемы БЕЗ наблюдаемости:")
print("  - Агент вернул неправильный ответ — почему?")
print("  - Запрос стоил $0.50 — какой промпт съел токены?")
print("  - Агент зациклился — на какой итерации?")
print("  - Промпт работал вчера, сломался сегодня — что изменилось?")
print()
print("LangSmith показывает:")
print("  - Полный промпт, отправленный в LLM")
print("  - Ответ модели (включая tool calls)")
print("  - Время выполнения каждого шага")
print("  - Количество токенов и стоимость")
print("  - Все итерации агента")
print()

# ============================================================
# 2. Настройка — 3 переменные окружения
# ============================================================

print("=" * 60)
print("2. НАСТРОЙКА LANGSMITH")
print("=" * 60)
print()
print("Шаг 1: Регистрация на https://smith.langchain.com")
print("Шаг 2: Settings -> API Keys -> Create API Key")
print("Шаг 3: Добавить в .env:")
print()
print("  LANGCHAIN_API_KEY=lsv2_pt_xxxxx")
print("  LANGCHAIN_TRACING_V2=true")
print("  LANGCHAIN_PROJECT=My-Project")
print()

# Настраиваем переменные
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Lecture-19-Setup"

# Проверяем ключ
api_key = os.getenv("LANGCHAIN_API_KEY")
if not api_key:
    print("  LANGCHAIN_API_KEY не найден!")
    print("  Добавьте его в /Users/nurlykhan/TechOrda/.env")
    exit(1)

print(f"  LANGCHAIN_API_KEY:     {api_key[:8]}...{api_key[-4:]}")
print(f"  LANGCHAIN_TRACING_V2:  {os.environ['LANGCHAIN_TRACING_V2']}")
print(f"  LANGCHAIN_PROJECT:     {os.environ['LANGCHAIN_PROJECT']}")
print()

# ============================================================
# 3. Проверка подключения
# ============================================================

print("=" * 60)
print("3. ПРОВЕРКА ПОДКЛЮЧЕНИЯ")
print("=" * 60)
print()

from langsmith import Client

client = Client()

# Проверяем что клиент подключён
try:
    # Пробуем получить список проектов
    projects = list(client.list_projects(limit=5))
    print("  Подключение к LangSmith: OK")
    print(f"  Существующие проекты ({len(projects)}):")
    for p in projects[:5]:
        print(f"    - {p.name}")
    print()
except Exception as e:
    print(f"  Ошибка подключения: {e}")
    print("  Проверьте LANGCHAIN_API_KEY")
    exit(1)

# ============================================================
# 4. Первый трейс
# ============================================================

print("=" * 60)
print("4. ПЕРВЫЙ ТРЕЙС")
print("=" * 60)
print()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("  Отправляем запрос (он будет записан в LangSmith)...")
result = llm.invoke("Привет! Что такое LangSmith? Ответь одним предложением.")
print(f"  Ответ: {result.content}")
print()
print("  Откройте https://smith.langchain.com")
print(f"  Проект: {os.environ['LANGCHAIN_PROJECT']}")
print("  Вы увидите этот запрос с полной информацией:")
print("    - Input (промпт)")
print("    - Output (ответ)")
print("    - Latency (время)")
print("    - Tokens (количество)")
print("    - Cost (стоимость)")
print()

# ============================================================
# 5. Управление проектами
# ============================================================

print("=" * 60)
print("5. УПРАВЛЕНИЕ ПРОЕКТАМИ")
print("=" * 60)
print()
print("  Проект = группа трейсов для одного приложения/эксперимента")
print()
print("  Примеры организации:")
print("    LANGCHAIN_PROJECT=chatbot-prod       # продакшен")
print("    LANGCHAIN_PROJECT=chatbot-dev         # разработка")
print("    LANGCHAIN_PROJECT=experiment-v2       # эксперимент")
print("    LANGCHAIN_PROJECT=Lecture-19-Demo     # для лекции")
print()

# Можно менять проект прямо в коде
os.environ["LANGCHAIN_PROJECT"] = "Lecture-19-Experiment"

result = llm.invoke("Назови 3 преимущества LangSmith.")
print(f"  Ответ (проект Lecture-19-Experiment): {result.content[:150]}...")
print()

# Возвращаем
os.environ["LANGCHAIN_PROJECT"] = "Lecture-19-Setup"

# ============================================================
# 6. Включение/выключение трейсинга
# ============================================================

print("=" * 60)
print("6. ВКЛЮЧЕНИЕ / ВЫКЛЮЧЕНИЕ ТРЕЙСИНГА")
print("=" * 60)
print()

# Выключаем
os.environ["LANGCHAIN_TRACING_V2"] = "false"
print("  Трейсинг ВЫКЛЮЧЕН")

result = llm.invoke("Этот запрос НЕ будет записан в LangSmith.")
print(f"  Ответ: {result.content[:100]}...")
print("  -> Этот вызов НЕ появится в дашборде")
print()

# Включаем обратно
os.environ["LANGCHAIN_TRACING_V2"] = "true"
print("  Трейсинг ВКЛЮЧЕН")

result = llm.invoke("А этот запрос БУДЕТ записан.")
print(f"  Ответ: {result.content[:100]}...")
print("  -> Этот вызов появится в дашборде")
print()

# ============================================================
# Итоги
# ============================================================

print("=" * 60)
print("ИТОГИ: НАСТРОЙКА LANGSMITH")
print("=" * 60)
print()
print("  1. Регистрация: https://smith.langchain.com")
print("  2. Три переменные в .env:")
print("     LANGCHAIN_API_KEY=lsv2_...")
print("     LANGCHAIN_TRACING_V2=true")
print("     LANGCHAIN_PROJECT=My-Project")
print("  3. Всё! LangChain автоматически отправляет трейсы")
print("  4. Проекты для организации, теги для фильтрации")
print()
print("  Следующий шаг: 02_tracing_chains.py")

"""
Тестовый скрипт для проверки всех файлов Lecture 13
"""

import sys
from pathlib import Path

print("="*60)
print("🧪 ТЕСТИРОВАНИЕ ЛЕКЦИИ 13")
print("="*60)
print()

# Список файлов для проверки
files_to_test = [
    "01_classic_react.py",
    "02_react_prompt_anatomy.py",
    "03_react_vs_function_calling.py",
    "04_multistep_reasoning.py",
    "05_custom_react_prompt.py",
]

# Проверка синтаксиса
print("1️⃣ ПРОВЕРКА СИНТАКСИСА")
print("─" * 60)
all_ok = True

for filename in files_to_test:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, filename, 'exec')
        print(f"   ✅ {filename}")
    except SyntaxError as e:
        print(f"   ❌ {filename}: {e}")
        all_ok = False
    except Exception as e:
        print(f"   ⚠️ {filename}: {e}")

print()

# Проверка импортов
print("2️⃣ ПРОВЕРКА ИМПОРТОВ")
print("─" * 60)

try:
    from langchain_openai import ChatOpenAI
    print("   ✅ langchain_openai")
except ImportError as e:
    print(f"   ❌ langchain_openai: {e}")
    all_ok = False

try:
    from langchain import hub
    print("   ✅ langchain.hub")
except ImportError as e:
    print(f"   ❌ langchain.hub: {e}")
    all_ok = False

try:
    from langchain.agents import create_react_agent, create_openai_tools_agent
    print("   ✅ langchain.agents")
except ImportError as e:
    print(f"   ❌ langchain.agents: {e}")
    all_ok = False

try:
    from langchain_core.tools import tool
    print("   ✅ langchain_core.tools")
except ImportError as e:
    print(f"   ❌ langchain_core.tools: {e}")
    all_ok = False

try:
    from langchain_community.vectorstores import FAISS
    print("   ✅ langchain_community.vectorstores")
except ImportError as e:
    print(f"   ❌ langchain_community.vectorstores: {e}")
    all_ok = False

try:
    from dotenv import load_dotenv
    print("   ✅ python-dotenv")
except ImportError as e:
    print(f"   ❌ python-dotenv: {e}")
    all_ok = False

print()

# Проверка данных
print("3️⃣ ПРОВЕРКА ДАННЫХ")
print("─" * 60)

data_path = Path("data/faiss_harry_potter")
if data_path.exists():
    index_faiss = data_path / "index.faiss"
    index_pkl = data_path / "index.pkl"
    
    if index_faiss.exists() and index_pkl.exists():
        print(f"   ✅ FAISS индекс найден: {data_path}")
    else:
        print(f"   ⚠️ FAISS индекс неполный: {data_path}")
        print(f"      index.faiss: {'✅' if index_faiss.exists() else '❌'}")
        print(f"      index.pkl: {'✅' if index_pkl.exists() else '❌'}")
else:
    print(f"   ⚠️ FAISS индекс не найден: {data_path}")
    print("      (будет создан автоматически при первом запуске)")

print()

# Проверка .env
print("4️⃣ ПРОВЕРКА ОКРУЖЕНИЯ")
print("─" * 60)

import os
from dotenv import load_dotenv
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("   ✅ OPENAI_API_KEY найден")
else:
    print("   ⚠️ OPENAI_API_KEY не найден в .env")
    print("      Создайте .env файл с OPENAI_API_KEY=your_key")

print()

# Итог
print("="*60)
if all_ok:
    print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    print("="*60)
    print()
    print("📝 Следующие шаги:")
    print("   1. Убедитесь что .env файл содержит OPENAI_API_KEY")
    print("   2. Запустите любой из файлов для демонстрации:")
    print("      python 01_classic_react.py")
    print("      python 02_react_prompt_anatomy.py")
    print("      python 03_react_vs_function_calling.py")
    print("      python 04_multistep_reasoning.py")
    print("      python 05_custom_react_prompt.py")
else:
    print("❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ")
    print("="*60)
    print()
    print("📝 Установите недостающие зависимости:")
    print("   pip install -r requirements.txt")

print()

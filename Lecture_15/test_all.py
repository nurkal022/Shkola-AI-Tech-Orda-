"""
Тест всех файлов Lecture 15
"""

import sys
from pathlib import Path

print("="*60)
print("🧪 ТЕСТИРОВАНИЕ ЛЕКЦИИ 15")
print("="*60)
print()

# 1. Проверка импортов
print("1️⃣ Проверка импортов")
print("─"*60)
try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    print("   ✅ pyautogen импортирован")
except ImportError as e:
    print(f"   ❌ pyautogen не установлен: {e}")
    print("   Установите: pip install pyautogen")
print()

# 2. Проверка синтаксиса
print("2️⃣ Проверка синтаксиса")
print("─"*60)
files = [
    "01_basic_two_agents.py",
    "02_group_chat.py",
    "03_specialized_agents.py",
    "04_code_generation.py",
    "05_problem_solving.py",
    "00_live_demo_autogen.py",
    "00_live_demo_step_by_step.py",
]

for f in files:
    try:
        with open(f, 'r') as file:
            code = file.read()
        compile(code, f, 'exec')
        print(f"   ✅ {f}")
    except SyntaxError as e:
        print(f"   ❌ {f}: {e}")
    except FileNotFoundError:
        print(f"   ⚠️ {f}: файл не найден")
print()

# 3. Проверка API ключа
print("3️⃣ Проверка окружения")
print("─"*60)
import os
from dotenv import load_dotenv
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("   ✅ OPENAI_API_KEY найден")
else:
    print("   ⚠️ OPENAI_API_KEY не найден (файлы требуют API)")
print()

print("="*60)
print("✅ ТЕСТЫ ЗАВЕРШЕНЫ")
print("="*60)
print()
print("📝 Для запуска примеров:")
print("   python 01_basic_two_agents.py")
print("   python 02_group_chat.py")
print("   python 00_live_demo_autogen.py")

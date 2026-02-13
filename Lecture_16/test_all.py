"""
Тест всех файлов Lecture 16
"""

import sys
from pathlib import Path

print("="*60)
print("🧪 ТЕСТИРОВАНИЕ ЛЕКЦИИ 16")
print("="*60)
print()

# 1. Проверка импортов
print("1️⃣ Проверка импортов")
print("─"*60)
try:
    from crewai import Agent, Task, Crew, Process
    print("   ✅ crewai импортирован")
except ImportError as e:
    print(f"   ❌ crewai не установлен: {e}")
    print("   Установите: pip install crewai")
print()

# 2. Проверка синтаксиса
print("2️⃣ Проверка синтаксиса")
print("─"*60)
files = [
    "01_basic_crew.py",
    "02_role_based_agents.py",
    "03_sequential_tasks.py",
    "04_hierarchical_process.py",
    "05_research_crew.py",
    "06_code_review_crew.py",
    "00_live_demo_crewai.py",
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

# 3. Проверка создания агентов и задач
print("3️⃣ Проверка создания агентов и задач")
print("─"*60)
try:
    from crewai import Agent, Task, Crew, Process
    import os
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")
    
    agent = Agent(
        role="Test Agent",
        goal="Тестировать",
        backstory="Тестовый агент",
        verbose=False,
    )
    
    task = Task(
        description="Тестовая задача",
        expected_output="Тестовый вывод",
        agent=agent,
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    
    print("   ✅ Agent создан успешно")
    print("   ✅ Task создана успешно")
    print("   ✅ Crew создан успешно")
except Exception as e:
    print(f"   ⚠️ Ошибка создания: {str(e)[:60]}")
print()

# 4. Проверка API ключа
print("4️⃣ Проверка окружения")
print("─"*60)
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

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
print("   python 01_basic_crew.py")
print("   python 02_role_based_agents.py")
print("   python 00_live_demo_crewai.py")

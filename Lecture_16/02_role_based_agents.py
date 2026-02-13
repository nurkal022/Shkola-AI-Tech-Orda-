"""
Лекция 16: Ролевой дизайн агентов
==================================
Демонстрация важности role, goal, backstory в CrewAI.
Показывает как ролевой дизайн влияет на поведение агента.
"""

import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# Загружаем .env из корневой папки проекта
load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# 1. Создание агентов с разными ролями
# ============================================================

print("="*60)
print("👤 СОЗДАНИЕ АГЕНТОВ С РАЗНЫМИ РОЛЯМИ")
print("="*60)
print()

# Агент 1: Исследователь
researcher = Agent(
    role="Senior Research Analyst",
    goal="Найти актуальную и достоверную информацию",
    backstory="""Ты опытный аналитик с 15 лет опыта в исследованиях.
    Ты специализируешься на поиске и анализе информации из различных источников.
    Ты всегда проверяешь факты и представляешь информацию структурированно.""",
    verbose=True,
    allow_delegation=False,
)

# Агент 2: Писатель
writer = Agent(
    role="Creative Content Writer",
    goal="Создавать интересный и понятный контент",
    backstory="""Ты креативный писатель с талантом превращать сложные темы
    в понятные и увлекательные тексты. Ты пишешь для широкой аудитории,
    используя простой язык и примеры.""",
    verbose=True,
    allow_delegation=False,
)

print("✅ Researcher Agent создан:")
print(f"   Role: {researcher.role}")
print(f"   Goal: {researcher.goal}")
print()

print("✅ Writer Agent создан:")
print(f"   Role: {writer.role}")
print(f"   Goal: {writer.goal}")
print()

# ============================================================
# 2. Создание задач для каждого агента
# ============================================================

print("="*60)
print("📋 СОЗДАНИЕ ЗАДАЧ")
print("="*60)
print()

research_task = Task(
    description="Исследовать тему 'Квантовые вычисления' и найти 3 ключевых достижения 2024-2025 года",
    expected_output="Список из 3 ключевых достижений в квантовых вычислениях с кратким описанием каждого",
    agent=researcher,
)

writing_task = Task(
    description="Написать краткую статью (2-3 абзаца) о квантовых вычислениях для широкой аудитории",
    expected_output="Статья из 2-3 абзацев о квантовых вычислениях, написанная простым языком",
    agent=writer,
    context=[research_task],  # Использует результаты исследования
)

print("✅ Research Task создана")
print("✅ Writing Task создана (использует результаты Research Task)")
print()

# ============================================================
# 3. Создание Crew
# ============================================================

print("="*60)
print("🚣 СОЗДАНИЕ CREW")
print("="*60)
print()

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

print("✅ Crew создан:")
print(f"   Agents: {len(crew.agents)}")
print(f"   Tasks: {len(crew.tasks)}")
print(f"   Process: sequential")
print()

# ============================================================
# 4. Запуск
# ============================================================

print("="*60)
print("🚀 ЗАПУСК CREW")
print("="*60)
print()

result = crew.kickoff()

print()
print("="*60)
print("✅ РЕЗУЛЬТАТ:")
print("="*60)
print(result)
print()
print("💡 Обратите внимание:")
print("   • Researcher выполнил исследование")
print("   • Writer использовал результаты исследования")
print("   • Каждый агент действовал согласно своей роли")
print()

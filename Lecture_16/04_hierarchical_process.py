"""
Лекция 16: Иерархический процесс
=================================
Демонстрация Process.hierarchical - Crew сам выбирает агента для задачи.
Показывает автоматическое распределение задач на основе ролей.
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
# 1. Создание специализированных агентов
# ============================================================

print("="*60)
print("👥 СОЗДАНИЕ СПЕЦИАЛИЗИРОВАННЫХ АГЕНТОВ")
print("="*60)
print()

researcher = Agent(
    role="Research Specialist",
    goal="Проводить глубокие исследования по различным темам",
    backstory="""Ты специалист по исследованиям с экспертизой в поиске
    и анализе информации из различных источников.""",
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Content Writer",
    goal="Создавать качественный контент на основе исследований",
    backstory="""Ты профессиональный писатель, специализирующийся на
    создании образовательного и информационного контента.""",
    verbose=True,
    allow_delegation=False,
)

analyst = Agent(
    role="Data Analyst",
    goal="Анализировать данные и выявлять закономерности",
    backstory="""Ты опытный аналитик данных с навыками статистического
    анализа и визуализации данных.""",
    verbose=True,
    allow_delegation=False,
)

manager = Agent(
    role="Project Manager",
    goal="Эффективно распределять задачи между членами команды и контролировать ход выполнения проекта.",
    backstory="""Ты организованный project manager, обладающий навыками координации, делегирования задач,
    постановки целей и контроля хода выполнения проектов. Всегда добиваешься максимальной эффективности команды.""",
    verbose=True,
    allow_delegation=True,
)

print("✅ Researcher создан")
print("✅ Writer создан")
print("✅ Analyst создан")
print("✅ Manager создан")
print()

# ============================================================
# 2. Создание задач БЕЗ указания агента
# ============================================================

print("="*60)
print("📋 СОЗДАНИЕ ЗАДАЧ (БЕЗ УКАЗАНИЯ АГЕНТА)")
print("="*60)
print()

task1 = Task(
    description="Исследовать тему 'Машинное обучение в медицине' и найти 5 ключевых применений",
    expected_output="Список из 5 применений машинного обучения в медицине с описанием",
    # agent не указан - Crew выберет сам
)

task2 = Task(
    description="Написать краткую статью (3-4 абзаца) о применении ИИ в медицине",
    expected_output="Статья из 3-4 абзацев о применении ИИ в медицине",
    # agent не указан - Crew выберет сам
)

task3 = Task(
    description="Проанализировать найденную информацию и выделить основные тренды",
    expected_output="Анализ трендов с выделением ключевых направлений",
    # agent не указан - Crew выберет сам
)

print("✅ Task 1 создана (без указания агента)")
print("✅ Task 2 создана (без указания агента)")
print("✅ Task 3 создана (без указания агента)")
print()
print("💡 Crew выберет подходящего агента на основе:")
print("   • Роли агента (role)")
print("   • Цели агента (goal)")
print("   • Описания задачи (description)")
print()

# ============================================================
# 3. Создание Crew с hierarchical процессом
# ============================================================

print("="*60)
print("🚣 СОЗДАНИЕ CREW (HIERARCHICAL PROCESS)")
print("="*60)
print()

# Согласно ошибке: "Manager agent should not be included in agents list."
# manager агент должен задаваться только как manager_agent, но не быть в agents

crew = Crew(
    agents=[researcher, writer, analyst],   # manager НЕ включён сюда!
    tasks=[task1, task2, task3],
    process=Process.hierarchical,  # Crew сам выбирает агента
    manager_agent=manager,
    verbose=True,
)

print("✅ Crew создан:")
print(f"   Agents: {len(crew.agents)}")
print(f"   Tasks: {len(crew.tasks)}")
print(f"   Process: hierarchical")
print()
print("💡 Process.hierarchical означает:")
print("   • Crew анализирует каждую задачу")
print("   • Выбирает наиболее подходящего агента")
print("   • Распределяет задачи автоматически (через manager_agent)")
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
print("   • Crew сам выбрал агентов для каждой задачи")
print("   • Выбор основан на ролях и целях агентов")
print("   • Hierarchical процесс более гибкий чем sequential")

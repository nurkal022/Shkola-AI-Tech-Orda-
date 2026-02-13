"""
Лекция 16: Последовательные задачи
===================================
Демонстрация Process.sequential - задачи выполняются по порядку.
Каждая задача может использовать результаты предыдущих.
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
# 1. Создание агентов
# ============================================================

planner = Agent(
    role="Project Planner",
    goal="Создавать детальные планы проектов",
    backstory="""Ты опытный планировщик проектов. Ты умеешь разбивать
    сложные задачи на управляемые шаги и создавать чёткие планы.""",
    verbose=True,
    allow_delegation=False,
)

executor = Agent(
    role="Project Executor",
    goal="Выполнять шаги плана последовательно",
    backstory="""Ты исполнитель проектов. Ты внимательно следуешь планам
    и выполняешь каждый шаг тщательно и последовательно.""",
    verbose=True,
    allow_delegation=False,
)

reviewer = Agent(
    role="Quality Reviewer",
    goal="Проверять качество выполненной работы",
    backstory="""Ты опытный рецензент. Ты проверяешь выполненную работу
    на соответствие требованиям и даёшь конструктивную обратную связь.""",
    verbose=True,
    allow_delegation=False,
)

# ============================================================
# 2. Создание последовательных задач
# ============================================================

task1_plan = Task(
    description="Создай план изучения Python для начинающих из 4 шагов",
    expected_output="План из 4 шагов изучения Python, каждый шаг с кратким описанием",
    agent=planner,
)

task2_execute = Task(
    description="Опиши подробно каждый шаг из плана (что изучать, как практиковаться)",
    expected_output="Подробное описание каждого шага плана с рекомендациями",
    agent=executor,
    context=[task1_plan],  # Использует план
)

task3_review = Task(
    description="Проверь план и описание шагов. Укажи что хорошо и что можно улучшить",
    expected_output="Отзыв о плане и описании шагов с рекомендациями по улучшению",
    agent=reviewer,
    context=[task1_plan, task2_execute],  # Использует оба предыдущих результата
)

# ============================================================
# 3. Создание Crew с sequential процессом
# ============================================================

crew = Crew(
    agents=[planner, executor, reviewer],
    tasks=[task1_plan, task2_execute, task3_review],
    process=Process.sequential,  # Задачи выполняются по порядку
    verbose=True,
)

# ============================================================
# 4. Запуск
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("📋 Последовательные задачи (Sequential Process)")
    print("="*60)
    print()
    print("Процесс выполнения:")
    print("   1. Planner создаёт план")
    print("   2. Executor описывает шаги (использует план)")
    print("   3. Reviewer проверяет (использует план и описание)")
    print()
    print("─" * 60)
    
    result = crew.kickoff()
    
    print("─" * 60)
    print()
    print("="*60)
    print("✅ РЕЗУЛЬТАТ:")
    print("="*60)
    print(result)
    print()
    print("💡 Ключевые моменты:")
    print("   • Задачи выполняются строго по порядку")
    print("   • Каждая задача видит результаты предыдущих (context)")
    print("   • Process.sequential обеспечивает последовательность")

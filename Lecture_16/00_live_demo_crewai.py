"""
Лекция 16: Live Demo - Пошаговое создание CrewAI системы
==========================================================
Интерактивная демонстрация для лекции.
Переписываем код по этапам, показывая архитектуру CrewAI.
"""

from dotenv import load_dotenv
import os

# Загружаем .env из корневой папки проекта
load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

# ============================================================
# ЭТАП 1: Импорты
# ============================================================
print("="*60)
print("📦 ЭТАП 1: ИМПОРТЫ")
print("="*60)
print()

from crewai import Agent, Task, Crew, Process

print("✅ Импортировали:")
print("   • Agent - агент с role, goal, backstory")
print("   • Task - структурированная задача")
print("   • Crew - команда агентов и задач")
print("   • Process - тип процесса (sequential, hierarchical)")
print()

# ============================================================
# ЭТАП 2: Настройка LLM
# ============================================================
print("="*60)
print("🧠 ЭТАП 2: НАСТРОЙКА LLM")
print("="*60)
print()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# CrewAI автоматически использует OPENAI_API_KEY из окружения
print("✅ LLM настроен:")
print("   CrewAI автоматически использует OPENAI_API_KEY из .env")
print("   Модель по умолчанию: gpt-4o-mini")
print()

# ============================================================
# ЭТАП 3: Создание агента с ролевым дизайном
# ============================================================
print("="*60)
print("👤 ЭТАП 3: СОЗДАНИЕ АГЕНТА (Ролевой дизайн)")
print("="*60)
print()

researcher_agent = Agent(
    role="Senior Researcher",
    goal="Найти актуальную и релевантную информацию по заданной теме",
    backstory="""Ты опытный исследователь с более чем 10 лет опыта в анализе данных.
    Ты известен своей способностью находить самую актуальную информацию
    и представлять её в структурированном виде.""",
    verbose=True,
    allow_delegation=False,
)

print("✅ Agent создан:")
print(f"   Role: {researcher_agent.role}")
print(f"   Goal: {researcher_agent.goal[:60]}...")
print(f"   Backstory: {researcher_agent.backstory[:60]}...")
print()
print("💡 Ключевые отличия от AutoGen:")
print("   • Role - роль агента (не просто имя)")
print("   • Goal - конкретная цель агента")
print("   • Backstory - контекст и опыт агента")
print()

# ============================================================
# ЭТАП 4: Создание задачи (Task)
# ============================================================
print("="*60)
print("📋 ЭТАП 4: СОЗДАНИЕ ЗАДАЧИ (Task)")
print("="*60)
print()

research_task = Task(
    description="Исследовать тему 'Искусственный интеллект в образовании' и найти 5 ключевых трендов",
    expected_output="Список из 5 ключевых трендов ИИ в образовании, каждый с кратким описанием",
    agent=researcher_agent,
)

print("✅ Task создана:")
print(f"   Description: {research_task.description[:60]}...")
print(f"   Expected Output: {research_task.expected_output[:60]}...")
print(f"   Agent: {research_task.agent.role}")
print()
print("💡 Ключевые особенности Task:")
print("   • Description - что нужно сделать")
print("   • Expected Output - что ожидается на выходе")
print("   • Agent - кто выполняет задачу")
print()

# ============================================================
# ЭТАП 5: Создание Crew
# ============================================================
print("="*60)
print("🚣 ЭТАП 5: СОЗДАНИЕ CREW")
print("="*60)
print()

crew = Crew(
    agents=[researcher_agent],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True,
)

print("✅ Crew создан:")
print(f"   Agents: {len(crew.agents)}")
print(f"   Tasks: {len(crew.tasks)}")
print(f"   Process: {crew.process}")
print()
print("💡 Process.sequential означает:")
print("   • Задачи выполняются по порядку")
print("   • Task 1 → Task 2 → Task 3")
print()

# ============================================================
# ЭТАП 6: Запуск Crew
# ============================================================
print("="*60)
print("🚀 ЭТАП 6: ЗАПУСК CREW")
print("="*60)
print()

print("⏳ Выполняем crew.kickoff()...")
print("─" * 60)

result = crew.kickoff()

print("─" * 60)
print()
print("✅ Crew выполнен!")
print()

# ============================================================
# ЭТАП 7: Результаты
# ============================================================
print("="*60)
print("📊 ЭТАП 7: РЕЗУЛЬТАТЫ")
print("="*60)
print()

print("Результат выполнения:")
print("─" * 60)
print(result)
print("─" * 60)
print()

# ============================================================
# ЭТАП 8: Расширенный пример - несколько агентов и задач
# ============================================================
print("="*60)
print("👥 ЭТАП 8: РАСШИРЕННЫЙ ПРИМЕР")
print("="*60)
print()

# Создаём второго агента
writer_agent = Agent(
    role="Content Writer",
    goal="Создавать качественный контент на основе исследований",
    backstory="""Ты профессиональный писатель с опытом создания
    образовательного контента. Ты умеешь превращать исследования
    в понятные и интересные тексты.""",
    verbose=True,
    allow_delegation=False,
)

# Создаём вторую задачу с контекстом
writing_task = Task(
    description="На основе исследования создать краткий отчёт (3-4 абзаца) о трендах ИИ в образовании",
    expected_output="Краткий отчёт из 3-4 абзацев о трендах ИИ в образовании",
    agent=writer_agent,
    context=[research_task],  # Использует результаты research_task
)

# Создаём расширенный Crew
extended_crew = Crew(
    agents=[researcher_agent, writer_agent],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

print("✅ Расширенный Crew создан:")
print(f"   Agents: {len(extended_crew.agents)}")
print(f"   Tasks: {len(extended_crew.tasks)}")
print(f"   Process: sequential (задачи выполняются по порядку)")
print()
print("💡 Context в Task:")
print("   • writing_task использует результаты research_task")
print("   • Агент видит вывод предыдущей задачи")
print()

# ============================================================
# Итоги
# ============================================================
print("="*60)
print("✅ LIVE DEMO ЗАВЕРШЕНА")
print("="*60)
print()
print("💡 Ключевые концепции CrewAI:")
print("   1. Agent - ролевой дизайн (role, goal, backstory)")
print("   2. Task - структурированная задача (description, expected_output)")
print("   3. Crew - оркестрация (agents, tasks, process)")
print("   4. Process - sequential (по порядку) или hierarchical (автоматический выбор)")
print("   5. Context - передача результатов между задачами")
print()

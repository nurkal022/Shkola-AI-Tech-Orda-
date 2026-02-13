"""
Лекция 16: Практический пример - Research Crew
===============================================
Команда для исследования темы и создания отчёта.
Researcher → Writer → Editor работают вместе.
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
# 1. Создание команды агентов
# ============================================================

print("="*60)
print("👥 СОЗДАНИЕ RESEARCH CREW")
print("="*60)
print()

researcher = Agent(
    role="Research Analyst",
    goal="Проводить глубокие исследования и находить актуальную информацию",
    backstory="""Ты опытный исследователь-аналитик с более чем 10 лет опыта.
    Ты специализируешься на поиске информации из различных источников,
    проверке фактов и структурировании данных.""",
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Technical Writer",
    goal="Создавать структурированные и понятные отчёты на основе исследований",
    backstory="""Ты технический писатель с опытом создания отчётов и документации.
    Ты умеешь превращать сырые данные исследований в структурированные документы
    с чёткими разделами и выводами.""",
    verbose=True,
    allow_delegation=False,
)

editor = Agent(
    role="Content Editor",
    goal="Улучшать качество контента и проверять соответствие требованиям",
    backstory="""Ты опытный редактор с вниманием к деталям. Ты проверяешь
    контент на ясность, структуру и соответствие требованиям. Ты даёшь
    конструктивную обратную связь для улучшения.""",
    verbose=True,
    allow_delegation=False,
)

print("✅ Research Analyst создан")
print("✅ Technical Writer создан")
print("✅ Content Editor создан")
print()

# ============================================================
# 2. Создание задач
# ============================================================

print("="*60)
print("📋 СОЗДАНИЕ ЗАДАЧ")
print("="*60)
print()

research_task = Task(
    description="""Исследовать тему 'Влияние искусственного интеллекта на рынок труда'.
    Найди информацию о:
    - Какие профессии могут быть затронуты
    - Новые возможности для работников
    - Прогнозы на будущее
    Представь информацию в структурированном виде.""",
    expected_output="Структурированная информация о влиянии ИИ на рынок труда с разделами по профессиям, возможностям и прогнозам",
    agent=researcher,
)

writing_task = Task(
    description="""На основе исследования создай отчёт из 4 разделов:
    1. Введение
    2. Затронутые профессии
    3. Новые возможности
    4. Прогнозы и выводы
    Отчёт должен быть структурированным и понятным.""",
    expected_output="Отчёт из 4 разделов о влиянии ИИ на рынок труда в формате markdown",
    agent=writer,
    context=[research_task],  # Использует результаты исследования
    output_file="output/research_report.md",  # Сохраняет в файл
)

editing_task = Task(
    description="""Проверь отчёт и улучши его:
    - Проверь ясность и структуру
    - Убедись что все разделы присутствуют
    - Улучши формулировки если нужно
    - Добавь краткое резюме в начале""",
    expected_output="Улучшенный отчёт с резюме и проверенной структурой",
    agent=editor,
    context=[research_task, writing_task],  # Использует оба предыдущих результата
    output_file="output/final_report.md",
)

print("✅ Research Task создана")
print("✅ Writing Task создана (использует research)")
print("✅ Editing Task создана (использует research и writing)")
print()

# ============================================================
# 3. Создание Crew
# ============================================================

print("="*60)
print("🚣 СОЗДАНИЕ CREW")
print("="*60)
print()

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True,
)

print("✅ Research Crew создан:")
print(f"   Agents: {len(crew.agents)}")
print(f"   Tasks: {len(crew.tasks)}")
print(f"   Process: sequential")
print()

# ============================================================
# 4. Запуск
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 ЗАПУСК RESEARCH CREW")
    print("="*60)
    print()
    
    result = crew.kickoff()
    
    print()
    print("="*60)
    print("✅ РЕЗУЛЬТАТ:")
    print("="*60)
    print(result)
    print()
    print("💡 Что произошло:")
    print("   1. Researcher провёл исследование")
    print("   2. Writer создал отчёт на основе исследования")
    print("   3. Editor улучшил отчёт")
    print("   4. Результаты сохранены в output/final_report.md")
    print()

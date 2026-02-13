"""
Лекция 16: Практический пример - Code Review Crew
==================================================
Команда для разработки кода: CodeWriter → CodeReviewer → Tester.
Показывает реальный workflow разработки.
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
# 1. Создание команды разработчиков
# ============================================================

print("="*60)
print("💻 СОЗДАНИЕ CODE REVIEW CREW")
print("="*60)
print()

code_writer = Agent(
    role="Senior Python Developer",
    goal="Писать чистый, рабочий код на Python",
    backstory="""Ты опытный Python разработчик с более чем 8 лет опыта.
    Ты специализируешься на написании чистого, читаемого кода
    с хорошей структурой и документацией.""",
    verbose=True,
    allow_delegation=False,
)

code_reviewer = Agent(
    role="Code Review Specialist",
    goal="Проверять код на качество, ошибки и соответствие стандартам",
    backstory="""Ты специалист по code review с глубоким пониманием
    best practices Python. Ты проверяешь код на:
    - Правильность логики
    - Читаемость и стиль
    - Оптимизацию
    - Потенциальные ошибки""",
    verbose=True,
    allow_delegation=False,
)

tester = Agent(
    role="QA Tester",
    goal="Тестировать код и проверять работоспособность",
    backstory="""Ты опытный тестировщик с навыками написания тестов
    и проверки функциональности. Ты умеешь находить edge cases
    и проверять граничные условия.""",
    verbose=True,
    allow_delegation=False,
)

print("✅ Senior Python Developer создан")
print("✅ Code Review Specialist создан")
print("✅ QA Tester создан")
print()

# ============================================================
# 2. Создание задач разработки
# ============================================================

print("="*60)
print("📋 СОЗДАНИЕ ЗАДАЧ РАЗРАБОТКИ")
print("="*60)
print()

coding_task = Task(
    description="""Напиши функцию на Python для вычисления факториала числа.
    Требования:
    - Функция должна называться factorial
    - Должна принимать целое число n
    - Должна возвращать факториал числа
    - Должна включать обработку ошибок (отрицательные числа, не числа)
    - Должна иметь docstring с описанием
    - Должна быть оптимизирована (можно использовать рекурсию или цикл)""",
    expected_output="Полный код функции factorial на Python с обработкой ошибок и docstring",
    agent=code_writer,
    output_file="output/factorial_code.py",
)

review_task = Task(
    description="""Проверь код функции factorial:
    - Правильность логики вычисления
    - Качество кода (читаемость, стиль)
    - Обработку ошибок
    - Оптимизацию
    - Документацию
    Дай конструктивную обратную связь и предложи улучшения если нужно.""",
    expected_output="Отзыв о коде с указанием сильных сторон и рекомендаций по улучшению",
    agent=code_reviewer,
    context=[coding_task],  # Использует код
)

testing_task = Task(
    description="""Проанализируй код функции factorial и предложи тестовые случаи:
    - Нормальные случаи (положительные числа)
    - Граничные случаи (0, 1)
    - Ошибочные случаи (отрицательные числа, не числа)
    - Большие числа
    Опиши что должно тестироваться и какие результаты ожидаются.""",
    expected_output="Список тестовых случаев с описанием входных данных и ожидаемых результатов",
    agent=tester,
    context=[coding_task, review_task],  # Использует код и отзыв
)

print("✅ Coding Task создана")
print("✅ Review Task создана (использует код)")
print("✅ Testing Task создана (использует код и отзыв)")
print()

# ============================================================
# 3. Создание Crew
# ============================================================

print("="*60)
print("🚣 СОЗДАНИЕ CREW")
print("="*60)
print()

crew = Crew(
    agents=[code_writer, code_reviewer, tester],
    tasks=[coding_task, review_task, testing_task],
    process=Process.sequential,
    verbose=True,
)

print("✅ Code Review Crew создан:")
print(f"   Agents: {len(crew.agents)}")
print(f"   Tasks: {len(crew.tasks)}")
print(f"   Process: sequential")
print()

# ============================================================
# 4. Запуск
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 ЗАПУСК CODE REVIEW CREW")
    print("="*60)
    print()
    print("Workflow:")
    print("   1. CodeWriter пишет код")
    print("   2. CodeReviewer проверяет код")
    print("   3. Tester предлагает тесты")
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
    print("💡 Что произошло:")
    print("   1. CodeWriter написал функцию factorial")
    print("   2. CodeReviewer проверил код и дал обратную связь")
    print("   3. Tester предложил тестовые случаи")
    print("   4. Код сохранён в output/factorial_code.py")
    print()

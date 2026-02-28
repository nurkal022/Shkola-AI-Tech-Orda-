"""
Лекция 18: Планировщик исследования
=====================================
Перед тем как искать — нужно спланировать!

Показываем как LLM может разбить сложную тему
на структурированный план исследования.

БЕЗ АГЕНТА — только цепочки (chains).
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

# ============================================================
# Проблема: поиск без плана
# ============================================================

print("=" * 60)
print("🗺️  ПЛАНИРОВЩИК ИССЛЕДОВАНИЯ")
print("=" * 60)
print()
print("Без плана:   случайный поиск → хаотичная информация → плохой отчёт")
print("С планом:    структурированный поиск → связные данные → отличный отчёт")
print()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. Генератор поисковых запросов
# ============================================================

print("=" * 60)
print("1️⃣  ГЕНЕРАТОР ПОИСКОВЫХ ЗАПРОСОВ")
print("=" * 60)
print()

query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты планировщик исследований. Для заданной темы сгенерируй 5 поисковых запросов на английском языке,
которые охватят тему с разных сторон.

Верни ТОЛЬКО JSON (без markdown):
{{
  "topic": "название темы",
  "queries": ["запрос 1", "запрос 2", "запрос 3", "запрос 4", "запрос 5"]
}}""",
    ),
    ("human", "Тема: {topic}"),
])

query_chain = query_prompt | llm | JsonOutputParser()

# Тест
topic = "Quantum Computing"
print(f"📝 Тема: '{topic}'")
plan = query_chain.invoke({"topic": topic})
print(f"\nПоисковые запросы ({len(plan['queries'])}):")
for i, q in enumerate(plan["queries"], 1):
    print(f"  {i}. {q}")
print()

# ============================================================
# 2. Структурированный план исследования
# ============================================================

print("=" * 60)
print("2️⃣  СТРУКТУРИРОВАННЫЙ ПЛАН ИССЛЕДОВАНИЯ")
print("=" * 60)
print()

research_plan_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты научный руководитель. Составь детальный план исследования.

Верни ТОЛЬКО JSON:
{{
  "topic": "название темы",
  "research_goal": "цель исследования одним предложением",
  "subtopics": [
    {{
      "name": "название подтемы",
      "description": "что изучаем в этой подтеме",
      "search_queries": ["запрос 1 на англ.", "запрос 2 на англ."],
      "wikipedia_topic": "тема для Wikipedia API"
    }}
  ],
  "report_sections": ["Введение", "Раздел 1", "Раздел 2", "Тренды", "Заключение"]
}}

Создай 3-4 подтемы, каждая с 2 поисковыми запросами.""",
    ),
    ("human", "Составь план исследования: {topic}"),
])

plan_chain = research_plan_prompt | llm | JsonOutputParser()


def print_research_plan(plan: dict):
    print(f"\n{'─' * 60}")
    print(f"📚 Тема: {plan['topic']}")
    print(f"🎯 Цель: {plan['research_goal']}")
    print(f"\n📌 Подтемы ({len(plan['subtopics'])}):")
    for i, st in enumerate(plan["subtopics"], 1):
        print(f"\n  {i}. {st['name']}")
        print(f"     Описание: {st['description']}")
        print(f"     Запросы: {', '.join(st['search_queries'])}")
        print(f"     Wikipedia: {st['wikipedia_topic']}")
    print(f"\n📄 Разделы отчёта: {' → '.join(plan['report_sections'])}")
    print(f"{'─' * 60}\n")


# Тест с более сложной темой
topic = "AI Agents in 2025"
print(f"📚 Тема: '{topic}'")
print()

plan = plan_chain.invoke({"topic": topic})
print_research_plan(plan)

# ============================================================
# 3. Функция планировщика (для использования в агенте)
# ============================================================

print("=" * 60)
print("3️⃣  ФУНКЦИЯ ПЛАНИРОВЩИКА")
print("=" * 60)
print()


def create_research_plan(topic: str) -> dict:
    """Создаёт план исследования для заданной темы."""
    print(f"🗺️  Создаю план: '{topic}'...")
    plan = plan_chain.invoke({"topic": topic})
    print(f"✅ Готово: {len(plan['subtopics'])} подтем, {len(plan['report_sections'])} разделов")
    return plan


def plan_to_agent_prompt(plan: dict) -> str:
    """Превращает план в промпт для агента-исследователя."""
    text = f"ТЕМА: {plan['topic']}\nЦЕЛЬ: {plan['research_goal']}\n\nПОДТЕМЫ ДЛЯ ИССЛЕДОВАНИЯ:\n"
    for i, st in enumerate(plan["subtopics"], 1):
        text += f"\n{i}. {st['name']}\n"
        text += f"   Wikipedia: {st['wikipedia_topic']}\n"
        text += f"   Запросы: {', '.join(st['search_queries'])}\n"
    text += f"\nСТРУКТУРА ОТЧЁТА: {' → '.join(plan['report_sections'])}"
    return text


# Демонстрация
plan = create_research_plan("Machine Learning in Healthcare")
print()
print("📋 Промпт для агента-исследователя:")
print("─" * 60)
print(plan_to_agent_prompt(plan))

# ============================================================
# Итоги
# ============================================================

print()
print("=" * 60)
print("✅ ИТОГИ: ЗАЧЕМ НУЖЕН ПЛАНИРОВЩИК")
print("=" * 60)
print()
print("1. Сначала думаем (план), потом делаем (поиск)")
print("2. Разбиваем сложную тему на конкретные подтемы")
print("3. Генерируем точные поисковые запросы заранее")
print("4. Знаем структуру отчёта до начала сбора данных")
print()
print("➡️  Следующий шаг: 02_deep_research_agent.py")
print("   Используем этот план как входные данные для агента")

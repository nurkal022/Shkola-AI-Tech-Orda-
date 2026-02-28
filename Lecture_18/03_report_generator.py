"""
Лекция 18: Генератор исследовательских отчётов
================================================
Один и тот же набор данных → разные форматы отчётов.

Показываем как LLM может превратить сырые данные
в профессиональный документ для разной аудитории.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

print("=" * 60)
print("📝 ГЕНЕРАТОР ИССЛЕДОВАТЕЛЬСКИХ ОТЧЁТОВ")
print("=" * 60)
print()
print("Идея: разделить агента на 2 роли:")
print("  1. Исследователь — собирает данные")
print("  2. Редактор      — форматирует под нужную аудиторию")
print()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ============================================================
# 1. Шаблоны для разных аудиторий
# ============================================================

print("=" * 60)
print("1️⃣  ШАБЛОНЫ ДЛЯ РАЗНЫХ АУДИТОРИЙ")
print("=" * 60)
print()

TEMPLATES = {
    "educational": """
Создай учебный материал для студентов.
Структура:
# [Тема]: Полный учебный гайд
## Что это такое? (простыми словами)
## История развития
## Как это работает?
## Примеры из реальной жизни
## Почему это важно?
## Ресурсы для изучения
Пиши просто, избегай жаргона, добавляй аналогии.""",

    "technical": """
Создай технический отчёт для разработчиков.
Структура:
# [Тема]: Технический обзор
## Архитектура и принципы работы
## Ключевые технологии и инструменты
## Практическое применение (с примерами)
## Open-Source проекты (названия, звёзды)
## Ограничения и риски
## Заключение
Используй технические термины, будь точным.""",

    "executive": """
Создай краткий исполнительный отчёт (executive summary) для руководителей.
Структура:
# [Тема]: Краткий обзор
## Суть (2-3 предложения)
## Ключевые выводы (5 пунктов)
## Возможности для бизнеса
## Риски
## Рекомендации
Объём: не более 300 слов. Только факты и выводы.""",

    "analytical": """
Создай аналитический отчёт.
Структура:
# Аналитический отчёт: [Тема]
## Текущее состояние
## SWOT-анализ
   - Сильные стороны
   - Слабые стороны
   - Возможности
   - Угрозы
## Тренды и прогнозы
## Сравнение с альтернативами
## Выводы
Включай цифры, обоснованные выводы.""",
}

print("📋 Доступные шаблоны:")
for name in TEMPLATES:
    first_line = TEMPLATES[name].strip().split("\n")[0]
    print(f"   • {name:12s} → {first_line}")
print()

# ============================================================
# 2. Функция генерации отчёта
# ============================================================

print("=" * 60)
print("2️⃣  ФУНКЦИЯ ГЕНЕРАЦИИ ОТЧЁТА")
print("=" * 60)
print()


def generate_report(topic: str, raw_data: str, template: str = "educational") -> str:
    """Генерирует структурированный отчёт из сырых данных."""

    template_instructions = TEMPLATES.get(template, TEMPLATES["educational"])
    today = datetime.now().strftime("%d.%m.%Y")

    messages = [
        SystemMessage(content=(
            f"Ты профессиональный редактор исследовательских отчётов.\n"
            f"Тебе предоставлены сырые данные по теме \"{topic}\".\n"
            f"Преобразуй их в полноценный отчёт на РУССКОМ языке.\n\n"
            f"Инструкции по форматированию:\n{template_instructions}\n\n"
            f"Правила:\n"
            f"- Пиши строго на русском языке\n"
            f"- Используй markdown форматирование\n"
            f"- Сохраняй все важные факты из исходных данных\n"
            f"- В начале отчёта укажи дату: {today}\n"
            f"- В конце добавь раздел \"Источники\" со списком источников"
        )),
        HumanMessage(content=f"Тема: {topic}\n\nИсходные данные:\n{raw_data}\n\nСоздай отчёт."),
    ]

    result = llm.invoke(messages)
    return result.content


# ============================================================
# 3. Демонстрация
# ============================================================

print("=" * 60)
print("3️⃣  ДЕМОНСТРАЦИЯ: 4 ФОРМАТA ДЛЯ ОДНИХ ДАННЫХ")
print("=" * 60)
print()

# Сырые данные (как будто собрал агент-исследователь)
TOPIC = "Large Language Models (LLMs)"
RAW_DATA = """
Источник 1 (Wikipedia):
Large Language Models — это нейронные сети с миллиардами параметров,
обученные на огромных объёмах текста. Появились благодаря архитектуре
Transformer (2017, Google). Ключевые модели: GPT-4 (OpenAI, 1 трлн параметров),
Claude 3 (Anthropic), Gemini (Google), Llama 3 (Meta, открытый).

Источник 2 (web_search: "LLM applications 2025"):
В 2025 году LLM используются в:
- Программирование: GitHub Copilot, Cursor — 40% разработчиков используют ИИ-помощников
- Юридические услуги: анализ контрактов, сокращение времени review на 70%
- Медицина: помощь в диагностике, анализ медицинских записей
- Образование: персональные туторы, генерация учебных материалов
- Клиентская поддержка: замена 30% тикетов уровня L1

Источник 3 (GitHub search: "large language model"):
Популярные open-source проекты:
- transformers (Hugging Face) — 130k ⭐, полная экосистема ML
- llama.cpp — 60k ⭐, запуск LLM локально (CPU/GPU)
- ollama — 50k ⭐, удобное управление локальными моделями
- vllm — 25k ⭐, высокопроизводительный inference

Источник 4 (search_news: "LLM trends 2025"):
Тренды 2025 года:
- Multimodal LLM: GPT-4o, Gemini 1.5 работают с текстом, изображениями, аудио
- Edge LLMs: Phi-3 (Microsoft), Gemma (Google) — запуск на телефонах
- Контекстное окно: Claude — 200k токенов, Gemini — 1M токенов
- Agents и Tool Use стали стандартом: LLM вызывают внешние API
- RAG (Retrieval Augmented Generation) — LLM + база знаний компании

Источник 5 (web_search: "LLM limitations risks"):
Ограничения:
- Hallucinations: модели уверенно выдают неверные факты
- Знания ограничены датой обучения (knowledge cutoff)
- Высокие вычислительные затраты: GPT-4 — $0.03 за 1k токенов
- Проблемы с конфиденциальностью: данные могут обучить модель
- Bias: модели воспроизводят предубеждения из обучающих данных
"""

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

for template_name in TEMPLATES:
    print(f"📄 Генерирую [{template_name}]...", end="", flush=True)
    try:
        report = generate_report(TOPIC, RAW_DATA, template_name)
        filepath = os.path.join(output_dir, f"report_llm_{template_name}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        print(f" ✅ ({len(report.split())} слов) → output/report_llm_{template_name}.md")
    except Exception as e:
        print(f" ❌ {e}")

print()

# ============================================================
# 4. Показываем educational отчёт
# ============================================================

print("=" * 60)
print("4️⃣  ПРИМЕР: ОБРАЗОВАТЕЛЬНЫЙ ОТЧЁТ")
print("=" * 60)
print()

educational_path = os.path.join(output_dir, "report_llm_educational.md")
if os.path.exists(educational_path):
    with open(educational_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(content[:1200])
    if len(content) > 1200:
        print(f"\n...[отчёт продолжается, всего {len(content.split())} слов]")
print()

# ============================================================
# 5. Функция для интеграции с агентом
# ============================================================

print("=" * 60)
print("5️⃣  ИНТЕГРАЦИЯ С АГЕНТОМ-ИССЛЕДОВАТЕЛЕМ")
print("=" * 60)
print()


def research_and_report(topic: str, notes_path: str, template: str = "educational") -> str:
    """
    Берёт накопленные заметки агента и генерирует отчёт.
    Используй эту функцию после того как агент собрал данные.

    Args:
        topic: Тема исследования
        notes_path: Путь к JSON файлу с заметками агента
        template: Формат отчёта (educational/technical/executive/analytical)
    """
    import json

    if not os.path.exists(notes_path):
        return "❌ Файл с заметками не найден"

    with open(notes_path, "r", encoding="utf-8") as f:
        notes = json.load(f)

    # Преобразуем заметки в текст
    raw_data = ""
    for section, items in notes.items():
        raw_data += f"\n## {section}\n"
        for item in items:
            raw_data += f"{item}\n\n"

    report = generate_report(topic, raw_data, template)

    # Сохраняем
    filename = f"report_{topic.replace(' ', '_').lower()[:25]}_{template}.md"
    filepath = os.path.join(os.path.dirname(notes_path), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return f"✅ Отчёт сохранён: {filepath}"


print("💡 Функция research_and_report() берёт заметки агента")
print("   и превращает их в любой формат отчёта:")
print()
print("   notes.json → generate_report(topic, notes, 'technical') → report.md")
print()

# Проверяем есть ли заметки от предыдущего запуска
notes_path = os.path.join(output_dir, "_notes.json")
if os.path.exists(notes_path):
    print("📋 Найдены заметки от агента-исследователя! Генерирую технический отчёт...")
    result = research_and_report("Deep Research Agent", notes_path, "technical")
    print(result)
else:
    print("ℹ️  Запустите 02_deep_research_agent.py чтобы собрать заметки,")
    print("   затем эта функция сгенерирует из них отчёт.")

# ============================================================
# Итоги
# ============================================================

print()
print("=" * 60)
print("✅ ИТОГИ: ПОЛНАЯ АРХИТЕКТУРА")
print("=" * 60)
print()
print("┌─────────────────────────────────────────┐")
print("│         Deep Research Pipeline          │")
print("├─────────────────────────────────────────┤")
print("│  [Тема]                                 │")
print("│     ↓                                   │")
print("│  Планировщик (LLM)                      │")
print("│     ↓ план + запросы                    │")
print("│  Агент-Исследователь                    │")
print("│    ├─ wikipedia_search()                │")
print("│    ├─ search_web()                      │")
print("│    ├─ search_news()                     │")
print("│    ├─ scrape_page()                     │")
print("│    ├─ github_search()                   │")
print("│    └─ add_note() → _notes.json          │")
print("│     ↓ сырые данные                      │")
print("│  Генератор отчётов (LLM)               │")
print("│    ├─ educational → для студентов       │")
print("│    ├─ technical   → для разработчиков  │")
print("│    ├─ executive   → для менеджеров     │")
print("│    └─ analytical  → для аналитиков     │")
print("│     ↓                                   │")
print("│  [Готовый отчёт .md]                    │")
print("└─────────────────────────────────────────┘")

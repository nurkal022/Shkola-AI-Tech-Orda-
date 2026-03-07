"""
Лекция 19: Оценка производительности с LangSmith
==================================================
Как оценивать качество LLM-приложений:
  - Создание датасетов с примерами
  - Запуск оценки (evaluate)
  - Встроенные и кастомные метрики
  - Сравнение промптов и моделей
"""

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Lecture-19-Evaluation"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, evaluate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
client = Client()

# ============================================================
# 1. Зачем нужна оценка?
# ============================================================

print("=" * 60)
print("1. ЗАЧЕМ НУЖНА ОЦЕНКА?")
print("=" * 60)
print()
print("  Проблема: 'работает' != 'работает хорошо'")
print()
print("  Без оценки:")
print("    - Промпт кажется хорошим после 3 ручных тестов")
print("    - Через неделю клиенты жалуются на плохие ответы")
print()
print("  С оценкой:")
print("    - Тестируем на 50+ примерах")
print("    - Измеряем точность, полноту, качество")
print("    - Сравниваем версии промптов числами, а не ощущениями")
print()

# ============================================================
# 2. Создание датасета
# ============================================================

print("=" * 60)
print("2. СОЗДАНИЕ ДАТАСЕТА")
print("=" * 60)
print()

DATASET_NAME = "Lecture-19-QA-Dataset"

# Проверяем, существует ли датасет
existing = list(client.list_datasets(dataset_name=DATASET_NAME))

if existing:
    dataset = existing[0]
    print(f"  Датасет '{DATASET_NAME}' уже существует (id: {dataset.id})")
else:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Тестовые вопросы и ожидаемые ответы для оценки QA-системы",
    )
    print(f"  Датасет '{DATASET_NAME}' создан")

    # Добавляем примеры (inputs + expected outputs)
    examples = [
        {
            "inputs": {"question": "Что такое Python?"},
            "outputs": {"answer": "Python — это высокоуровневый язык программирования общего назначения."},
        },
        {
            "inputs": {"question": "Кто создал Linux?"},
            "outputs": {"answer": "Linux был создан Линусом Торвальдсом в 1991 году."},
        },
        {
            "inputs": {"question": "Что такое API?"},
            "outputs": {"answer": "API — это интерфейс программирования приложений, набор правил для взаимодействия между программами."},
        },
        {
            "inputs": {"question": "Что такое machine learning?"},
            "outputs": {"answer": "Machine learning — это подраздел искусственного интеллекта, где модели обучаются на данных без явного программирования."},
        },
        {
            "inputs": {"question": "Что такое LangChain?"},
            "outputs": {"answer": "LangChain — это фреймворк для разработки приложений на базе больших языковых моделей."},
        },
        {
            "inputs": {"question": "Что такое Docker?"},
            "outputs": {"answer": "Docker — это платформа для контейнеризации, позволяющая упаковывать приложения с их зависимостями."},
        },
        {
            "inputs": {"question": "Что такое REST?"},
            "outputs": {"answer": "REST — это архитектурный стиль для создания веб-сервисов, использующий HTTP-методы."},
        },
        {
            "inputs": {"question": "Что такое Git?"},
            "outputs": {"answer": "Git — это распределённая система контроля версий для отслеживания изменений в коде."},
        },
    ]

    for ex in examples:
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            dataset_id=dataset.id,
        )

    print(f"  Добавлено {len(examples)} примеров")

print(f"  Датасет: {DATASET_NAME}")
print(f"  Посмотреть: https://smith.langchain.com -> Datasets")
print()

# ============================================================
# 3. Целевая функция (что оцениваем)
# ============================================================

print("=" * 60)
print("3. ЦЕЛЕВАЯ ФУНКЦИЯ")
print("=" * 60)
print()

# Вариант A: краткий промпт
chain_concise = (
    ChatPromptTemplate.from_messages([
        ("system", "Отвечай кратко (1-2 предложения) на русском."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

# Вариант B: подробный промпт
chain_detailed = (
    ChatPromptTemplate.from_messages([
        ("system", "Дай подробный и точный ответ на русском (3-4 предложения). Включи ключевые факты."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)


def target_concise(inputs: dict) -> dict:
    """Целевая функция для краткого варианта."""
    answer = chain_concise.invoke(inputs)
    return {"answer": answer}


def target_detailed(inputs: dict) -> dict:
    """Целевая функция для подробного варианта."""
    answer = chain_detailed.invoke(inputs)
    return {"answer": answer}


# Быстрый тест
print("  Вариант A (краткий):")
r = target_concise({"question": "Что такое Python?"})
print(f"    {r['answer'][:120]}")
print()

print("  Вариант B (подробный):")
r = target_detailed({"question": "Что такое Python?"})
print(f"    {r['answer'][:200]}")
print()

# ============================================================
# 4. Кастомные метрики (evaluators)
# ============================================================

print("=" * 60)
print("4. КАСТОМНЫЕ МЕТРИКИ")
print("=" * 60)
print()


def correctness(run, example) -> dict:
    """
    Оценивает корректность ответа через LLM.
    Сравнивает ответ модели с эталонным ответом.
    """
    predicted = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    grader = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    grade_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Ты оценщик ответов. Сравни ответ модели с эталонным.\n"
            "Оцени: содержит ли ответ ту же ключевую информацию?\n"
            "Ответь ТОЛЬКО одним словом: CORRECT или INCORRECT.",
        ),
        (
            "human",
            f"Вопрос: {question}\n"
            f"Эталонный ответ: {expected}\n"
            f"Ответ модели: {predicted}\n\n"
            f"Оценка:",
        ),
    ])

    grade_chain = grade_prompt | grader | StrOutputParser()
    grade = grade_chain.invoke({})

    is_correct = "correct" in grade.lower()
    return {"key": "correctness", "score": 1.0 if is_correct else 0.0}


def answer_length(run, example) -> dict:
    """Измеряет длину ответа в словах."""
    predicted = run.outputs.get("answer", "")
    word_count = len(predicted.split())
    return {"key": "answer_length", "score": word_count}


def contains_keywords(run, example) -> dict:
    """Проверяет, содержит ли ответ ключевые слова из эталона."""
    predicted = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()

    expected_words = set(expected.split())
    predicted_words = set(predicted.split())

    # Убираем стоп-слова
    stop_words = {"это", "и", "в", "на", "для", "—", "что", "с", "из", "по", "не", "а", "как", "его", "или"}
    expected_keywords = expected_words - stop_words
    if not expected_keywords:
        return {"key": "keyword_overlap", "score": 1.0}

    overlap = expected_keywords & predicted_words
    score = len(overlap) / len(expected_keywords)

    return {"key": "keyword_overlap", "score": round(score, 2)}


print("  Метрики:")
print("    - correctness:     LLM оценивает корректность (0 или 1)")
print("    - answer_length:   длина ответа в словах")
print("    - keyword_overlap: совпадение ключевых слов с эталоном")
print()

# ============================================================
# 5. Запуск оценки
# ============================================================

print("=" * 60)
print("5. ЗАПУСК ОЦЕНКИ")
print("=" * 60)
print()

evaluators = [correctness, answer_length, contains_keywords]

# Оценка варианта A (краткий)
print("  Оцениваем Вариант A (краткий)...")
results_a = evaluate(
    target_concise,
    data=DATASET_NAME,
    evaluators=evaluators,
    experiment_prefix="QA-Concise",
    max_concurrency=2,
)
print(f"  Вариант A завершён!")
print()

# Оценка варианта B (подробный)
print("  Оцениваем Вариант B (подробный)...")
results_b = evaluate(
    target_detailed,
    data=DATASET_NAME,
    evaluators=evaluators,
    experiment_prefix="QA-Detailed",
    max_concurrency=2,
)
print(f"  Вариант B завершён!")
print()

# ============================================================
# 6. Анализ результатов
# ============================================================

print("=" * 60)
print("6. АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 60)
print()


def summarize_results(results, name: str):
    """Выводит сводку по результатам оценки."""
    scores = {"correctness": [], "answer_length": [], "keyword_overlap": []}

    for result in results:
        feedback = result.get("evaluation_results", {}).get("results", [])
        for f in feedback:
            key = f.key if hasattr(f, "key") else f.get("key", "")
            score = f.score if hasattr(f, "score") else f.get("score", 0)
            if key in scores:
                scores[key].append(score)

    print(f"  {name}:")
    for metric, values in scores.items():
        if values:
            avg = sum(values) / len(values)
            print(f"    {metric:20s}: avg={avg:.2f} (n={len(values)})")
        else:
            print(f"    {metric:20s}: нет данных")
    print()


# Пытаемся вывести сводку
try:
    summarize_results(results_a, "Вариант A (краткий)")
    summarize_results(results_b, "Вариант B (подробный)")
except Exception:
    print("  Сводка доступна в LangSmith дашборде:")
    print("  -> Datasets -> Lecture-19-QA-Dataset -> Experiments")
    print("  -> Сравните QA-Concise и QA-Detailed")
    print()

print("  В LangSmith (Datasets -> Experiments):")
print("    - Таблица: каждый пример + ответ + оценки")
print("    - Графики: распределение correctness, length")
print("    - Сравнение: Concise vs Detailed рядом")
print()

# ============================================================
# 7. Пример: оценка с другой моделью
# ============================================================

print("=" * 60)
print("7. СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)
print()
print("  Тот же подход работает для сравнения моделей:")
print()
print("  # Модель A")
print("  llm_a = ChatOpenAI(model='gpt-4o-mini')")
print("  # Модель B")
print("  llm_b = ChatOpenAI(model='gpt-4o')")
print()
print("  evaluate(target_a, data=dataset, experiment_prefix='GPT4o-mini')")
print("  evaluate(target_b, data=dataset, experiment_prefix='GPT4o')")
print()
print("  -> В LangSmith сравните accuracy, latency и стоимость!")
print()

# ============================================================
# Итоги
# ============================================================

print("=" * 60)
print("ИТОГИ: ОЦЕНКА С LANGSMITH")
print("=" * 60)
print()
print("  1. Датасет = коллекция (input, expected_output) пар")
print("  2. Целевая функция = то, что оцениваем")
print("  3. Метрики (evaluators) = как оцениваем:")
print("     - LLM-as-judge (correctness)")
print("     - Числовые (длина ответа)")
print("     - Overlap (совпадение ключевых слов)")
print("  4. evaluate() запускает всё и сохраняет в LangSmith")
print("  5. Дашборд: сравнение экспериментов, графики, таблицы")
print()
print("  Паттерн улучшения:")
print("    Промпт v1 -> evaluate -> 70% accuracy")
print("    Промпт v2 -> evaluate -> 85% accuracy")
print("    Промпт v3 -> evaluate -> 92% accuracy")
print("    -> Данные, а не ощущения!")

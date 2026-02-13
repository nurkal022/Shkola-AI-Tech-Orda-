# Лекция 16: Совместная работа агентов с CrewAI

## 📋 Содержание

Изучение фреймворка CrewAI для оркестровки мультиагентных систем. CrewAI фокусируется на ролевом дизайне агентов, позволяя определить «команду» агентов с конкретными задачами и процессом для достижения общей цели.

---

## 📁 Файлы лекции

| Файл | Описание |
|------|----------|
| `00_live_demo_crewai.py` | **Live Demo** - Полная пошаговая демонстрация (8 этапов) |
| `00_live_demo_step_by_step.py` | **Live Demo** - Упрощённая версия для переписывания |
| `01_basic_crew.py` | Базовый Crew: один Agent, одна Task |
| `02_role_based_agents.py` | Ролевой дизайн агентов (role, goal, backstory) |
| `03_sequential_tasks.py` | Последовательные задачи (Process.sequential) |
| `04_hierarchical_process.py` | Иерархический процесс (Process.hierarchical) |
| `05_research_crew.py` | Практический пример: Research Crew |
| `06_code_review_crew.py` | Практический пример: Code Review Crew |

---

## 🎯 Ключевые концепции

### Что такое CrewAI?

```
CrewAI = Ролевой дизайн + Структурированные задачи + Оркестрация

Agent (Роль):
  • role - роль агента
  • goal - цель агента
  • backstory - контекст и опыт

Task (Задача):
  • description - что нужно сделать
  • expected_output - что ожидается
  • agent - кто выполняет
  • context - результаты предыдущих задач

Crew (Команда):
  • agents - список агентов
  • tasks - список задач
  • process - тип процесса (sequential/hierarchical)
```

### CrewAI vs AutoGen

| Аспект | AutoGen | CrewAI |
|--------|---------|--------|
| **Дизайн агента** | system_message | role, goal, backstory |
| **Структура задач** | Нет явных задач | Task с description/output |
| **Оркестрация** | GroupChat с speaker selection | Crew с Process |
| **Предсказуемость** | Менее предсказуемо | Более структурировано |
| **Использование** | Диалоги, обсуждения | Чёткие задачи и роли |
| **Контекст** | Через сообщения | Через context в Task |

---

## 🚀 Быстрый старт

```bash
pip install -r requirements.txt
```

Создайте `.env`:
```
OPENAI_API_KEY=your_key
```

### Запуск примеров

```bash
# Базовый Crew
python 01_basic_crew.py

# Ролевой дизайн
python 02_role_based_agents.py

# Последовательные задачи
python 03_sequential_tasks.py

# Иерархический процесс
python 04_hierarchical_process.py

# Research Crew
python 05_research_crew.py

# Code Review Crew
python 06_code_review_crew.py
```

---

## 💡 Примеры использования

### Пример 1: Базовый Crew
Один агент выполняет одну задачу - минимальная структура.

### Пример 2: Ролевой дизайн
Агенты с чётко определёнными ролями, целями и контекстом.

### Пример 3: Последовательные задачи
Задачи выполняются по порядку, каждая использует результаты предыдущих.

### Пример 4: Иерархический процесс
Crew автоматически выбирает подходящего агента для каждой задачи.

### Пример 5: Research Crew
Команда для исследования темы и создания отчёта (Researcher → Writer → Editor).

### Пример 6: Code Review Crew
Команда для разработки кода (CodeWriter → CodeReviewer → Tester).

---

## 🔧 Технические детали

### Создание агента

```python
agent = Agent(
    role="Senior Researcher",
    goal="Найти актуальную информацию",
    backstory="Опытный исследователь с 10+ лет опыта",
    verbose=True,
    allow_delegation=False,
)
```

### Создание задачи

```python
task = Task(
    description="Исследовать тему X",
    expected_output="Список из 10 ключевых пунктов",
    agent=researcher_agent,
    context=[previous_task],  # Использует результаты предыдущей задачи
    output_file="output/report.md",  # Сохраняет результат в файл
)
```

### Создание Crew

```python
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process=Process.sequential,  # или Process.hierarchical
    verbose=True,
)

result = crew.kickoff()
```

### Типы процессов

- **Process.sequential**: Задачи выполняются строго по порядку
- **Process.hierarchical**: Crew выбирает агента для задачи на основе роли

---

## 📚 Дополнительные ресурсы

- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)

---

## 🔗 Связь с другими лекциями

- **Лекция 12**: Введение в ИИ-агенты (одиночные агенты)
- **Лекция 13**: ReAct фреймворк (рассуждения агента)
- **Лекция 14**: LangGraph (графы состояний)
- **Лекция 15**: AutoGen (мультиагентные системы с диалогами)
- **Лекция 16**: CrewAI ← *текущая* (ролевой дизайн и структурированные задачи)

---

## 🎓 Практические задания

1. **Создайте Research Crew:**
   - Researcher исследует тему
   - Writer создаёт отчёт
   - Editor улучшает отчёт

2. **Создайте Development Crew:**
   - Planner планирует проект
   - Developer пишет код
   - Reviewer проверяет код
   - Tester тестирует код

3. **Сравните Sequential и Hierarchical:**
   - Создайте один и тот же Crew с разными процессами
   - Проанализируйте различия в выполнении

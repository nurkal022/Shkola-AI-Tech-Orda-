# Лекция 15: Мультиагентные системы с AutoGen

## 📋 Содержание

Изучение создания мультиагентных систем с использованием фреймворка AutoGen от Microsoft. Специализированные агенты работают вместе, общаясь друг с другом для решения сложных задач.

---

## 📁 Файлы лекции

| Файл | Описание |
|------|----------|
| `00_live_demo_autogen.py` | **Live Demo** - Полная пошаговая демонстрация |
| `00_live_demo_step_by_step.py` | **Live Demo** - Упрощённая версия для переписывания |
| `01_basic_two_agents.py` | Базовый диалог двух агентов (User и Assistant) |
| `02_group_chat.py` | Групповой чат с Writer, Critic, Editor |
| `03_specialized_agents.py` | Planner, Executor, Reviewer работают вместе |
| `04_code_generation.py` | CodeWriter, CodeReviewer, Tester для разработки кода |
| `05_problem_solving.py` | Researcher, Analyst, Strategist, Implementer решают сложные задачи |

---

## 🎯 Ключевые концепции

### Что такое мультиагентная система?

```
Один агент:
  User → Assistant → Ответ

Мультиагентная система:
  User → Writer → Critic → Editor → User
         (каждый вносит свой вклад)
```

### Компоненты AutoGen

- **ConversableAgent** - базовый класс для агентов
- **GroupChat** - групповой чат с несколькими агентами
- **GroupChatManager** - управляет общением в группе
- **Speaker Selection** - метод выбора следующего говорящего

### Методы выбора говорящего

| Метод | Описание |
|-------|----------|
| `round_robin` | По очереди |
| `random` | Случайный выбор |
| `auto` | LLM выбирает (умный) |
| `manual` | Человек выбирает |

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
# Базовый диалог
python 01_basic_two_agents.py

# Групповой чат
python 02_group_chat.py

# Специализированные агенты
python 03_specialized_agents.py

# Генерация кода
python 04_code_generation.py

# Решение сложных задач
python 05_problem_solving.py
```

---

## 💡 Примеры использования

### Пример 1: Базовый диалог
User и Assistant общаются друг с другом.

### Пример 2: Групповой чат
Writer создаёт контент → Critic даёт обратную связь → Editor улучшает → User одобряет.

### Пример 3: Специализированные агенты
Planner планирует → Executor выполняет → Reviewer проверяет.

### Пример 4: Генерация кода
CodeWriter пишет код → CodeReviewer проверяет → Tester тестирует.

### Пример 5: Решение сложных задач
Researcher исследует → Analyst анализирует → Strategist предлагает стратегию → Implementer реализует.

---

## 🔧 Технические детали

### Создание агента

```python
agent = ConversableAgent(
    name="AgentName",
    system_message="Роль и задачи агента",
    llm_config={
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    human_input_mode="NEVER",  # или "ALWAYS"
)
```

### Создание группового чата

```python
group_chat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=10,
    speaker_selection_method="auto",
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)
```

### Запуск чата

```python
user_agent.initiate_chat(
    recipient=manager,
    message="Задача для агентов",
)
```

---

## 📚 Дополнительные ресурсы

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [AutoGen GitHub](https://github.com/microsoft/autogen)

---

## 🔗 Связь с другими лекциями

- **Лекция 12**: Введение в ИИ-агенты (одиночные агенты)
- **Лекция 13**: ReAct фреймворк (рассуждения агента)
- **Лекция 14**: LangGraph (графы состояний)
- **Лекция 15**: Мультиагентные системы ← *текущая*

# Лекция 19: Наблюдаемость и отладка с LangSmith

## Описание

Отслеживайте и отлаживайте свои LLM-приложения. Используем LangSmith для:
- **Трейсинга** — видим каждый вызов LLM, каждый шаг цепочки, каждую итерацию агента
- **Отладки** — находим причины ошибок, зацикливаний и неправильных ответов
- **Оценки** — измеряем качество промптов и моделей на датасетах с метриками

---

## Настройка

### 1. Получите ключ LangSmith

1. Зарегистрируйтесь: https://smith.langchain.com
2. Settings -> API Keys -> Create API Key
3. Добавьте в `.env`:

```
LANGCHAIN_API_KEY=lsv2_pt_xxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=My-Project
```

### 2. Установите зависимости

```bash
pip install -r requirements.txt
```

---

## Файлы

| Файл | Описание |
|------|----------|
| `00_live_demo.py` | **Live Demo** — LLM, Chain, Agent с трейсингом + теги |
| `01_langsmith_setup.py` | Настройка LangSmith — подключение, проекты, вкл/выкл |
| `02_tracing_chains.py` | Трейсинг цепочек + `@traceable` для своих функций |
| `03_debugging_agents.py` | Отладка агентов — ошибки, retry, callbacks, чеклист |
| `04_evaluation.py` | Оценка — датасеты, метрики, сравнение промптов |

---

## Ключевые концепции

### Что видно в LangSmith?

```
Простой вызов LLM:
  ChatOpenAI -> [промпт] -> [ответ] -> latency, tokens, cost

Цепочка (Chain):
  RunnableSequence
    ├── ChatPromptTemplate  (формирование промпта)
    ├── ChatOpenAI          (вызов LLM)
    └── StrOutputParser     (парсинг ответа)

Агент:
  AgentExecutor
    ├── Iteration 1: LLM -> Tool Call -> Tool Result
    ├── Iteration 2: LLM -> Tool Call -> Tool Result
    └── Iteration 3: LLM -> Final Answer
```

### @traceable — трейсинг своих функций

```python
from langsmith import traceable

@traceable(name="my_pipeline")
def my_pipeline(query):
    data = fetch_data(query)      # child trace
    result = analyze(data)        # child trace
    return result
```

### Метаданные и теги

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    tags=["experiment-A", "v2"],
    metadata={"user_id": "123", "version": "2.0"},
    run_name="My-Experiment",
)

chain.invoke(inputs, config=config)
```

### Оценка (Evaluation)

```
Датасет (вопрос + ожидаемый ответ)
     |
     v
evaluate(target_fn, data=dataset, evaluators=[...])
     |
     v
LangSmith Dashboard: таблицы, графики, сравнение
```

---

## Чеклист отладки

| Проблема | Что смотреть в LangSmith |
|----------|--------------------------|
| Неправильный ответ | Промпт, tool calls, ответы инструментов |
| Зацикливание агента | Количество итераций, повторяющиеся вызовы |
| Медленная работа | Latency каждого шага |
| Высокая стоимость | Total tokens, стоимость за запрос |
| Регрессия качества | Сравнение экспериментов в Datasets |

---

## Запуск

```bash
# Live demo
python 00_live_demo.py

# Настройка и подключение
python 01_langsmith_setup.py

# Трейсинг цепочек и функций
python 02_tracing_chains.py

# Отладка агентов
python 03_debugging_agents.py

# Оценка производительности
python 04_evaluation.py
```

Результаты видны в дашборде: https://smith.langchain.com

# Лекция 14: Создание агентов с помощью LangGraph

## 📋 Содержание

Переход от линейных цепочек к циклическим графам. LangGraph позволяет создавать надёжные агентные системы с полным контролем над потоком выполнения.

---

## 📁 Файлы лекции

| Файл | Описание |
|------|----------|
| `00_live_demo_simple_graph.py` | **Live Demo** - Простой граф по этапам |
| `00_live_demo_step_by_step.py` | **Live Demo** - Упрощённая версия для переписывания |
| `01_simple_graph.py` | Простой граф (минимальный код) |
| `02_react_langgraph.py` | ReAct агент на LangGraph |
| `03_human_in_loop.py` | **НОВОЕ** - Human-in-the-Loop |
| `04_self_correcting.py` | **НОВОЕ** - Self-Correcting агент |
| `05_plan_execute.py` | **НОВОЕ** - Plan-and-Execute |

---

## 🎯 Ключевые концепции

### LangGraph vs AgentExecutor

| Аспект | AgentExecutor | LangGraph |
|--------|--------------|-----------|
| Структура | Линейный цикл | Граф состояний |
| Контроль | Ограниченный | Полный |
| Условия | Сложно | add_conditional_edges |
| Human-in-the-Loop | Нет | Да |
| Self-Correcting | Нет | Да |
| Plan-and-Execute | Нет | Да |
| Визуализация | Нет | draw_mermaid() |

### Компоненты LangGraph

- **State** - состояние между узлами
- **Node** - функция обработки
- **Edge** - переход между узлами
- **Conditional Edge** - условный переход
- **START / END** - вход и выход графа

---

## 🚀 Быстрый старт

```bash
pip install -r requirements.txt
```

Создайте `.env`:
```
OPENAI_API_KEY=your_key
```

### Запуск (без API - простой граф)
```bash
python 00_live_demo_simple_graph.py
```

### Запуск (с API)
```bash
python 02_react_langgraph.py
python 03_human_in_loop.py
python 04_self_correcting.py
python 05_plan_execute.py
```

---

## 🆕 Что нового (чего не было в Лекции 12-13)

### 1. Human-in-the-Loop
Агент спрашивает подтверждение перед важными действиями (например, отправка email).

### 2. Self-Correcting
Агент генерирует ответ → Критик проверяет → При низкой оценке переделывает.

### 3. Plan-and-Execute
Агент сначала планирует шаги, потом выполняет по одному, затем синтезирует ответ.

---

## 📚 Дополнительные ресурсы

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph PyPI](https://pypi.org/project/langgraph/)

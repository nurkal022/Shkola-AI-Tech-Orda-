# Лекция 17: Интеграция инструментов — Внешние API и Веб-скрапинг

## Содержание

Сила агента заключается в его инструментах. Эта лекция посвящена практическим навыкам интеграции внешних API и веб-скрапинга, что позволяет агентам взаимодействовать с цифровым миром и получать доступ к информации в реальном времени.

---

## Файлы лекции

| Файл | Описание |
|------|----------|
| `00_live_demo_tools_integration.py` | **Live Demo** — Полная пошаговая демонстрация (8 этапов) |
| `00_live_demo_step_by_step.py` | **Live Demo** — Упрощённая версия для переписывания |
| `01_basic_api_call.py` | Базовые вызовы API (GET, POST, JSON, обработка ошибок) |
| `02_web_scraping_basics.py` | Основы веб-скрапинга с BeautifulSoup |
| `03_agent_with_api_tools.py` | LangChain агент с кастомными API-инструментами |
| `04_web_scraping_agent.py` | Агент с инструментами для веб-скрапинга |
| `05_weather_news_agent.py` | Практика: информационный агент (погода, новости, валюты) |
| `06_multi_tool_research_agent.py` | Практика: мульти-инструментальный исследователь |
| `07_search_agent.py` | Агент с поиском в интернете (DuckDuckGo + скрапинг) |

---

## Ключевые концепции

### Зачем агенту инструменты?

```
Без инструментов:
  Пользователь: "Какая погода в Астане?"
  Агент: "Я не могу получить актуальные данные..." ❌

С инструментами:
  Пользователь: "Какая погода в Астане?"
  Агент: [вызывает get_weather("Astana")]
  Агент: "Сейчас в Астане -5°C, облачно" ✅
```

### Два способа получения данных

```
1. API (Application Programming Interface)
   ├── Структурированные данные (JSON)
   ├── Официальный интерфейс сервиса
   ├── Стабильный и надёжный
   └── Примеры: Wikipedia API, GitHub API, wttr.in

2. Веб-скрапинг (Web Scraping)
   ├── Парсинг HTML-страниц
   ├── Когда нет API
   ├── Менее стабильный (HTML может измениться)
   └── Примеры: quotes.toscrape.com, books.toscrape.com
```

### Архитектура агента с инструментами

```
┌─────────────┐
│ Пользователь│
└──────┬──────┘
       │ "Какая погода?"
┌──────▼──────┐
│     LLM     │ ← Анализирует запрос
│  (GPT-4o)   │ ← Выбирает инструмент
└──────┬──────┘
       │ get_weather("Astana")
┌──────▼──────┐
│   @tool     │ ← Выполняет запрос
│ get_weather │ ← requests.get(...)
└──────┬──────┘
       │ "Температура: -5°C"
┌──────▼──────┐
│     LLM     │ ← Формирует ответ
└──────┬──────┘
       │ "В Астане сейчас -5°C..."
┌──────▼──────┐
│ Пользователь│
└─────────────┘
```

### Создание инструмента (@tool)

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Получает погоду для города.    # ← Описание для LLM

    Args:
        city: Город на английском      # ← Описание параметра
    """
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    data = response.json()
    return f"Температура: {data['current_condition'][0]['temp_C']}°C"
```

**Важно**: docstring — это инструкция для LLM. Чем лучше описание, тем точнее агент выбирает инструмент.

---

## Быстрый старт

```bash
pip install -r requirements.txt
```

Создайте `.env` в корне проекта:
```
OPENAI_API_KEY=your_key
```

### Запуск примеров

```bash
# Базовые API-вызовы (без OpenAI ключа)
python 01_basic_api_call.py

# Веб-скрапинг (без OpenAI ключа)
python 02_web_scraping_basics.py

# Агент с API инструментами
python 03_agent_with_api_tools.py

# Агент с веб-скрапингом
python 04_web_scraping_agent.py

# Информационный агент (погода + новости + валюты)
python 05_weather_news_agent.py

# Мульти-инструментальный исследователь
python 06_multi_tool_research_agent.py

# Поисковый агент (DuckDuckGo)
python 07_search_agent.py
```

---

## Используемые API (все бесплатные, без ключей)

| API | URL | Что даёт |
|-----|-----|----------|
| JSONPlaceholder | jsonplaceholder.typicode.com | Тестовый REST API |
| wttr.in | wttr.in | Погода без ключа |
| Wikipedia | en.wikipedia.org/api | Статьи и факты |
| GitHub | api.github.com | Репозитории |
| Hacker News | hacker-news.firebaseio.com | IT новости |
| Exchange Rates | open.er-api.com | Курсы валют |
| REST Countries | restcountries.com | Информация о странах |
| DuckDuckGo | duckduckgo-search (pip) | Поиск в интернете |

Для скрапинга:

| Сайт | URL | Что скрапим |
|------|-----|-------------|
| Quotes to Scrape | quotes.toscrape.com | Цитаты |
| Books to Scrape | books.toscrape.com | Каталог книг |

---

## Примеры использования

### Пример 1: Базовый API-вызов
GET/POST запросы, JSON-парсинг, query-параметры, обработка ошибок.

### Пример 2: Веб-скрапинг
BeautifulSoup, CSS-селекторы, извлечение текста/ссылок, пагинация.

### Пример 3: Агент с API
LangChain агент с инструментами для погоды, Wikipedia, GitHub.

### Пример 4: Агент-скрапер
Агент, извлекающий данные с веб-страниц через BeautifulSoup.

### Пример 5: Информационный агент
Комбинация погоды, новостей, курсов валют и данных о странах.

### Пример 6: Исследователь
Полноценный агент с 7 инструментами для глубокого исследования тем.

---

## Технические детали

### requests — HTTP-запросы

```python
import requests

# GET запрос
response = requests.get(url, params={"key": "value"}, timeout=10)
data = response.json()

# POST запрос
response = requests.post(url, json={"title": "Hello"})

# Обработка ошибок
response.raise_for_status()  # Исключение при 4xx/5xx
```

### BeautifulSoup — парсинг HTML

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "lxml")

# Поиск элементов
soup.find("div", class_="quote")       # Один элемент
soup.find_all("a", href=True)          # Все элементы
soup.select("div.quote > span.text")   # CSS-селектор

# Извлечение данных
element.get_text(strip=True)           # Текст
element["href"]                         # Атрибут
element.get("class", [])               # Атрибут с default
```

### @tool — создание инструментов

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Описание инструмента для LLM.

    Args:
        param: Описание параметра
    """
    # Логика инструмента
    return "результат"
```

---

## Дополнительные ресурсы

- [Requests Documentation](https://docs.python-requests.org/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)

---

## Связь с другими лекциями

- **Лекция 12**: Введение в AI-агенты (базовые концепции)
- **Лекция 13**: ReAct Framework (рассуждения + действия)
- **Лекция 14**: LangGraph (графы состояний)
- **Лекция 15**: AutoGen (мультиагентные системы)
- **Лекция 16**: CrewAI (командная работа агентов)
- **Лекция 17**: Интеграция инструментов ← *текущая*

---

## Практические задания

1. **Создайте свой инструмент:**
   - Выберите бесплатный API (список выше)
   - Создайте @tool функцию
   - Интегрируйте в агента

2. **Скрапинг-агент:**
   - Напишите скрапер для любого сайта
   - Оберните в @tool
   - Агент должен уметь скрапить по запросу

3. **Комбинированный агент:**
   - Минимум 3 инструмента (API + скрапинг)
   - Агент отвечает на составные вопросы
   - Сохранение результатов в файл

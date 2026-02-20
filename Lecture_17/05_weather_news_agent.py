"""
Лекция 17: Практический пример — Агент "Погода + Новости"
==========================================================
Агент, который комбинирует несколько API:
- Погода (wttr.in)
- Новости (RSS/Atom через скрапинг)
- Курсы валют (бесплатный API)
- Wikipedia для контекста
"""

import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Загружаем .env
load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден в .env файле")
    exit(1)

# ============================================================
# 1. Определяем набор инструментов
# ============================================================

print("=" * 60)
print("🔧 ОПРЕДЕЛЯЕМ НАБОР ИНСТРУМЕНТОВ")
print("=" * 60)
print()


@tool
def get_weather_detailed(city: str) -> str:
    """Получает подробный прогноз погоды для города.
    Включает текущую погоду и прогноз на 3 дня.

    Args:
        city: Название города на английском (например 'Astana', 'Moscow', 'London')
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        data = response.json()

        current = data["current_condition"][0]
        result = f"=== Погода в {city} ===\n\n"
        result += f"Сейчас:\n"
        result += f"  Температура: {current['temp_C']}°C (ощущается как {current['FeelsLikeC']}°C)\n"
        result += f"  Ветер: {current['windspeedKmph']} км/ч, направление {current['winddir16Point']}\n"
        result += f"  Влажность: {current['humidity']}%\n"
        result += f"  Давление: {current['pressure']} мбар\n"
        result += f"  Описание: {current['weatherDesc'][0]['value']}\n"
        result += f"  Видимость: {current['visibility']} км\n\n"

        # Прогноз на 3 дня
        result += "Прогноз:\n"
        for day in data.get("weather", [])[:3]:
            date = day["date"]
            max_temp = day["maxtempC"]
            min_temp = day["mintempC"]
            desc = day["hourly"][4]["weatherDesc"][0]["value"]  # ~12:00
            result += f"  {date}: {min_temp}°C — {max_temp}°C, {desc}\n"

        return result
    except Exception as e:
        return f"Ошибка получения погоды для {city}: {e}"


@tool
def get_tech_news() -> str:
    """Получает последние технологические новости из Hacker News.
    Используй когда пользователь спрашивает о новостях в IT/технологиях.
    """
    try:
        # Hacker News API — бесплатный, без ключа
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        response = requests.get(url, timeout=10)
        story_ids = response.json()[:10]  # Топ 10 новостей

        result = "=== Топ новости Hacker News ===\n\n"
        for i, story_id in enumerate(story_ids, 1):
            story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            story = requests.get(story_url, timeout=5).json()

            title = story.get("title", "N/A")
            link = story.get("url", "N/A")
            score = story.get("score", 0)
            comments = story.get("descendants", 0)

            result += f"{i}. {title}\n"
            result += f"   Score: {score} | Комментарии: {comments}\n"
            if link != "N/A":
                result += f"   URL: {link}\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Ошибка получения новостей: {e}"


@tool
def get_exchange_rates(base_currency: str = "USD") -> str:
    """Получает текущие курсы валют.
    Показывает курсы основных валют относительно базовой.

    Args:
        base_currency: Базовая валюта (USD, EUR, KZT, RUB и др.)
    """
    try:
        # Бесплатный API курсов валют (без ключа)
        url = f"https://open.er-api.com/v6/latest/{base_currency}"
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("result") != "success":
            return f"Ошибка: не удалось получить курсы для {base_currency}"

        rates = data["rates"]
        important_currencies = ["USD", "EUR", "KZT", "RUB", "GBP", "CNY", "JPY", "TRY"]

        result = f"=== Курсы валют (базовая: {base_currency}) ===\n"
        result += f"Дата обновления: {data.get('time_last_update_utc', 'N/A')}\n\n"

        for currency in important_currencies:
            if currency in rates and currency != base_currency:
                rate = rates[currency]
                result += f"  1 {base_currency} = {rate:.4f} {currency}\n"

        return result
    except Exception as e:
        return f"Ошибка получения курсов: {e}"


@tool
def get_country_info(country: str) -> str:
    """Получает информацию о стране (столица, население, валюта, языки).
    Используй когда пользователь спрашивает о стране.

    Args:
        country: Название страны на английском (например 'Kazakhstan', 'Japan', 'France')
    """
    try:
        url = f"https://restcountries.com/v3.1/name/{country}"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return f"Страна '{country}' не найдена"

        data = response.json()[0]

        name = data.get("name", {}).get("common", country)
        capital = ", ".join(data.get("capital", ["N/A"]))
        population = data.get("population", 0)
        region = data.get("region", "N/A")
        subregion = data.get("subregion", "N/A")
        languages = ", ".join(data.get("languages", {}).values())
        currencies = ", ".join(
            f"{v['name']} ({v['symbol']})"
            for v in data.get("currencies", {}).values()
        )
        timezones = ", ".join(data.get("timezones", [])[:3])

        result = f"=== {name} ===\n"
        result += f"  Столица: {capital}\n"
        result += f"  Население: {population:,}\n"
        result += f"  Регион: {region} / {subregion}\n"
        result += f"  Языки: {languages}\n"
        result += f"  Валюта: {currencies}\n"
        result += f"  Часовые пояса: {timezones}\n"

        return result
    except Exception as e:
        return f"Ошибка: {e}"


tools = [get_weather_detailed, get_tech_news, get_exchange_rates, get_country_info]

print("✅ Инструменты определены:")
for t in tools:
    print(f"   🔧 {t.name}: {t.description.split(chr(10))[0]}")
print()

# ============================================================
# 2. Создаём информационного агента
# ============================================================

print("=" * 60)
print("🤖 СОЗДАЁМ ИНФОРМАЦИОННОГО АГЕНТА")
print("=" * 60)
print()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты информационный AI-ассистент с доступом к реальным данным. "
        "У тебя есть инструменты для получения погоды, новостей, курсов валют и информации о странах. "
        "Всегда отвечай на русском языке. "
        "Предоставляй структурированные и полезные ответы. "
        "Можешь комбинировать несколько инструментов для полного ответа."
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("✅ Информационный агент создан")
print()

# ============================================================
# 3. Тестовые запросы
# ============================================================

print("=" * 60)
print("🧪 ТЕСТИРУЕМ ИНФОРМАЦИОННОГО АГЕНТА")
print("=" * 60)
print()

# Запрос 1: Погода + информация о стране
print("─" * 60)
print("📝 Запрос 1: 'Расскажи о Казахстане и какая там погода'")
print("─" * 60)
result = agent_executor.invoke({
    "input": "Расскажи о Казахстане — основная информация о стране и текущая погода в Астане"
})
print()
print(f"🤖 Ответ:\n{result['output']}")
print()

# Запрос 2: Новости технологий
print("─" * 60)
print("📝 Запрос 2: 'Какие сейчас топ новости в IT?'")
print("─" * 60)
result = agent_executor.invoke({
    "input": "Какие сейчас самые горячие новости в мире технологий?"
})
print()
print(f"🤖 Ответ:\n{result['output'][:800]}")
print()

# Запрос 3: Курсы валют
print("─" * 60)
print("📝 Запрос 3: 'Какой сейчас курс доллара к тенге?'")
print("─" * 60)
result = agent_executor.invoke({
    "input": "Какой сейчас курс доллара к тенге и другим валютам?"
})
print()
print(f"🤖 Ответ:\n{result['output']}")
print()

# ============================================================
# Итоги
# ============================================================

print("=" * 60)
print("✅ ИТОГИ: ИНФОРМАЦИОННЫЙ АГЕНТ")
print("=" * 60)
print()
print("📚 Что мы создали:")
print("   1. get_weather_detailed — подробный прогноз погоды")
print("   2. get_tech_news — топ новости Hacker News")
print("   3. get_exchange_rates — курсы валют в реальном времени")
print("   4. get_country_info — информация о странах")
print()
print("💡 Агент умеет:")
print("   • Комбинировать несколько инструментов")
print("   • Выбирать подходящий инструмент по запросу")
print("   • Предоставлять актуальную информацию")
print("   • Отвечать на сложные составные вопросы")
print()

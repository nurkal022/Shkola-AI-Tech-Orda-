"""
Лекция 17: Агент с веб-скрапингом
===================================
LangChain агент, который умеет скрапить веб-страницы
и извлекать из них структурированную информацию.
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
# 1. Инструменты для веб-скрапинга
# ============================================================

print("=" * 60)
print("🔧 1. ИНСТРУМЕНТЫ ДЛЯ ВЕБ-СКРАПИНГА")
print("=" * 60)
print()


@tool
def scrape_webpage(url: str) -> str:
    """Скрапит веб-страницу и возвращает основной текстовый контент.
    Используй этот инструмент когда нужно получить содержимое веб-страницы.

    Args:
        url: Полный URL страницы, например 'https://example.com'
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Educational Bot) ScrapingDemo/1.0"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Удаляем скрипты и стили
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Извлекаем текст
        text = soup.get_text(separator="\n", strip=True)

        # Ограничиваем длину
        if len(text) > 2000:
            text = text[:2000] + "\n\n...[текст обрезан]..."

        return f"Содержимое {url}:\n\n{text}"
    except Exception as e:
        return f"Ошибка скрапинга {url}: {e}"


@tool
def scrape_quotes(page: int = 1) -> str:
    """Скрапит цитаты с сайта quotes.toscrape.com.
    Возвращает список цитат с авторами и тегами.

    Args:
        page: Номер страницы (1-10)
    """
    try:
        url = f"https://quotes.toscrape.com/page/{page}/"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")

        quotes = soup.find_all("div", class_="quote")

        if not quotes:
            return f"Страница {page} не содержит цитат"

        result = f"Цитаты (страница {page}):\n\n"
        for i, quote in enumerate(quotes, 1):
            text = quote.find("span", class_="text").get_text()
            author = quote.find("small", class_="author").get_text()
            tags = [t.get_text() for t in quote.find_all("a", class_="tag")]
            result += f"{i}. {text}\n   — {author}\n   Теги: {', '.join(tags)}\n\n"

        return result
    except Exception as e:
        return f"Ошибка: {e}"


@tool
def scrape_books(category: str = "") -> str:
    """Скрапит каталог книг с books.toscrape.com.
    Возвращает список книг с названиями, ценами и рейтингами.

    Args:
        category: Категория книг (пустая строка = все книги). Примеры: 'travel', 'mystery', 'science'
    """
    try:
        if category:
            # Ищем нужную категорию
            main_url = "https://books.toscrape.com/"
            response = requests.get(main_url, timeout=10)
            soup = BeautifulSoup(response.text, "lxml")

            # Находим ссылку на категорию
            cat_links = soup.select("ul.nav-list ul a")
            cat_url = None
            for link in cat_links:
                if category.lower() in link.get_text(strip=True).lower():
                    cat_url = main_url + link["href"]
                    break

            if not cat_url:
                available = [l.get_text(strip=True) for l in cat_links]
                return f"Категория '{category}' не найдена. Доступные: {', '.join(available[:10])}"

            url = cat_url
        else:
            url = "https://books.toscrape.com/"

        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")

        books = soup.find_all("article", class_="product_pod")

        rating_map = {"One": "⭐", "Two": "⭐⭐", "Three": "⭐⭐⭐", "Four": "⭐⭐⭐⭐", "Five": "⭐⭐⭐⭐⭐"}

        result = f"Каталог книг ({category or 'все категории'}):\n\n"
        for i, book in enumerate(books[:10], 1):
            title = book.find("h3").find("a")["title"]
            price = book.find("p", class_="price_color").get_text()
            rating_class = book.find("p", class_="star-rating")["class"][1]
            rating = rating_map.get(rating_class, rating_class)

            result += f"{i}. {title}\n   Цена: {price} | Рейтинг: {rating}\n\n"

        return result
    except Exception as e:
        return f"Ошибка: {e}"


@tool
def extract_links(url: str) -> str:
    """Извлекает все ссылки с веб-страницы.
    Используй когда нужно найти ссылки на странице.

    Args:
        url: Полный URL страницы
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if text and href.startswith("http"):
                links.append(f"[{text[:50]}] → {href}")

        if not links:
            return f"На странице {url} не найдено внешних ссылок"

        result = f"Ссылки на {url}:\n\n"
        for link in links[:20]:
            result += f"• {link}\n"

        if len(links) > 20:
            result += f"\n...и ещё {len(links) - 20} ссылок"

        return result
    except Exception as e:
        return f"Ошибка: {e}"


tools = [scrape_webpage, scrape_quotes, scrape_books, extract_links]

print("✅ Инструменты скрапинга определены:")
for t in tools:
    print(f"   🔧 {t.name}")
print()

# ============================================================
# 2. Создаём агента-скрапера
# ============================================================

print("=" * 60)
print("🤖 2. СОЗДАЁМ АГЕНТА-СКРАПЕРА")
print("=" * 60)
print()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты AI-ассистент, специализирующийся на сборе информации из интернета. "
        "У тебя есть инструменты для скрапинга веб-страниц. "
        "Используй их для получения актуальной информации. "
        "Отвечай на русском языке. "
        "Анализируй полученные данные и предоставляй структурированные ответы."
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("✅ Агент-скрапер создан")
print()

# ============================================================
# 3. Тестируем агента
# ============================================================

print("=" * 60)
print("🧪 3. ТЕСТИРУЕМ АГЕНТА-СКРАПЕРА")
print("=" * 60)
print()

# Запрос 1: Скрапинг цитат
print("─" * 60)
print("📝 Запрос 1: 'Покажи цитаты со страницы 2'")
print("─" * 60)
result = agent_executor.invoke({"input": "Покажи мне цитаты со страницы 2 сайта quotes.toscrape.com"})
print()
print(f"🤖 Ответ:\n{result['output'][:500]}")
print()

# Запрос 2: Скрапинг книг по категории
print("─" * 60)
print("📝 Запрос 2: 'Какие книги есть в категории Science?'")
print("─" * 60)
result = agent_executor.invoke({"input": "Какие книги есть в категории Science на books.toscrape.com?"})
print()
print(f"🤖 Ответ:\n{result['output'][:500]}")
print()

# Запрос 3: Скрапинг произвольной страницы
print("─" * 60)
print("📝 Запрос 3: 'Получи содержимое example.com'")
print("─" * 60)
result = agent_executor.invoke({"input": "Что написано на странице https://example.com?"})
print()
print(f"🤖 Ответ:\n{result['output'][:500]}")
print()

# ============================================================
# Итоги
# ============================================================

print("=" * 60)
print("✅ ИТОГИ: АГЕНТ С ВЕБ-СКРАПИНГОМ")
print("=" * 60)
print()
print("📚 Что мы изучили:")
print("   1. Инструменты скрапинга для агента")
print("   2. scrape_webpage — универсальный скрапер")
print("   3. Специализированные скраперы (quotes, books)")
print("   4. Извлечение ссылок (extract_links)")
print("   5. Агент выбирает подходящий скрапер")
print()
print("💡 Важно при скрапинге:")
print("   • User-Agent header — идентифицируй бота")
print("   • Timeout — не зависай на медленных сайтах")
print("   • Ограничение длины — не перегружай LLM")
print("   • Удаление script/style — чистый текст")
print()

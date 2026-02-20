"""
Тесты для Лекции 17: Интеграция инструментов
==============================================
Проверяет работоспособность API-вызовов и скрапинга.
Запуск: python test_all.py
"""

import sys
import requests
from bs4 import BeautifulSoup


def test_jsonplaceholder_api():
    """Тест: JSONPlaceholder API доступен"""
    print("🧪 Тест 1: JSONPlaceholder API...", end=" ")
    try:
        response = requests.get(
            "https://jsonplaceholder.typicode.com/posts/1", timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "body" in data
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_wikipedia_api():
    """Тест: Wikipedia API доступен"""
    print("🧪 Тест 2: Wikipedia API...", end=" ")
    try:
        headers = {"User-Agent": "TechOrda-Education/1.0 (Python requests)"}
        response = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/Python_(programming_language)",
            headers=headers,
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "extract" in data
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_weather_api():
    """Тест: wttr.in API доступен"""
    print("🧪 Тест 3: wttr.in API (погода)...", end=" ")
    try:
        response = requests.get("https://wttr.in/London?format=j1", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "current_condition" in data
        assert "temp_C" in data["current_condition"][0]
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_github_api():
    """Тест: GitHub API доступен"""
    print("🧪 Тест 4: GitHub API...", end=" ")
    try:
        response = requests.get(
            "https://api.github.com/repos/langchain-ai/langchain", timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert "full_name" in data
        assert "stargazers_count" in data
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_exchange_rate_api():
    """Тест: Exchange Rate API доступен"""
    print("🧪 Тест 5: Exchange Rate API...", end=" ")
    try:
        response = requests.get(
            "https://open.er-api.com/v6/latest/USD", timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("result") == "success"
        assert "KZT" in data.get("rates", {})
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_hacker_news_api():
    """Тест: Hacker News API доступен"""
    print("🧪 Тест 6: Hacker News API...", end=" ")
    try:
        response = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_quotes_scraping():
    """Тест: quotes.toscrape.com скрапинг работает"""
    print("🧪 Тест 7: Скрапинг quotes.toscrape.com...", end=" ")
    try:
        response = requests.get("https://quotes.toscrape.com/", timeout=10)
        assert response.status_code == 200
        soup = BeautifulSoup(response.text, "lxml")
        quotes = soup.find_all("div", class_="quote")
        assert len(quotes) > 0
        # Проверяем структуру
        first_quote = quotes[0]
        assert first_quote.find("span", class_="text") is not None
        assert first_quote.find("small", class_="author") is not None
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_books_scraping():
    """Тест: books.toscrape.com скрапинг работает"""
    print("🧪 Тест 8: Скрапинг books.toscrape.com...", end=" ")
    try:
        response = requests.get("https://books.toscrape.com/", timeout=10)
        assert response.status_code == 200
        soup = BeautifulSoup(response.text, "lxml")
        books = soup.find_all("article", class_="product_pod")
        assert len(books) > 0
        # Проверяем структуру
        first_book = books[0]
        assert first_book.find("h3") is not None
        assert first_book.find("p", class_="price_color") is not None
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_imports():
    """Тест: Все необходимые библиотеки установлены"""
    print("🧪 Тест 9: Импорты библиотек...", end=" ")
    try:
        import requests
        import bs4
        import lxml
        import langchain
        import langchain_openai
        print("✅ OK")
        return True
    except ImportError as e:
        print(f"❌ FAIL: {e}")
        print(f"   Установите: pip install -r requirements.txt")
        return False


def test_openai_key():
    """Тест: OPENAI_API_KEY настроен"""
    print("🧪 Тест 10: OPENAI_API_KEY...", end=" ")
    import os
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OK")
        return True
    else:
        print("⚠️ НЕ НАЙДЕН (нужен для файлов 03-06)")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ТЕСТЫ ЛЕКЦИИ 17: Интеграция инструментов")
    print("=" * 60)
    print()

    tests = [
        test_imports,
        test_openai_key,
        test_jsonplaceholder_api,
        test_wikipedia_api,
        test_weather_api,
        test_github_api,
        test_exchange_rate_api,
        test_hacker_news_api,
        test_quotes_scraping,
        test_books_scraping,
    ]

    results = []
    for test_func in tests:
        results.append(test_func())

    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"📊 Результаты: {passed}/{total} тестов пройдено")

    if passed == total:
        print("✅ Все тесты пройдены! Можно запускать примеры.")
    elif passed >= total - 2:
        print("⚠️ Почти все тесты пройдены. Некоторые API могут быть временно недоступны.")
    else:
        print("❌ Есть проблемы. Проверьте подключение к интернету и установку библиотек.")

    print("=" * 60)
    sys.exit(0 if passed >= total - 2 else 1)

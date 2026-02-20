"""
Лекция 17: Основы веб-скрапинга
=================================
Парсинг HTML-страниц с BeautifulSoup.
Извлечение текста, ссылок, таблиц из веб-страниц.
"""

import requests
from bs4 import BeautifulSoup

# ============================================================
# 1. Базовый скрапинг — получение HTML и парсинг
# ============================================================

print("=" * 60)
print("🕷️ 1. БАЗОВЫЙ СКРАПИНГ — ПОЛУЧЕНИЕ HTML")
print("=" * 60)
print()

# quotes.toscrape.com — специальный сайт для практики скрапинга
url = "https://quotes.toscrape.com/"

print(f"🔗 URL: {url}")
print("⏳ Загружаем страницу...")
print()

response = requests.get(url)
print(f"📊 Статус: {response.status_code}")
print(f"📋 Размер HTML: {len(response.text)} символов")
print()

# Создаём объект BeautifulSoup
soup = BeautifulSoup(response.text, "lxml")

print(f"📄 Заголовок страницы: {soup.title.string}")
print()

# ============================================================
# 2. Извлечение цитат (CSS-селекторы)
# ============================================================

print("=" * 60)
print("🕷️ 2. ИЗВЛЕЧЕНИЕ ЦИТАТ (CSS-СЕЛЕКТОРЫ)")
print("=" * 60)
print()

# Находим все блоки с цитатами
quotes = soup.find_all("div", class_="quote")

print(f"📊 Найдено цитат: {len(quotes)}")
print()

for i, quote in enumerate(quotes[:5], 1):
    text = quote.find("span", class_="text").get_text()
    author = quote.find("small", class_="author").get_text()
    tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]

    print(f"   {i}. {text[:60]}...")
    print(f"      — {author}")
    print(f"      🏷️ Теги: {', '.join(tags)}")
    print()

# ============================================================
# 3. Извлечение ссылок
# ============================================================

print("=" * 60)
print("🕷️ 3. ИЗВЛЕЧЕНИЕ ССЫЛОК")
print("=" * 60)
print()

# Находим все ссылки на странице
links = soup.find_all("a")
print(f"📊 Всего ссылок: {len(links)}")
print()

print("🔗 Первые 10 ссылок:")
for link in links[:10]:
    href = link.get("href", "N/A")
    text = link.get_text(strip=True)
    if text:
        print(f"   • [{text}] → {href}")
print()

# ============================================================
# 4. Скрапинг таблиц — Books to Scrape
# ============================================================

print("=" * 60)
print("🕷️ 4. СКРАПИНГ КАТАЛОГА КНИГ")
print("=" * 60)
print()

url = "https://books.toscrape.com/"
print(f"🔗 URL: {url}")
print("⏳ Загружаем каталог книг...")
print()

response = requests.get(url)
soup = BeautifulSoup(response.text, "lxml")

# Извлекаем книги
books = soup.find_all("article", class_="product_pod")
print(f"📊 Найдено книг: {len(books)}")
print()

print("📚 Первые 5 книг:")
for i, book in enumerate(books[:5], 1):
    title = book.find("h3").find("a")["title"]
    price = book.find("p", class_="price_color").get_text()
    rating = book.find("p", class_="star-rating")["class"][1]

    print(f"   {i}. {title[:50]}")
    print(f"      💰 Цена: {price} | ⭐ Рейтинг: {rating}")
    print()

# ============================================================
# 5. Извлечение структурированных данных
# ============================================================

print("=" * 60)
print("🕷️ 5. СТРУКТУРИРОВАННЫЕ ДАННЫЕ")
print("=" * 60)
print()

# Возвращаемся к цитатам и собираем структурированные данные
url = "https://quotes.toscrape.com/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "lxml")

# Собираем все данные в список словарей
structured_data = []
for quote in soup.find_all("div", class_="quote"):
    item = {
        "text": quote.find("span", class_="text").get_text(),
        "author": quote.find("small", class_="author").get_text(),
        "tags": [tag.get_text() for tag in quote.find_all("a", class_="tag")],
    }
    structured_data.append(item)

print(f"📊 Собрано записей: {len(structured_data)}")
print()

# Статистика по авторам
authors = {}
for item in structured_data:
    author = item["author"]
    authors[author] = authors.get(author, 0) + 1

print("👤 Статистика по авторам:")
for author, count in sorted(authors.items(), key=lambda x: -x[1]):
    print(f"   • {author}: {count} цитат(ы)")
print()

# Статистика по тегам
all_tags = {}
for item in structured_data:
    for tag in item["tags"]:
        all_tags[tag] = all_tags.get(tag, 0) + 1

print("🏷️ Топ-5 тегов:")
for tag, count in sorted(all_tags.items(), key=lambda x: -x[1])[:5]:
    print(f"   • {tag}: {count}")
print()

# ============================================================
# 6. Навигация по страницам (пагинация)
# ============================================================

print("=" * 60)
print("🕷️ 6. НАВИГАЦИЯ ПО СТРАНИЦАМ (ПАГИНАЦИЯ)")
print("=" * 60)
print()

all_quotes = []
base_url = "https://quotes.toscrape.com"

# Скрапим первые 3 страницы
for page in range(1, 4):
    url = f"{base_url}/page/{page}/"
    print(f"   📄 Загружаем страницу {page}... ({url})")

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "lxml")

    quotes = soup.find_all("div", class_="quote")
    for quote in quotes:
        all_quotes.append({
            "text": quote.find("span", class_="text").get_text(),
            "author": quote.find("small", class_="author").get_text(),
        })

    print(f"      ✅ Найдено цитат: {len(quotes)}")

print()
print(f"📊 Всего собрано цитат: {len(all_quotes)}")
print()

# Показываем уникальных авторов
unique_authors = set(q["author"] for q in all_quotes)
print(f"👤 Уникальных авторов: {len(unique_authors)}")
for author in sorted(unique_authors):
    print(f"   • {author}")
print()

# ============================================================
# 7. Функция-скрапер (переиспользуемый код)
# ============================================================

print("=" * 60)
print("🕷️ 7. ФУНКЦИЯ-СКРАПЕР (ПЕРЕИСПОЛЬЗУЕМЫЙ КОД)")
print("=" * 60)
print()


def scrape_quotes(max_pages=2):
    """
    Скрапит цитаты с quotes.toscrape.com.
    Возвращает список словарей с цитатами.
    """
    all_data = []
    base_url = "https://quotes.toscrape.com"

    for page in range(1, max_pages + 1):
        url = f"{base_url}/page/{page}/"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"   ⚠️ Ошибка на странице {page}: {e}")
            break

        soup = BeautifulSoup(response.text, "lxml")
        quotes = soup.find_all("div", class_="quote")

        if not quotes:
            break  # Больше страниц нет

        for quote in quotes:
            all_data.append({
                "text": quote.find("span", class_="text").get_text(),
                "author": quote.find("small", class_="author").get_text(),
                "tags": [t.get_text() for t in quote.find_all("a", class_="tag")],
            })

    return all_data


# Используем функцию
print("💡 Используем функцию scrape_quotes():")
quotes_data = scrape_quotes(max_pages=2)
print(f"   📊 Собрано: {len(quotes_data)} цитат")
print(f"   📄 Пример: {quotes_data[0]['text'][:50]}...")
print(f"   👤 Автор: {quotes_data[0]['author']}")
print()

# ============================================================
# Итоги
# ============================================================

print("=" * 60)
print("✅ ИТОГИ: ОСНОВЫ ВЕБ-СКРАПИНГА")
print("=" * 60)
print()
print("📚 Что мы изучили:")
print("   1. BeautifulSoup — парсинг HTML")
print("   2. find() / find_all() — поиск элементов")
print("   3. CSS-селекторы — class_, id, tag")
print("   4. get_text() — извлечение текста")
print("   5. Пагинация — навигация по страницам")
print("   6. Структурированные данные — сбор в словари")
print()
print("💡 Этичный скрапинг:")
print("   • Проверяйте robots.txt сайта")
print("   • Не перегружайте сервер (добавляйте задержки)")
print("   • Используйте API если они доступны")
print("   • Соблюдайте условия использования сайта")
print()

"""
Лекция 17: Базовые вызовы API
==============================
Основы работы с внешними API через библиотеку requests.
Демонстрация GET, POST запросов и обработки JSON-ответов.
"""

import requests
import json

# ============================================================
# 1. Простой GET-запрос (JSONPlaceholder - бесплатный тестовый API)
# ============================================================

print("=" * 60)
print("📡 1. ПРОСТОЙ GET-ЗАПРОС")
print("=" * 60)
print()

# JSONPlaceholder — бесплатный фейковый REST API для тестирования
url = "https://jsonplaceholder.typicode.com/posts/1"

print(f"🔗 URL: {url}")
print("⏳ Отправляем GET запрос...")
print()

response = requests.get(url)

print(f"📊 Статус код: {response.status_code}")
print(f"📋 Content-Type: {response.headers.get('Content-Type')}")
print()

# Парсим JSON-ответ
data = response.json()

print("📦 Полученные данные:")
print(f"   ID: {data['id']}")
print(f"   User ID: {data['userId']}")
print(f"   Title: {data['title'][:50]}...")
print(f"   Body: {data['body'][:80]}...")
print()

# ============================================================
# 2. GET-запрос со списком данных
# ============================================================

print("=" * 60)
print("📡 2. GET-ЗАПРОС СО СПИСКОМ ДАННЫХ")
print("=" * 60)
print()

url = "https://jsonplaceholder.typicode.com/users"
print(f"🔗 URL: {url}")
print("⏳ Получаем список пользователей...")
print()

response = requests.get(url)
users = response.json()

print(f"📊 Получено пользователей: {len(users)}")
print()
print("👥 Первые 5 пользователей:")
for user in users[:5]:
    print(f"   • {user['name']} ({user['email']}) — {user['company']['name']}")
print()

# ============================================================
# 3. GET-запрос с параметрами (Query Parameters)
# ============================================================

print("=" * 60)
print("📡 3. GET-ЗАПРОС С ПАРАМЕТРАМИ")
print("=" * 60)
print()

url = "https://jsonplaceholder.typicode.com/posts"
params = {
    "userId": 1,
    "_limit": 3,
}

print(f"🔗 URL: {url}")
print(f"📋 Параметры: {params}")
print("⏳ Отправляем запрос с параметрами...")
print()

response = requests.get(url, params=params)
posts = response.json()

print(f"📊 Полный URL: {response.url}")
print(f"📊 Получено постов: {len(posts)}")
print()
for post in posts:
    print(f"   📝 [{post['id']}] {post['title'][:50]}...")
print()

# ============================================================
# 4. POST-запрос (отправка данных)
# ============================================================

print("=" * 60)
print("📡 4. POST-ЗАПРОС (ОТПРАВКА ДАННЫХ)")
print("=" * 60)
print()

url = "https://jsonplaceholder.typicode.com/posts"

# Данные для отправки
new_post = {
    "title": "Мой новый пост об AI агентах",
    "body": "AI агенты могут использовать внешние API для получения данных в реальном времени",
    "userId": 1,
}

print(f"🔗 URL: {url}")
print(f"📋 Данные: {json.dumps(new_post, ensure_ascii=False, indent=2)}")
print("⏳ Отправляем POST запрос...")
print()

response = requests.post(url, json=new_post)

print(f"📊 Статус код: {response.status_code}")  # 201 = Created
created = response.json()
print(f"✅ Создан пост с ID: {created['id']}")
print()

# ============================================================
# 5. Wikipedia API — получение реальных данных
# ============================================================

print("=" * 60)
print("📡 5. WIKIPEDIA API — РЕАЛЬНЫЕ ДАННЫЕ")
print("=" * 60)
print()

url = "https://en.wikipedia.org/api/rest_v1/page/summary/Artificial_intelligence"

print(f"🔗 URL: {url}")
print("⏳ Получаем статью из Wikipedia...")
print()

try:
    headers = {"User-Agent": "TechOrda-Education/1.0 (Python requests)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    wiki_data = response.json()

    print(f"📖 Заголовок: {wiki_data['title']}")
    print(f"📝 Описание: {wiki_data.get('description', 'N/A')}")
    print()
    print("📄 Краткое содержание:")
    extract = wiki_data.get("extract", "N/A")
    # Показываем первые 300 символов
    print(f"   {extract[:300]}...")
except Exception as e:
    print(f"⚠️ Wikipedia API временно недоступен: {e}")
    print("   (это нормально — бесплатные API могут быть нестабильны)")
print()

# ============================================================
# 6. Погода — wttr.in (бесплатный API без ключа)
# ============================================================

print("=" * 60)
print("📡 6. ПОГОДА — WTTR.IN (БЕЗ API КЛЮЧА)")
print("=" * 60)
print()

city = "Astana"
url = f"https://wttr.in/{city}?format=j1"

print(f"🔗 URL: {url}")
print(f"🏙️ Город: {city}")
print("⏳ Получаем погоду...")
print()

try:
    response = requests.get(url, timeout=10)
    weather = response.json()

    current = weather["current_condition"][0]
    print(f"🌡️ Температура: {current['temp_C']}°C")
    print(f"💨 Ветер: {current['windspeedKmph']} км/ч")
    print(f"💧 Влажность: {current['humidity']}%")
    print(f"☁️ Описание: {current['weatherDesc'][0]['value']}")
except Exception as e:
    print(f"⚠️ Ошибка: {e}")
    print("   (wttr.in может быть временно недоступен)")

print()

# ============================================================
# 7. Обработка ошибок API
# ============================================================

print("=" * 60)
print("📡 7. ОБРАБОТКА ОШИБОК API")
print("=" * 60)
print()

# Запрос к несуществующему ресурсу
url = "https://jsonplaceholder.typicode.com/posts/99999"

print(f"🔗 URL: {url}")
print("⏳ Запрашиваем несуществующий ресурс...")
print()

response = requests.get(url)
print(f"📊 Статус код: {response.status_code}")

# Проверяем статус-код
if response.status_code == 200:
    print("✅ Успех!")
elif response.status_code == 404:
    print("❌ Ресурс не найден (404)")
elif response.status_code >= 500:
    print("❌ Ошибка сервера (5xx)")
print()

# Правильная обработка ошибок
print("💡 Правильный подход — использовать try/except:")
print()

try:
    response = requests.get("https://httpbin.org/status/500", timeout=5)
    response.raise_for_status()  # Выбросит исключение для 4xx/5xx
    print("✅ Запрос успешен")
except requests.exceptions.HTTPError as e:
    print(f"❌ HTTP ошибка: {e}")
except requests.exceptions.ConnectionError:
    print("❌ Ошибка соединения")
except requests.exceptions.Timeout:
    print("❌ Таймаут запроса")
except requests.exceptions.RequestException as e:
    print(f"❌ Общая ошибка: {e}")

print()

# ============================================================
# Итоги
# ============================================================

print("=" * 60)
print("✅ ИТОГИ: БАЗОВЫЕ ВЫЗОВЫ API")
print("=" * 60)
print()
print("📚 Что мы изучили:")
print("   1. GET-запрос — получение данных")
print("   2. POST-запрос — отправка данных")
print("   3. Query параметры — фильтрация данных")
print("   4. JSON парсинг — работа с ответами")
print("   5. Реальные API — Wikipedia, wttr.in")
print("   6. Обработка ошибок — try/except + status codes")
print()
print("💡 Ключевые функции requests:")
print("   • requests.get(url, params=...) — GET запрос")
print("   • requests.post(url, json=...) — POST запрос")
print("   • response.json() — парсинг JSON")
print("   • response.status_code — код статуса")
print("   • response.raise_for_status() — исключение при ошибке")
print()

# Лекция 22: Создание производственного API с FastAPI

## Содержание

| # | Файл | Тема |
|---|------|------|
| 0 | `00_fastapi_basics.py` | Основы FastAPI: маршруты, Pydantic, валидация, Swagger |
| 1 | `01_ai_chat_api.py` | ИИ-чат API: OpenAI, async, lifespan, обработка ошибок |
| 2 | `02_streaming.py` | Стриминг ответов (SSE), демо-страница в браузере |
| 3 | `03_async_tasks.py` | Фоновые задачи, batch-обработка, asyncio.gather |
| 4 | `04_production_ready.py` | Продакшен: rate limiting, auth, CORS, middleware, логи |
| 5 | `05_full_project.py` | Полный проект — всё в одном с демо UI |

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
# Любой файл запускается отдельно:
python 00_fastapi_basics.py    # порт 8000
python 01_ai_chat_api.py       # порт 8001
python 02_streaming.py         # порт 8002
python 03_async_tasks.py       # порт 8003
python 04_production_ready.py  # порт 8004
python 05_full_project.py      # порт 8000 (полный проект)
```

## Демо

```bash
# Полный проект с UI
python 05_full_project.py
# Откройте http://localhost:8000/demo
```

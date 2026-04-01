# Лекция 23: Контейнеризация с Docker, CI/CD и облачное развертывание

## Содержание

| Файл | Описание |
|------|----------|
| `00_docker_basics.py` | Теория Docker: контейнеры, образы, команды, best practices |
| `01_github_actions.yml` | CI/CD пайплайн: тесты -> сборка -> деплой |
| `02_deploy_guide.py` | Гайд по деплою: Railway, Render, Fly.io |
| `app/main.py` | FastAPI приложение (адаптировано для Docker) |
| `Dockerfile` | Сборка контейнера |
| `docker-compose.yml` | Запуск через Docker Compose |
| `.dockerignore` | Исключения при сборке |
| `tests/test_api.py` | Тесты для CI/CD |

---

## Быстрый старт

### 1. Без Docker (локально)

```bash
cd Lecture_23
pip install -r app/requirements.txt
export OPENAI_API_KEY=sk-xxx
python app/main.py
# -> http://localhost:8000/docs
```

### 2. С Docker

```bash
cd Lecture_23

# Сборка
docker build -t ai-chat-api .

# Запуск
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx ai-chat-api

# -> http://localhost:8000/docs
```

### 3. С Docker Compose

```bash
cd Lecture_23
export OPENAI_API_KEY=sk-xxx

docker compose up --build
# -> http://localhost:8000/docs
```

---

## Архитектура

```
┌──────────────────────────────────────────────┐
│              Docker Container                │
│  ┌────────────────────────────────────────┐  │
│  │  Python 3.12-slim                      │  │
│  │  ┌──────────────────────────────────┐  │  │
│  │  │  FastAPI + Uvicorn               │  │  │
│  │  │  /health   → Health check        │  │  │
│  │  │  /chat     → AI чат              │  │  │
│  │  │  /chat/stream → SSE стриминг     │  │  │
│  │  └──────────────────────────────────┘  │  │
│  │  OpenAI SDK                            │  │
│  └────────────────────────────────────────┘  │
│  Port 8000                                   │
└──────────────────────────────────────────────┘
```

## CI/CD пайплайн

```
git push main
    │
    ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Тесты   │───>│  Docker  │───>│  Deploy  │
│  pytest  │    │  build   │    │ Railway  │
└──────────┘    └──────────┘    └──────────┘
```

## Деплой

| Платформа | Сложность | Бесплатно | Команда |
|-----------|-----------|-----------|---------|
| Railway | Просто | $5/мес | `railway up` |
| Render | Просто | 750 ч/мес | Через UI |
| Fly.io | Средне | 3 VM | `fly deploy` |

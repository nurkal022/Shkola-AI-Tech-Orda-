"""
Лекция 22: Создание производственного API с FastAPI
00 — Основы FastAPI
====================================================
Знакомимся с FastAPI: маршруты, модели Pydantic,
валидация входных данных и автодокументация.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# 1. Создание приложения
# ─────────────────────────────────────────────

app = FastAPI(
    title="AI Chat API",
    description="Производственный API для ИИ-приложения",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# 2. Pydantic модели — валидация данных
# ─────────────────────────────────────────────

# Pydantic автоматически:
#   - Проверяет типы
#   - Генерирует JSON Schema (для Swagger)
#   - Даёт понятные ошибки при невалидных данных


class ChatRequest(BaseModel):
    """Входящий запрос на чат."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Сообщение пользователя",
        examples=["Что такое машинное обучение?"],
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Модель для генерации ответа",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Температура генерации (0 = детерминированный, 2 = креативный)",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Максимум токенов в ответе",
    )


class ChatResponse(BaseModel):
    """Ответ от ИИ."""

    answer: str = Field(description="Ответ модели")
    model: str = Field(description="Использованная модель")
    tokens_used: int = Field(description="Количество использованных токенов")


class HealthResponse(BaseModel):
    """Статус здоровья сервиса."""

    status: str
    version: str


# ─────────────────────────────────────────────
# 3. Эндпоинты
# ─────────────────────────────────────────────


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Проверка здоровья сервиса.
    Используется для мониторинга (Kubernetes, load balancer).
    """
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Отправить сообщение ИИ-модели и получить ответ.

    - **message**: Текст вопроса
    - **model**: Модель (по умолчанию gpt-4o-mini)
    - **temperature**: Креативность (0.0 - 2.0)
    - **max_tokens**: Лимит токенов ответа
    """
    # Пока заглушка — в следующем файле подключим OpenAI
    return ChatResponse(
        answer=f"[ЗАГЛУШКА] Вы спросили: {request.message}",
        model=request.model,
        tokens_used=0,
    )


# ─────────────────────────────────────────────
# 4. Обработка ошибок
# ─────────────────────────────────────────────


class ErrorResponse(BaseModel):
    detail: str


@app.post("/chat/strict", responses={400: {"model": ErrorResponse}})
async def chat_strict(request: ChatRequest):
    """Эндпоинт со строгой валидацией модели."""
    allowed_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    if request.model not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{request.model}' не поддерживается. "
            f"Доступные: {allowed_models}",
        )
    return ChatResponse(
        answer=f"[ЗАГЛУШКА] {request.message}",
        model=request.model,
        tokens_used=0,
    )


# ─────────────────────────────────────────────
# 5. Запуск
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  FastAPI — Основы")
    print("=" * 60)
    print()
    print("  Swagger UI:  http://localhost:8000/docs")
    print("  ReDoc:       http://localhost:8000/redoc")
    print("  OpenAPI:     http://localhost:8000/openapi.json")
    print()
    print("  Попробуйте:")
    print('    curl http://localhost:8000/')
    print('    curl -X POST http://localhost:8000/chat \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"message": "Привет!"}\'')
    print()

    uvicorn.run(app, host="0.0.0.0", port=8005)

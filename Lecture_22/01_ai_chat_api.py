"""
Лекция 22: Создание производственного API с FastAPI
01 — ИИ-чат API с OpenAI
====================================================
Подключаем реальную LLM модель. Обработка ошибок,
таймауты, история сообщений.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI, APIError, APITimeoutError
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

# ─────────────────────────────────────────────
# 1. Клиент OpenAI (async!)
# ─────────────────────────────────────────────

# В FastAPI всё async — используем AsyncOpenAI
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# 2. Lifespan — инициализация и завершение
# ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan управляет жизненным циклом приложения.
    startup: проверяем подключение к OpenAI
    shutdown: освобождаем ресурсы
    """
    # Startup
    print("=" * 60)
    print("  AI Chat API — Запуск")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ОШИБКА: OPENAI_API_KEY не найден в .env!")
        raise RuntimeError("OPENAI_API_KEY not set")
    print(f"  OpenAI ключ: {api_key[:8]}...{api_key[-4:]}")

    # Проверяем подключение
    try:
        test = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        print(f"  Подключение к OpenAI: OK ({test.model})")
    except Exception as e:
        print(f"  Подключение к OpenAI: ОШИБКА — {e}")
        raise

    print()
    print("  Swagger UI: http://localhost:8001/docs")
    print("=" * 60)
    print()

    yield  # Приложение работает

    # Shutdown
    await client.close()
    print("  AI Chat API — Остановлен")


app = FastAPI(
    title="AI Chat API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# 3. Модели
# ─────────────────────────────────────────────


class Message(BaseModel):
    role: str = Field(
        ...,
        pattern="^(user|assistant|system)$",
        description="Роль: user, assistant, или system",
    )
    content: str = Field(..., min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="История сообщений",
    )
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    system_prompt: Optional[str] = Field(
        default=None,
        description="Системный промпт (добавляется в начало)",
    )


class ChatResponse(BaseModel):
    answer: str
    model: str
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


# ─────────────────────────────────────────────
# 4. Основной эндпоинт — чат
# ─────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Чат с ИИ-моделью.

    Принимает историю сообщений и возвращает ответ.
    Поддерживает системный промпт и настройку параметров.
    """
    start = time.time()

    # Формируем сообщения
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.extend([{"role": m.role, "content": m.content} for m in request.messages])

    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            timeout=30.0,  # Таймаут 30 секунд
        )
    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Модель не ответила за 30 секунд. Попробуйте уменьшить max_tokens.",
        )
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка OpenAI API: {e.message}",
        )

    latency_ms = int((time.time() - start) * 1000)
    usage = response.usage

    return ChatResponse(
        answer=response.choices[0].message.content,
        model=response.model,
        tokens_used=usage.total_tokens,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────
# 5. Простой эндпоинт — один вопрос
# ─────────────────────────────────────────────


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["Что такое FastAPI?"],
    )
    model: str = Field(default="gpt-4o-mini")


class QuestionResponse(BaseModel):
    answer: str
    tokens_used: int
    latency_ms: int


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    """
    Задать один вопрос (без истории).
    Упрощённый эндпоинт для быстрых запросов.
    """
    start = time.time()

    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "system",
                    "content": "Ты полезный ассистент. Отвечай кратко и по делу.",
                },
                {"role": "user", "content": request.question},
            ],
            max_tokens=512,
            timeout=30.0,
        )
    except (APITimeoutError, APIError) as e:
        raise HTTPException(status_code=502, detail=str(e))

    latency_ms = int((time.time() - start) * 1000)

    return QuestionResponse(
        answer=response.choices[0].message.content,
        tokens_used=response.usage.total_tokens,
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────
# 6. Получение списка моделей
# ─────────────────────────────────────────────


class ModelInfo(BaseModel):
    id: str
    description: str


AVAILABLE_MODELS = [
    ModelInfo(id="gpt-4o-mini", description="Быстрая и дешёвая, хороша для большинства задач"),
    ModelInfo(id="gpt-4o", description="Самая умная, для сложных задач"),
    ModelInfo(id="gpt-3.5-turbo", description="Устаревшая, но самая дешёвая"),
]


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """Список доступных моделей."""
    return AVAILABLE_MODELS


# ─────────────────────────────────────────────
# 7. Health check
# ─────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai-chat-api"}


# ─────────────────────────────────────────────
# 8. Запуск
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print()
    print("  Попробуйте:")
    print('    curl -X POST http://localhost:8001/ask \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"question": "Что такое FastAPI?"}\'')
    print()
    print('    curl -X POST http://localhost:8001/chat \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"messages": [{"role": "user", "content": "Привет!"}]}\'')
    print()

    uvicorn.run(app, host="0.0.0.0", port=8001)

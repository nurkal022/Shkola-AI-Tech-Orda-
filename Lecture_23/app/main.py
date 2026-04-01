"""
Лекция 23: FastAPI приложение для контейнеризации
==================================================
Это приложение из Лекции 22, адаптированное для Docker.
Запускается внутри контейнера с правильным хостом и портом.
"""

import os
import time
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI, APIError, APITimeoutError
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════
# КОНФИГУРАЦИЯ ИЗ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ═══════════════════════════════════════════

# В Docker все настройки приходят через env vars — НЕ из .env файла!
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai-api")

# ═══════════════════════════════════════════
# КЛИЕНТ OPENAI
# ═══════════════════════════════════════════

client: Optional[AsyncOpenAI] = None


def get_client() -> AsyncOpenAI:
    global client
    if client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return client


# ═══════════════════════════════════════════
# ПРИЛОЖЕНИЕ
# ═══════════════════════════════════════════

app = FastAPI(
    title="AI Chat API",
    description="Контейнеризованный AI Chat API (Лекция 23)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware: логирование
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = int((time.time() - start) * 1000)
    if request.url.path != "/health":
        logger.info(f"{request.method} {request.url.path} | {response.status_code} | {latency}ms")
    return response


# Глобальный обработчик ошибок
@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    logger.error(f"ERROR | {request.url.path} | {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ═══════════════════════════════════════════
# МОДЕЛИ
# ═══════════════════════════════════════════


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    model: str = Field(default=DEFAULT_MODEL)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    model: str
    tokens_used: int
    latency_ms: int


# ═══════════════════════════════════════════
# ЭНДПОИНТЫ
# ═══════════════════════════════════════════


@app.get("/health")
async def health():
    """Health check для Docker, Kubernetes, load balancer."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "model": DEFAULT_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Чат с ИИ-моделью."""
    start = time.time()

    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.message})

    try:
        response = await get_client().chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            timeout=30.0,
        )
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Model timeout")
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e.message}")

    return ChatResponse(
        answer=response.choices[0].message.content,
        model=response.model,
        tokens_used=response.usage.total_tokens,
        latency_ms=int((time.time() - start) * 1000),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Стриминг ответа через SSE."""
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.message})

    async def generate():
        start = time.time()
        try:
            stream = await get_client().chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'chunk': delta.content}, ensure_ascii=False)}\n\n"
                if chunk.choices[0].finish_reason:
                    break
            latency_ms = int((time.time() - start) * 1000)
            yield f"data: {json.dumps({'done': True, 'latency_ms': latency_ms})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════
# ЗАПУСК
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

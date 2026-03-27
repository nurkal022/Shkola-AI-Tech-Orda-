"""
Лекция 22: Создание производственного API с FastAPI
04 — Продакшен: middleware, rate limiting, CORS, логирование
=============================================================
Всё, что нужно для production-ready API:
аутентификация, rate limiting, CORS, структурированное
логирование и обработка ошибок.
"""

import os
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

# ─────────────────────────────────────────────
# 1. Логирование
# ─────────────────────────────────────────────

# Структурированное логирование — каждый запрос оставляет след

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai-api")

# ─────────────────────────────────────────────
# 2. Rate Limiting (ограничение запросов)
# ─────────────────────────────────────────────

# В продакшене: Redis + sliding window
# Здесь: простой in-memory rate limiter

RATE_LIMIT = 10  # максимум запросов
RATE_WINDOW = 60  # за 60 секунд

# {ip: [timestamp1, timestamp2, ...]}
request_log: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(ip: str) -> bool:
    """Проверяет, не превышен ли лимит запросов для IP."""
    now = time.time()
    # Убираем старые записи
    request_log[ip] = [t for t in request_log[ip] if now - t < RATE_WINDOW]
    # Проверяем лимит
    if len(request_log[ip]) >= RATE_LIMIT:
        return False
    request_log[ip].append(now)
    return True


# ─────────────────────────────────────────────
# 3. Аутентификация (API Key)
# ─────────────────────────────────────────────

# В продакшене: JWT, OAuth2, или сервис аутентификации
# Здесь: простая проверка Bearer token

security = HTTPBearer()

# "База" API ключей (в продакшене — в БД)
API_KEYS = {
    "sk-demo-key-12345": {"user": "student", "tier": "free"},
    "sk-pro-key-67890": {"user": "developer", "tier": "pro"},
}


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Проверяет API ключ из заголовка Authorization: Bearer <key>."""
    key = credentials.credentials
    if key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Невалидный API ключ",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return API_KEYS[key]


# ─────────────────────────────────────────────
# 4. Приложение с middleware
# ─────────────────────────────────────────────

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI API запущен (production mode)")
    yield
    await client.close()
    logger.info("AI API остановлен")


app = FastAPI(
    title="AI Production API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — разрешаем запросы с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev
        "http://localhost:5173",  # Vite dev
        "https://myapp.com",  # Продакшен
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# 5. Middleware: логирование + rate limiting
# ─────────────────────────────────────────────


@app.middleware("http")
async def logging_and_ratelimit_middleware(request: Request, call_next):
    """
    Middleware выполняется для КАЖДОГО запроса.
    1. Проверяет rate limit
    2. Логирует запрос и время ответа
    """
    start = time.time()
    client_ip = request.client.host

    # Rate limiting
    if not check_rate_limit(client_ip):
        logger.warning(f"RATE LIMIT | {client_ip} | {request.url.path}")
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Превышен лимит: {RATE_LIMIT} запросов за {RATE_WINDOW}с. "
                "Подождите и попробуйте снова."
            },
        )

    # Выполняем запрос
    response = await call_next(request)

    # Логируем результат
    latency = int((time.time() - start) * 1000)
    logger.info(
        f"{request.method} {request.url.path} | "
        f"{response.status_code} | {latency}ms | {client_ip}"
    )

    # Добавляем полезные заголовки
    response.headers["X-Request-Latency"] = f"{latency}ms"
    response.headers["X-Rate-Limit"] = str(RATE_LIMIT)
    response.headers["X-Rate-Remaining"] = str(
        RATE_LIMIT - len(request_log.get(client_ip, []))
    )

    return response


# ─────────────────────────────────────────────
# 6. Глобальный обработчик ошибок
# ─────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ловит ВСЕ необработанные ошибки — клиент никогда не увидит traceback."""
    logger.error(f"UNHANDLED ERROR | {request.url.path} | {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера. Попробуйте позже."},
    )


# ─────────────────────────────────────────────
# 7. Защищённые эндпоинты
# ─────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    model: str = Field(default="gpt-4o-mini")


class ChatResponse(BaseModel):
    answer: str
    model: str
    user: str
    tier: str
    latency_ms: int


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_info: dict = Depends(verify_api_key)):
    """
    Защищённый чат — требует API ключ.

    Заголовок: `Authorization: Bearer sk-demo-key-12345`
    """
    start = time.time()

    # Tier-based лимиты
    max_tokens = 512 if user_info["tier"] == "free" else 2048

    response = await client.chat.completions.create(
        model=request.model,
        messages=[{"role": "user", "content": request.message}],
        max_tokens=max_tokens,
    )

    return ChatResponse(
        answer=response.choices[0].message.content,
        model=response.model,
        user=user_info["user"],
        tier=user_info["tier"],
        latency_ms=int((time.time() - start) * 1000),
    )


# ─────────────────────────────────────────────
# 8. Публичные эндпоинты (без аутентификации)
# ─────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check — для мониторинга и load balancer."""
    return {"status": "ok", "service": "ai-production-api"}


@app.get("/info")
async def info():
    """Информация о сервисе."""
    return {
        "name": "AI Production API",
        "version": "1.0.0",
        "rate_limit": f"{RATE_LIMIT} req / {RATE_WINDOW}s",
        "auth": "Bearer token required for /chat",
        "demo_keys": list(API_KEYS.keys()),
    }


# ─────────────────────────────────────────────
# 9. Запуск
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  AI Production API")
    print("=" * 60)
    print()
    print("  Swagger UI: http://localhost:8004/docs")
    print()
    print("  Демо-ключи:")
    for key, info in API_KEYS.items():
        print(f"    {key} -> {info}")
    print()
    print("  Примеры:")
    print()
    print("  # Без ключа — 403:")
    print('    curl -X POST http://localhost:8004/chat \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"message": "Привет"}\'')
    print()
    print("  # С ключом — работает:")
    print('    curl -X POST http://localhost:8004/chat \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -H "Authorization: Bearer sk-demo-key-12345" \\')
    print('      -d \'{"message": "Что такое FastAPI?"}\'')
    print()
    print(f"  Rate limit: {RATE_LIMIT} запросов / {RATE_WINDOW}с")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8004)

"""
Лекция 24: Безопасность в продакшене
03 — Rate Limiting и защита от злоупотреблений
====================================================
Ограничение частоты запросов: sliding window,
token bucket, per-user limits, cost-based limiting.
"""

import time
import asyncio
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

app = FastAPI(title="Rate Limiting Demo")


# ============================================================
# 1. Sliding Window Rate Limiter
# ============================================================

class SlidingWindowLimiter:
    """
    Sliding Window — скользящее окно.
    Считает запросы за последние N секунд.
    Самый простой и понятный алгоритм.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        now = time.time()
        # Убираем старые записи
        self.requests[key] = [
            t for t in self.requests[key] if now - t < self.window
        ]
        remaining = self.max_requests - len(self.requests[key])
        info = {
            "limit": self.max_requests,
            "remaining": max(0, remaining),
            "reset_in": int(self.window - (now - self.requests[key][0])) if self.requests[key] else 0,
        }

        if remaining <= 0:
            return False, info

        self.requests[key].append(now)
        info["remaining"] = max(0, remaining - 1)
        return True, info


# ============================================================
# 2. Token Bucket Rate Limiter
# ============================================================

class TokenBucketLimiter:
    """
    Token Bucket — ведро токенов.
    Токены добавляются с постоянной скоростью.
    Позволяет короткие всплески трафика (bursts).

    Пример: 10 req/min с burst до 5.
    """

    def __init__(self, rate: float, capacity: int):
        """
        rate: токенов в секунду (напр. 0.167 = 10/мин)
        capacity: максимум токенов в ведре (burst size)
        """
        self.rate = rate
        self.capacity = capacity
        self.buckets: dict[str, dict] = {}

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        now = time.time()

        if key not in self.buckets:
            self.buckets[key] = {"tokens": self.capacity, "last": now}

        bucket = self.buckets[key]
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(
            self.capacity, bucket["tokens"] + elapsed * self.rate
        )
        bucket["last"] = now

        info = {
            "tokens": round(bucket["tokens"], 1),
            "capacity": self.capacity,
            "rate": f"{self.rate * 60:.0f}/min",
        }

        if bucket["tokens"] < 1:
            return False, info

        bucket["tokens"] -= 1
        info["tokens"] = round(bucket["tokens"], 1)
        return True, info


# ============================================================
# 3. Per-User Tiered Rate Limiting
# ============================================================

TIERS = {
    "free": {"requests_per_min": 5, "tokens_per_day": 10_000},
    "pro": {"requests_per_min": 30, "tokens_per_day": 100_000},
    "enterprise": {"requests_per_min": 100, "tokens_per_day": 1_000_000},
}

API_KEYS = {
    "sk-free-001": {"user": "student", "tier": "free"},
    "sk-pro-001": {"user": "developer", "tier": "pro"},
    "sk-ent-001": {"user": "company", "tier": "enterprise"},
}

# Лимитеры per-tier
tier_limiters = {
    tier: SlidingWindowLimiter(cfg["requests_per_min"], 60)
    for tier, cfg in TIERS.items()
}

# Трекинг токенов per-user
daily_tokens: dict[str, int] = defaultdict(int)

security = HTTPBearer(auto_error=False)


async def get_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    if credentials is None:
        return {"user": "anonymous", "tier": "free"}
    key = credentials.credentials
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Невалидный API ключ")
    return API_KEYS[key]


# ============================================================
# 4. Middleware с Rate Limiting
# ============================================================

# Глобальный лимитер для IP (защита от DDoS)
ip_limiter = SlidingWindowLimiter(max_requests=30, window_seconds=60)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware — проверяет IP-based лимит."""
    client_ip = request.client.host

    # Пропускаем health check
    if request.url.path in ("/health", "/docs", "/openapi.json"):
        return await call_next(request)

    allowed, info = ip_limiter.is_allowed(client_ip)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Too Many Requests",
                "limit": info["limit"],
                "retry_after": info["reset_in"],
            },
            headers={
                "Retry-After": str(info["reset_in"]),
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0",
            },
        )

    response = await call_next(request)

    # Добавляем rate limit headers к каждому ответу
    response.headers["X-RateLimit-Limit"] = str(info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(info["remaining"])

    return response


# ============================================================
# 5. Эндпоинты с per-user лимитами
# ============================================================


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


class ChatResponse(BaseModel):
    answer: str
    tier: str
    rate_limit: dict


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: dict = Depends(get_user)):
    """Чат с per-user rate limiting."""
    tier = user["tier"]
    limiter = tier_limiters[tier]

    # Проверяем rate limit для пользователя
    allowed, info = limiter.is_allowed(user["user"])
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"Превышен лимит для тарифа '{tier}'",
                "limit": info["limit"],
                "tier": tier,
                "retry_after_seconds": info["reset_in"],
                "upgrade": "Для увеличения лимита перейдите на тариф Pro",
            },
        )

    # Проверяем дневной лимит токенов
    max_daily = TIERS[tier]["tokens_per_day"]
    if daily_tokens[user["user"]] >= max_daily:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Дневной лимит токенов исчерпан",
                "used": daily_tokens[user["user"]],
                "limit": max_daily,
            },
        )

    # Симулируем ответ (без реального вызова LLM)
    estimated_tokens = len(req.message.split()) * 3
    daily_tokens[user["user"]] += estimated_tokens

    return ChatResponse(
        answer=f"[Mock] Ответ на: {req.message[:50]}",
        tier=tier,
        rate_limit=info,
    )


# ============================================================
# 6. Информация о лимитах
# ============================================================


@app.get("/limits")
async def get_limits(user: dict = Depends(get_user)):
    """Текущие лимиты пользователя."""
    tier = user["tier"]
    cfg = TIERS[tier]
    return {
        "user": user["user"],
        "tier": tier,
        "requests_per_min": cfg["requests_per_min"],
        "tokens_per_day": cfg["tokens_per_day"],
        "tokens_used_today": daily_tokens.get(user["user"], 0),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================
# 7. Тестирование
# ============================================================

if __name__ == "__main__":
    t = TestClient(app)

    print("=" * 60)
    print("  RATE LIMITING DEMO")
    print("=" * 60)

    # --- Sliding Window ---
    print()
    print("1. Sliding Window (5 req / 10s):")
    limiter = SlidingWindowLimiter(max_requests=5, window_seconds=10)
    for i in range(7):
        ok, info = limiter.is_allowed("user1")
        print(f"    Запрос {i+1}: {'OK' if ok else 'BLOCKED':7s} | remaining={info['remaining']}")

    # --- Token Bucket ---
    print()
    print("2. Token Bucket (rate=2/s, capacity=5):")
    bucket = TokenBucketLimiter(rate=2.0, capacity=5)
    for i in range(7):
        ok, info = bucket.is_allowed("user1")
        print(f"    Запрос {i+1}: {'OK' if ok else 'BLOCKED':7s} | tokens={info['tokens']}")

    # --- Per-User Limits ---
    print()
    print("3. Per-User Limits (free: 5 req/min):")
    for i in range(7):
        r = t.post(
            "/chat",
            json={"message": f"Запрос {i+1}"},
            headers={"Authorization": "Bearer sk-free-001"},
        )
        status = "OK" if r.status_code == 200 else f"BLOCKED ({r.status_code})"
        print(f"    Запрос {i+1}: {status}")

    # --- Pro User ---
    print()
    print("4. Pro User (30 req/min) — первые 7 запросов:")
    for i in range(7):
        r = t.post(
            "/chat",
            json={"message": f"Запрос {i+1}"},
            headers={"Authorization": "Bearer sk-pro-001"},
        )
        status = "OK" if r.status_code == 200 else f"BLOCKED ({r.status_code})"
        print(f"    Запрос {i+1}: {status}")

    # --- Rate limit info ---
    print()
    print("5. Информация о лимитах:")
    for key, info in API_KEYS.items():
        r = t.get("/limits", headers={"Authorization": f"Bearer {key}"})
        data = r.json()
        print(f"    {info['user']:12s} ({info['tier']:10s}): {data['requests_per_min']} req/min, {data['tokens_per_day']:>10,} tok/day")

    print()
    print("=" * 60)
    print("  ИТОГИ:")
    print("  Sliding Window — простой, понятный, для большинства случаев")
    print("  Token Bucket   — позволяет bursts, для API с пиками")
    print("  Per-User        — разные лимиты для разных тарифов")
    print("  Headers         — X-RateLimit-Limit, Remaining, Retry-After")
    print("=" * 60)

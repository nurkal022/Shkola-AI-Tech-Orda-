"""
Лекция 24: Безопасность в продакшене
02 — Серверная валидация входных данных
====================================================
Pydantic + FastAPI для строгой валидации:
типы, диапазоны, regex, кастомные валидаторы.
"""

import re
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, field_validator, model_validator

app = FastAPI(title="Input Validation Demo")


# ============================================================
# 1. Базовая валидация с Pydantic
# ============================================================

# Pydantic автоматически:
# - Проверяет типы (str, int, float)
# - Проверяет ограничения (min_length, max_length, ge, le)
# - Отдаёт понятные ошибки 422 при невалидных данных


class AllowedModel(str, Enum):
    """Только разрешённые модели — нельзя передать произвольную строку."""
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O = "gpt-4o"
    GPT35 = "gpt-3.5-turbo"


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Сообщение пользователя",
    )
    model: AllowedModel = Field(
        default=AllowedModel.GPT4O_MINI,
        description="Модель (только из списка разрешённых)",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    return {"message": req.message, "model": req.model, "valid": True}


# ============================================================
# 2. Кастомные валидаторы
# ============================================================


class SecureChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    model: AllowedModel = Field(default=AllowedModel.GPT4O_MINI)
    user_id: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",  # Только буквы, цифры, _, -
        description="ID пользователя (только алфавитно-цифровые символы)",
    )
    language: str = Field(
        default="ru",
        pattern=r"^(ru|en|kz)$",
        description="Язык ответа",
    )

    @field_validator("message")
    @classmethod
    def check_message_content(cls, v: str) -> str:
        """Кастомная проверка содержимого сообщения."""
        # Убираем управляющие символы
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", v)

        # Проверка на пустоту после очистки
        if not v.strip():
            raise ValueError("Сообщение не может быть пустым после очистки")

        # Блокируем HTML/скрипты
        if re.search(r"<script|<iframe|javascript:|on\w+\s*=", v, re.IGNORECASE):
            raise ValueError("HTML-теги и скрипты запрещены")

        return v.strip()

    @field_validator("user_id")
    @classmethod
    def check_user_id(cls, v: str) -> str:
        """Проверка user_id на SQL-injection паттерны."""
        dangerous = ["'", '"', ";", "--", "/*", "*/", "DROP", "DELETE", "UNION"]
        for d in dangerous:
            if d.lower() in v.lower():
                raise ValueError(f"Недопустимый символ в user_id: {d}")
        return v


@app.post("/chat/secure")
async def secure_chat(req: SecureChatRequest):
    return {
        "message": req.message,
        "user_id": req.user_id,
        "model": req.model,
        "language": req.language,
    }


# ============================================================
# 3. Модель с model_validator (cross-field validation)
# ============================================================


class TransferRequest(BaseModel):
    """Запрос на перевод — валидация зависимостей между полями."""

    from_account: str = Field(..., pattern=r"^KZ\d{18}$")
    to_account: str = Field(..., pattern=r"^KZ\d{18}$")
    amount: float = Field(..., gt=0, le=1_000_000)
    currency: str = Field(default="KZT", pattern=r"^(KZT|USD|EUR)$")
    description: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def check_accounts(self):
        """Проверка: нельзя переводить самому себе."""
        if self.from_account == self.to_account:
            raise ValueError("Нельзя переводить на тот же счёт")
        return self


@app.post("/transfer")
async def transfer(req: TransferRequest):
    return {
        "from": req.from_account,
        "to": req.to_account,
        "amount": req.amount,
        "currency": req.currency,
        "status": "ok",
    }


# ============================================================
# 4. Тестирование валидации
# ============================================================

if __name__ == "__main__":
    t = TestClient(app)

    print("=" * 60)
    print("  ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ")
    print("=" * 60)

    # --- Базовая валидация ---
    print()
    print("1. Базовая валидация (ChatRequest):")
    tests = [
        ({"message": "Привет"}, "Норма"),
        ({"message": ""}, "Пустая строка"),
        ({"message": "Hi", "model": "bad-model"}, "Плохая модель"),
        ({"message": "Hi", "temperature": 5.0}, "temperature > 2"),
        ({"message": "Hi", "max_tokens": -1}, "max_tokens < 1"),
    ]
    for body, desc in tests:
        r = t.post("/chat", json=body)
        print(f"    [{r.status_code}] {desc:30s} | {str(body)[:50]}")

    # --- Кастомные валидаторы ---
    print()
    print("2. Кастомная валидация (SecureChatRequest):")
    tests = [
        ({"message": "Привет", "user_id": "user_123"}, "Норма"),
        ({"message": "Привет", "user_id": "user'; DROP TABLE--"}, "SQL injection в user_id"),
        ({"message": "<script>alert(1)</script>", "user_id": "ok"}, "XSS в message"),
        ({"message": "Hi", "user_id": "ok", "language": "fr"}, "Неподдерживаемый язык"),
        ({"message": "Hi", "user_id": "a b c"}, "Пробелы в user_id"),
    ]
    for body, desc in tests:
        r = t.post("/chat/secure", json=body)
        print(f"    [{r.status_code}] {desc:35s} | {str(body)[:50]}")

    # --- Cross-field валидация ---
    print()
    print("3. Cross-field валидация (TransferRequest):")
    tests = [
        (
            {"from_account": "KZ123456789012345678", "to_account": "KZ987654321098765432", "amount": 1000},
            "Норма",
        ),
        (
            {"from_account": "KZ123456789012345678", "to_account": "KZ123456789012345678", "amount": 1000},
            "Перевод самому себе",
        ),
        (
            {"from_account": "INVALID", "to_account": "KZ987654321098765432", "amount": 1000},
            "Невалидный формат счёта",
        ),
        (
            {"from_account": "KZ123456789012345678", "to_account": "KZ987654321098765432", "amount": -100},
            "Отрицательная сумма",
        ),
        (
            {"from_account": "KZ123456789012345678", "to_account": "KZ987654321098765432", "amount": 5_000_000},
            "Сумма > лимита",
        ),
    ]
    for body, desc in tests:
        r = t.post("/transfer", json=body)
        print(f"    [{r.status_code}] {desc:30s} | {str(body)[:60]}")

    print()
    print("=" * 60)
    print("  ИТОГИ:")
    print("  1. Pydantic Field — типы, диапазоны, regex")
    print("  2. Enum — фиксированный набор значений")
    print("  3. @field_validator — кастомная логика проверки")
    print("  4. @model_validator — зависимости между полями")
    print("  5. Всё отдаёт 422 с понятными ошибками")
    print("=" * 60)

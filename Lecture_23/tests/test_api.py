"""
Лекция 23: Тесты для AI Chat API
==================================
Тесты запускаются в CI/CD пайплайне.

Запуск: pytest tests/ -v
"""

import os
import importlib.util

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Создаёт тестовый клиент для FastAPI."""
    spec = importlib.util.spec_from_file_location(
        "main",
        os.path.join(os.path.dirname(__file__), "..", "app", "main.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return TestClient(mod.app)


# ─────────────────────────────────────────────
# 1. Health check
# ─────────────────────────────────────────────


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_contains_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


# ─────────────────────────────────────────────
# 2. Валидация запросов
# ─────────────────────────────────────────────


class TestValidation:
    def test_empty_message_returns_422(self, client):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422

    def test_missing_message_returns_422(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_temperature_too_high_returns_422(self, client):
        response = client.post(
            "/chat",
            json={"message": "test", "temperature": 3.0},
        )
        assert response.status_code == 422

    def test_max_tokens_too_high_returns_422(self, client):
        response = client.post(
            "/chat",
            json={"message": "test", "max_tokens": 99999},
        )
        assert response.status_code == 422


# ─────────────────────────────────────────────
# 3. Чат (требует OPENAI_API_KEY)
# ─────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestChat:
    def test_chat_returns_200(self, client):
        response = client.post("/chat", json={"message": "Say yes"})
        assert response.status_code == 200

    def test_chat_returns_answer(self, client):
        data = client.post("/chat", json={"message": "Say yes"}).json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_chat_returns_metadata(self, client):
        data = client.post("/chat", json={"message": "Say yes"}).json()
        assert "model" in data
        assert "tokens_used" in data
        assert "latency_ms" in data

    def test_chat_with_system_prompt(self, client):
        data = client.post(
            "/chat",
            json={
                "message": "What is 2+2?",
                "system_prompt": "Answer only with a number.",
            },
        ).json()
        assert "4" in data["answer"]


# ─────────────────────────────────────────────
# 4. Стриминг
# ─────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestStreaming:
    def test_stream_returns_200(self, client):
        response = client.post("/chat/stream", json={"message": "Say hi"})
        assert response.status_code == 200

    def test_stream_returns_sse(self, client):
        response = client.post("/chat/stream", json={"message": "Say hi"})
        assert "text/event-stream" in response.headers["content-type"]

    def test_stream_contains_data(self, client):
        response = client.post("/chat/stream", json={"message": "Say hi"})
        lines = [l for l in response.text.split("\n") if l.startswith("data:")]
        assert len(lines) > 0

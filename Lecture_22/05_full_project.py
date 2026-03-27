"""
Лекция 22: Создание производственного API с FastAPI
05 — Полный проект: AI Chat API
====================================================
Итоговый файл — объединяет всё:
- Чат с историей
- Стриминг
- Фоновые задачи (суммаризация)
- Rate limiting
- API ключи
- CORS
- Логирование
- Health check

Запуск: python 05_full_project.py
Документация: http://localhost:8000/docs
"""

import os
import uuid
import time
import json
import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from openai import AsyncOpenAI, APIError, APITimeoutError
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

# ═════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═════════════════════════════════════════════

RATE_LIMIT = 20
RATE_WINDOW = 60
DEFAULT_MODEL = "gpt-4o-mini"
MAX_HISTORY = 50
REQUEST_TIMEOUT = 30.0

# ═════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ═════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai-api")

# ═════════════════════════════════════════════
# КЛИЕНТ И ХРАНИЛИЩА
# ═════════════════════════════════════════════

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API ключи (в продакшене — БД)
API_KEYS = {
    "sk-demo-12345": {"user": "demo", "tier": "free", "max_tokens": 512},
    "sk-pro-67890": {"user": "pro_user", "tier": "pro", "max_tokens": 4096},
}

# Rate limiting
request_log: dict[str, list[float]] = defaultdict(list)

# Истории чатов
chat_sessions: dict[str, list[dict]] = {}


# Задачи
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


tasks_store: dict[str, dict] = {}

# ═════════════════════════════════════════════
# УТИЛИТЫ
# ═════════════════════════════════════════════

security = HTTPBearer(auto_error=False)


async def get_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Извлекает пользователя из API ключа. Если ключа нет — анонимный пользователь."""
    if credentials is None:
        return {"user": "anonymous", "tier": "free", "max_tokens": 256}
    key = credentials.credentials
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Невалидный API ключ")
    return API_KEYS[key]


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    request_log[ip] = [t for t in request_log[ip] if now - t < RATE_WINDOW]
    if len(request_log[ip]) >= RATE_LIMIT:
        return False
    request_log[ip].append(now)
    return True


# ═════════════════════════════════════════════
# ПРИЛОЖЕНИЕ
# ═════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Chat API запущен")
    yield
    await client.close()
    logger.info("AI Chat API остановлен")


app = FastAPI(
    title="AI Chat API",
    description=(
        "Производственный API для ИИ-приложений.\n\n"
        "## Возможности\n"
        "- Чат с историей сообщений\n"
        "- Стриминг ответов (SSE)\n"
        "- Фоновая генерация отчётов\n"
        "- Rate limiting и аутентификация\n\n"
        "## Аутентификация\n"
        "Передайте API ключ в заголовке: `Authorization: Bearer <key>`\n\n"
        "Демо-ключи: `sk-demo-12345` (free), `sk-pro-67890` (pro)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене — список доменов
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware: логирование + rate limit
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    start = time.time()
    client_ip = request.client.host

    if request.url.path not in ("/health", "/docs", "/openapi.json", "/redoc"):
        if not check_rate_limit(client_ip):
            logger.warning(f"RATE LIMIT | {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit: {RATE_LIMIT} req / {RATE_WINDOW}s"},
            )

    response = await call_next(request)
    latency = int((time.time() - start) * 1000)

    if request.url.path not in ("/health",):
        logger.info(f"{request.method} {request.url.path} | {response.status_code} | {latency}ms")

    response.headers["X-Latency"] = f"{latency}ms"
    return response


# Глобальный обработчик ошибок
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    logger.error(f"ERROR | {request.url.path} | {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера"},
    )


# ═════════════════════════════════════════════
# МОДЕЛИ
# ═════════════════════════════════════════════


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = Field(
        default=None,
        description="ID сессии для сохранения истории",
    )
    model: str = Field(default=DEFAULT_MODEL)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system_prompt: Optional[str] = Field(default=None)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    model: str
    tokens_used: int
    latency_ms: int
    user: str
    tier: str


# ═════════════════════════════════════════════
# ЭНДПОИНТЫ: ЧАТ
# ═════════════════════════════════════════════


@app.post("/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, user: dict = Depends(get_user)):
    """
    Чат с ИИ-моделью (с историей сообщений).

    - Без `session_id` — создаётся новая сессия
    - С `session_id` — продолжение существующей сессии
    """
    start = time.time()

    # Создаём или получаем сессию
    session_id = request.session_id or str(uuid.uuid4())[:8]
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    history = chat_sessions[session_id]

    # Формируем сообщения
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.extend(history[-MAX_HISTORY:])  # Последние N сообщений
    messages.append({"role": "user", "content": request.message})

    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=user["max_tokens"],
            timeout=REQUEST_TIMEOUT,
        )
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Таймаут модели")
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка OpenAI: {e.message}")

    answer = response.choices[0].message.content

    # Сохраняем в историю
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": answer})

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        model=response.model,
        tokens_used=response.usage.total_tokens,
        latency_ms=int((time.time() - start) * 1000),
        user=user["user"],
        tier=user["tier"],
    )


# ═════════════════════════════════════════════
# ЭНДПОИНТЫ: СТРИМИНГ
# ═════════════════════════════════════════════


@app.post("/v1/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest, user: dict = Depends(get_user)):
    """
    Стриминг ответа через Server-Sent Events.
    Текст приходит по частям — как в ChatGPT.
    """
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.message})

    async def generate():
        start = time.time()
        try:
            stream = await client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=user["max_tokens"],
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


# ═════════════════════════════════════════════
# ЭНДПОИНТЫ: ФОНОВЫЕ ЗАДАЧИ
# ═════════════════════════════════════════════


class ReportRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    sections: int = Field(default=3, ge=1, le=10)


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


async def generate_report_task(task_id: str, topic: str, sections: int):
    task = tasks_store[task_id]
    task["status"] = TaskStatus.RUNNING

    try:
        parts = []
        for i in range(sections):
            task["progress"] = int((i / sections) * 100)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Ты аналитик. Пиши кратко, 2-3 предложения."},
                    {"role": "user", "content": f"Секция {i+1}/{sections} отчёта: '{topic}'"},
                ],
                max_tokens=300,
            )
            parts.append(f"## Секция {i+1}\n{response.choices[0].message.content}")

        task["result"] = "\n\n".join(parts)
        task["status"] = TaskStatus.COMPLETED
        task["progress"] = 100
    except Exception as e:
        task["status"] = TaskStatus.FAILED
        task["error"] = str(e)


@app.post("/v1/report", response_model=TaskResponse, tags=["Tasks"])
async def create_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_user),
):
    """Запуск генерации отчёта в фоне."""
    task_id = str(uuid.uuid4())[:8]
    tasks_store[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "progress": 0,
        "result": None,
        "error": None,
        "user": user["user"],
        "created_at": datetime.now().isoformat(),
    }
    background_tasks.add_task(generate_report_task, task_id, request.topic, request.sections)
    return TaskResponse(
        task_id=task_id,
        status="pending",
        message=f"Проверяйте: GET /v1/tasks/{task_id}",
    )


@app.get("/v1/tasks/{task_id}", tags=["Tasks"])
async def get_task(task_id: str):
    """Получить статус и результат задачи."""
    if task_id not in tasks_store:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return tasks_store[task_id]


# ═════════════════════════════════════════════
# ЭНДПОИНТЫ: СЕССИИ
# ═════════════════════════════════════════════


@app.get("/v1/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Получить историю чата."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {
        "session_id": session_id,
        "messages": chat_sessions[session_id],
        "message_count": len(chat_sessions[session_id]),
    }


@app.delete("/v1/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    """Удалить историю чата."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    del chat_sessions[session_id]
    return {"message": f"Сессия {session_id} удалена"}


# ═════════════════════════════════════════════
# СЛУЖЕБНЫЕ ЭНДПОИНТЫ
# ═════════════════════════════════════════════


@app.get("/health", tags=["Service"])
async def health():
    return {"status": "ok"}


@app.get("/info", tags=["Service"])
async def info():
    return {
        "name": "AI Chat API",
        "version": "1.0.0",
        "endpoints": {
            "POST /v1/chat": "Чат с ИИ (с историей)",
            "POST /v1/chat/stream": "Стриминг ответа (SSE)",
            "POST /v1/report": "Генерация отчёта (фон)",
            "GET /v1/tasks/{id}": "Статус задачи",
            "GET /v1/sessions/{id}": "История сессии",
        },
        "rate_limit": f"{RATE_LIMIT} req / {RATE_WINDOW}s",
        "auth": "Authorization: Bearer <api-key>",
        "demo_keys": ["sk-demo-12345 (free)", "sk-pro-67890 (pro)"],
    }


# ═════════════════════════════════════════════
# ДЕМО-СТРАНИЦА
# ═════════════════════════════════════════════

DEMO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Chat API — Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f172a; color: #e2e8f0;
            display: flex; justify-content: center; padding: 40px 20px;
        }
        .app { max-width: 700px; width: 100%; }
        h1 { text-align: center; color: #38bdf8; margin-bottom: 8px; }
        .subtitle { text-align: center; color: #64748b; margin-bottom: 24px; }
        .chat {
            background: #1e293b; border-radius: 12px;
            padding: 20px; min-height: 300px; max-height: 500px;
            overflow-y: auto; margin-bottom: 16px;
        }
        .msg { margin-bottom: 12px; line-height: 1.5; }
        .msg.user { color: #38bdf8; }
        .msg.user::before { content: "Вы: "; font-weight: bold; }
        .msg.ai::before { content: "AI: "; font-weight: bold; color: #a78bfa; }
        .input-row { display: flex; gap: 8px; }
        input {
            flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #334155;
            background: #1e293b; color: #e2e8f0; font-size: 15px;
        }
        button {
            padding: 12px 20px; border-radius: 8px; border: none;
            background: #38bdf8; color: #0f172a; font-weight: bold;
            font-size: 15px; cursor: pointer;
        }
        button:hover { background: #0ea5e9; }
        .config { display: flex; gap: 8px; margin-bottom: 12px; }
        .config input { font-size: 13px; padding: 8px; }
        .config label { color: #64748b; font-size: 13px; align-self: center; }
        .meta { color: #475569; font-size: 12px; margin-top: 8px; }
    </style>
</head>
<body>
<div class="app">
    <h1>AI Chat API</h1>
    <p class="subtitle">FastAPI + OpenAI — Production Demo</p>
    <div class="config">
        <label>API Key:</label>
        <input type="text" id="apikey" value="sk-demo-12345" style="flex:1">
        <label>Stream:</label>
        <input type="checkbox" id="streaming" checked style="flex:none; width:20px">
    </div>
    <div class="chat" id="chat"></div>
    <div class="input-row">
        <input type="text" id="msg" placeholder="Введите сообщение..."
               onkeypress="if(event.key==='Enter')send()">
        <button onclick="send()">Отправить</button>
    </div>
    <div class="meta" id="meta"></div>
</div>
<script>
let sessionId = null;
async function send() {
    const msg = document.getElementById('msg').value;
    if (!msg) return;
    document.getElementById('msg').value = '';
    const chat = document.getElementById('chat');
    const meta = document.getElementById('meta');
    const apikey = document.getElementById('apikey').value;
    const streaming = document.getElementById('streaming').checked;

    chat.innerHTML += `<div class="msg user">${msg}</div>`;
    const aiDiv = document.createElement('div');
    aiDiv.className = 'msg ai';
    chat.appendChild(aiDiv);
    chat.scrollTop = chat.scrollHeight;

    const headers = {'Content-Type': 'application/json'};
    if (apikey) headers['Authorization'] = `Bearer ${apikey}`;
    const body = {message: msg};
    if (sessionId) body.session_id = sessionId;

    if (streaming) {
        const res = await fetch('/v1/chat/stream', {method:'POST', headers, body: JSON.stringify(body)});
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        while (true) {
            const {value, done} = await reader.read();
            if (done) break;
            for (const line of dec.decode(value).split('\\n')) {
                if (!line.startsWith('data: ')) continue;
                const d = JSON.parse(line.slice(6));
                if (d.chunk) { aiDiv.textContent += d.chunk; chat.scrollTop = chat.scrollHeight; }
                if (d.done) meta.textContent = `Latency: ${d.latency_ms}ms | Stream`;
            }
        }
    } else {
        const res = await fetch('/v1/chat', {method:'POST', headers, body: JSON.stringify(body)});
        const data = await res.json();
        if (data.answer) {
            aiDiv.textContent = data.answer;
            sessionId = data.session_id;
            meta.textContent = `Latency: ${data.latency_ms}ms | Tokens: ${data.tokens_used} | Session: ${data.session_id}`;
        } else {
            aiDiv.textContent = '[Ошибка] ' + (data.detail || JSON.stringify(data));
        }
    }
    chat.scrollTop = chat.scrollHeight;
}
</script>
</body>
</html>
"""


@app.get("/demo", response_class=HTMLResponse, tags=["Service"])
async def demo():
    """Интерактивная демо-страница."""
    return DEMO_HTML


# ═════════════════════════════════════════════
# ЗАПУСК
# ═════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  AI Chat API — Full Project")
    print("=" * 60)
    print()
    print("  Демо:    http://localhost:8000/demo")
    print("  Swagger: http://localhost:8000/docs")
    print("  ReDoc:   http://localhost:8000/redoc")
    print()
    print("  Ключи:")
    print("    sk-demo-12345 (free, 512 tokens)")
    print("    sk-pro-67890  (pro, 4096 tokens)")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)

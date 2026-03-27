"""
Лекция 22: Создание производственного API с FastAPI
02 — Стриминг ответов (Server-Sent Events)
====================================================
Стриминг позволяет отправлять ответ по частям,
как в ChatGPT — пользователь видит текст по мере генерации.
"""

import os
import json
import time
import asyncio
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, APIError
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI Streaming API", version="1.0.0")


# ─────────────────────────────────────────────
# 1. Модели
# ─────────────────────────────────────────────


class StreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    system_prompt: Optional[str] = Field(default=None)


# ─────────────────────────────────────────────
# 2. Стриминг через SSE (Server-Sent Events)
# ─────────────────────────────────────────────

#   Формат SSE:
#     data: {"chunk": "Привет"}
#     data: {"chunk": ", как"}
#     data: {"chunk": " дела?"}
#     data: [DONE]
#
#   Браузер / фронтенд читает EventSource и
#   рендерит текст по мере поступления.


@app.post("/chat/stream")
async def chat_stream(request: StreamRequest):
    """
    Стриминг ответа от ИИ через Server-Sent Events.

    Ответ приходит по частям (чанкам) — пользователь видит
    текст по мере генерации, как в ChatGPT.
    """

    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.message})

    async def generate():
        start = time.time()
        total_tokens = 0

        try:
            # stream=True — OpenAI возвращает AsyncGenerator
            stream = await client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta

                if delta.content:
                    total_tokens += 1  # Приблизительный подсчёт
                    # Формат SSE: data: <json>\n\n
                    data = json.dumps(
                        {"chunk": delta.content},
                        ensure_ascii=False,
                    )
                    yield f"data: {data}\n\n"

                # Проверяем, закончилась ли генерация
                if chunk.choices[0].finish_reason is not None:
                    break

            # Финальное сообщение с метаданными
            latency_ms = int((time.time() - start) * 1000)
            done_data = json.dumps(
                {
                    "done": True,
                    "model": request.model,
                    "latency_ms": latency_ms,
                    "approx_tokens": total_tokens,
                },
                ensure_ascii=False,
            )
            yield f"data: {done_data}\n\n"

        except APIError as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Отключает буферизацию в nginx
        },
    )


# ─────────────────────────────────────────────
# 3. Обычный (не стриминговый) эндпоинт для сравнения
# ─────────────────────────────────────────────


class ChatResponse(BaseModel):
    answer: str
    model: str
    latency_ms: int


@app.post("/chat", response_model=ChatResponse)
async def chat(request: StreamRequest):
    """Обычный (блокирующий) чат — ждём полный ответ."""
    start = time.time()

    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.message})

    response = await client.chat.completions.create(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    return ChatResponse(
        answer=response.choices[0].message.content,
        model=response.model,
        latency_ms=int((time.time() - start) * 1000),
    )


# ─────────────────────────────────────────────
# 4. HTML-страница для демонстрации стриминга
# ─────────────────────────────────────────────

DEMO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Chat — Streaming Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 700px; margin: 50px auto; padding: 20px;
            background: #1a1a2e; color: #eee;
        }
        h1 { color: #00d4aa; text-align: center; }
        .chat-box {
            background: #16213e; border-radius: 12px;
            padding: 20px; min-height: 200px;
            margin: 20px 0; white-space: pre-wrap;
            line-height: 1.6; font-size: 16px;
        }
        .input-row { display: flex; gap: 10px; }
        input {
            flex: 1; padding: 12px; border-radius: 8px;
            border: 1px solid #444; background: #0f3460;
            color: #eee; font-size: 16px;
        }
        button {
            padding: 12px 24px; border-radius: 8px; border: none;
            background: #00d4aa; color: #1a1a2e;
            font-weight: bold; font-size: 16px; cursor: pointer;
        }
        button:hover { background: #00b894; }
        .meta { color: #888; font-size: 13px; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>AI Chat — Streaming</h1>
    <div class="chat-box" id="output">Введите вопрос и нажмите "Отправить"...</div>
    <div class="input-row">
        <input type="text" id="msg" placeholder="Ваш вопрос..."
               onkeypress="if(event.key==='Enter')send()">
        <button onclick="send()">Отправить</button>
    </div>
    <div class="meta" id="meta"></div>

    <script>
    async function send() {
        const msg = document.getElementById('msg').value;
        if (!msg) return;

        const output = document.getElementById('output');
        const meta = document.getElementById('meta');
        output.textContent = '';
        meta.textContent = 'Генерация...';

        const response = await fetch('/chat/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg})
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const {value, done} = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            for (const line of text.split('\\n')) {
                if (!line.startsWith('data: ')) continue;
                const data = JSON.parse(line.slice(6));

                if (data.chunk) {
                    output.textContent += data.chunk;
                } else if (data.done) {
                    meta.textContent =
                        `Модель: ${data.model} | ` +
                        `Время: ${data.latency_ms}ms | ` +
                        `Токенов: ~${data.approx_tokens}`;
                } else if (data.error) {
                    output.textContent += '\\n\\n[ОШИБКА] ' + data.error;
                }
            }
        }
    }
    </script>
</body>
</html>
"""

from fastapi.responses import HTMLResponse


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """HTML-страница для демонстрации стриминга в браузере."""
    return DEMO_HTML


# ─────────────────────────────────────────────
# 5. Запуск
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  AI Streaming API")
    print("=" * 60)
    print()
    print("  Демо в браузере: http://localhost:8002/demo")
    print("  Swagger UI:      http://localhost:8002/docs")
    print()
    print("  cURL стриминг:")
    print('    curl -N -X POST http://localhost:8002/chat/stream \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"message": "Расскажи про Python"}\'')
    print()

    uvicorn.run(app, host="0.0.0.0", port=8002)

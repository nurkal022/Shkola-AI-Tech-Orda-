"""
Лекция 22: Создание производственного API с FastAPI
03 — Асинхронные задачи (Background Tasks)
====================================================
Длительные задачи (генерация отчётов, batch-обработка)
выполняются в фоне — API не блокируется.
"""

import os
import uuid
import time
import asyncio
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="/Users/nurlykhan/TechOrda/.env")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI Async Tasks API", version="1.0.0")


# ─────────────────────────────────────────────
# 1. Хранилище задач (в продакшене — Redis/DB)
# ─────────────────────────────────────────────

# Для демонстрации храним в памяти.
# В продакшене используйте Redis, PostgreSQL или Celery.

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: str
    result: str | None = None
    error: str | None = None
    progress: int = Field(default=0, ge=0, le=100)


# In-memory storage
tasks: dict[str, TaskInfo] = {}


# ─────────────────────────────────────────────
# 2. Background Task — генерация отчёта
# ─────────────────────────────────────────────


class ReportRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=["Тренды искусственного интеллекта в 2025"],
    )
    sections: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Количество секций в отчёте",
    )


class TaskCreated(BaseModel):
    task_id: str
    status: str
    message: str


async def generate_report(task_id: str, topic: str, sections: int):
    """
    Фоновая задача: генерация отчёта по секциям.
    Обновляет статус и прогресс по мере выполнения.
    """
    task = tasks[task_id]
    task.status = TaskStatus.RUNNING

    try:
        report_parts = []

        for i in range(sections):
            # Обновляем прогресс
            task.progress = int((i / sections) * 100)

            # Генерируем секцию
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Ты аналитик. Пиши кратко и по делу, 2-3 предложения на секцию.",
                    },
                    {
                        "role": "user",
                        "content": f"Напиши секцию {i + 1} из {sections} "
                        f"для отчёта на тему: '{topic}'",
                    },
                ],
                max_tokens=300,
            )
            section_text = response.choices[0].message.content
            report_parts.append(f"## Секция {i + 1}\n{section_text}")

        # Готово!
        task.result = "\n\n".join(report_parts)
        task.status = TaskStatus.COMPLETED
        task.progress = 100

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)


@app.post("/report", response_model=TaskCreated)
async def create_report(request: ReportRequest, background_tasks: BackgroundTasks):
    """
    Запуск генерации отчёта в фоне.

    Возвращает task_id для отслеживания прогресса.
    Используйте GET /tasks/{task_id} для проверки статуса.
    """
    task_id = str(uuid.uuid4())[:8]

    tasks[task_id] = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=datetime.now().isoformat(),
    )

    # Запускаем в фоне — API сразу возвращает ответ
    background_tasks.add_task(generate_report, task_id, request.topic, request.sections)

    return TaskCreated(
        task_id=task_id,
        status="pending",
        message=f"Отчёт по теме '{request.topic}' генерируется. "
        f"Проверяйте статус: GET /tasks/{task_id}",
    )


# ─────────────────────────────────────────────
# 3. Проверка статуса задачи
# ─────────────────────────────────────────────


@app.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    """Получить статус и результат задачи."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Задача '{task_id}' не найдена")
    return tasks[task_id]


@app.get("/tasks", response_model=list[TaskInfo])
async def list_tasks():
    """Список всех задач."""
    return list(tasks.values())


# ─────────────────────────────────────────────
# 4. Batch-обработка — несколько вопросов сразу
# ─────────────────────────────────────────────


class BatchRequest(BaseModel):
    questions: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Список вопросов для обработки",
    )


class BatchAnswer(BaseModel):
    question: str
    answer: str


class BatchResult(BaseModel):
    task_id: str
    total: int
    completed: int
    results: list[BatchAnswer]
    status: TaskStatus


# Отдельное хранилище для batch задач
batch_tasks: dict[str, BatchResult] = {}


async def process_batch(task_id: str, questions: list[str]):
    """Обработка batch запросов параллельно с asyncio.gather."""
    batch = batch_tasks[task_id]
    batch.status = TaskStatus.RUNNING

    async def process_one(question: str) -> BatchAnswer:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Отвечай кратко, 1-2 предложения."},
                {"role": "user", "content": question},
            ],
            max_tokens=150,
        )
        return BatchAnswer(
            question=question,
            answer=response.choices[0].message.content,
        )

    try:
        # asyncio.gather — выполняем ВСЕ запросы параллельно!
        results = await asyncio.gather(
            *[process_one(q) for q in questions],
            return_exceptions=True,
        )

        for r in results:
            if isinstance(r, BatchAnswer):
                batch.results.append(r)
                batch.completed += 1
            else:
                batch.results.append(
                    BatchAnswer(question="?", answer=f"Ошибка: {r}")
                )
                batch.completed += 1

        batch.status = TaskStatus.COMPLETED

    except Exception as e:
        batch.status = TaskStatus.FAILED


@app.post("/batch", response_model=TaskCreated)
async def create_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Batch-обработка: отправить несколько вопросов одним запросом.

    Вопросы обрабатываются параллельно через asyncio.gather.
    """
    task_id = str(uuid.uuid4())[:8]

    batch_tasks[task_id] = BatchResult(
        task_id=task_id,
        total=len(request.questions),
        completed=0,
        results=[],
        status=TaskStatus.PENDING,
    )

    background_tasks.add_task(process_batch, task_id, request.questions)

    return TaskCreated(
        task_id=task_id,
        status="pending",
        message=f"Batch из {len(request.questions)} вопросов обрабатывается. "
        f"Проверяйте: GET /batch/{task_id}",
    )


@app.get("/batch/{task_id}", response_model=BatchResult)
async def get_batch(task_id: str):
    """Получить результат batch-обработки."""
    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail=f"Batch '{task_id}' не найден")
    return batch_tasks[task_id]


# ─────────────────────────────────────────────
# 5. Health + Запуск
# ─────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_tasks": len([t for t in tasks.values() if t.status == TaskStatus.RUNNING]),
        "total_tasks": len(tasks),
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  AI Async Tasks API")
    print("=" * 60)
    print()
    print("  Swagger UI: http://localhost:8003/docs")
    print()
    print("  Сценарий:")
    print("    1. POST /report — запуск генерации")
    print("    2. GET /tasks/{id} — проверка прогресса")
    print("    3. GET /tasks/{id} — получение результата")
    print()
    print("  Batch:")
    print('    curl -X POST http://localhost:8003/batch \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"questions": ["Что такое Python?", "Что такое FastAPI?"]}\'')
    print()

    uvicorn.run(app, host="0.0.0.0", port=8003)

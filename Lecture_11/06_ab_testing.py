"""
Лекция 11: A/B Testing для RAG
==============================
Сравнение разных конфигураций RAG системы.
Оценка различных стратегий и выбор оптимальной.
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import time
from typing import Dict, List
import json

load_dotenv()

# ============================================================
# Подготовка данных
# ============================================================
print("="*60)
print("📖 Подготовка данных")
print("="*60)

text = Path("../Lecture_07/data/Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt").read_text(encoding='utf-8')

print(f"   Загружено: {len(text):,} символов\n")

if not os.getenv("OPENAI_API_KEY"):
    print("   ⚠️ OPENAI_API_KEY не найден")
    exit(1)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm_gpt35 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_gpt4 = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Разные конфигурации RAG
# ============================================================
print("="*60)
print("🔧 Создание разных конфигураций RAG")
print("="*60)

def build_rag_config(chunk_size: int, top_k: int, model: str, use_rerank: bool = False):
    """Создаёт RAG конфигурацию."""
    # Чанкинг
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5,
    )
    chunks = splitter.split_text(text)
    
    # Vector store
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # LLM
    if model == "gpt-3.5":
        llm = llm_gpt35
    elif model == "gpt-4":
        llm = llm_gpt4
    else:
        llm = llm_gpt35
    
    return {
        "chunk_size": chunk_size,
        "top_k": top_k,
        "model": model,
        "use_rerank": use_rerank,
        "vectorstore": vectorstore,
        "llm": llm,
        "num_chunks": len(chunks),
    }


# Конфигурации для тестирования
configs = [
    {
        "name": "baseline",
        "chunk_size": 500,
        "top_k": 3,
        "model": "gpt-3.5",
        "use_rerank": False,
    },
    {
        "name": "large_chunks",
        "chunk_size": 1500,
        "top_k": 3,
        "model": "gpt-3.5",
        "use_rerank": False,
    },
    {
        "name": "more_context",
        "chunk_size": 500,
        "top_k": 5,
        "model": "gpt-3.5",
        "use_rerank": False,
    },
    {
        "name": "gpt4",
        "chunk_size": 500,
        "top_k": 3,
        "model": "gpt-4",
        "use_rerank": False,
    },
]

print(f"   Создано {len(configs)} конфигураций:\n")
for config in configs:
    print(f"   • {config['name']}: chunk_size={config['chunk_size']}, top_k={config['top_k']}, model={config['model']}")


# ============================================================
# Функция оценки конфигурации
# ============================================================
def evaluate_config(config_dict: Dict, test_questions: List[str]) -> Dict:
    """Оценивает конфигурацию RAG."""
    config = build_rag_config(
        config_dict["chunk_size"],
        config_dict["top_k"],
        config_dict["model"],
        config_dict["use_rerank"]
    )
    
    metrics = {
        "latency": [],
        "num_tokens": [],
        "answer_length": [],
    }
    
    for question in test_questions:
        start = time.time()
        
        # Retrieval
        docs = config["vectorstore"].similarity_search(question, k=config["top_k"])
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generation
        prompt = f"""Ответь на вопрос, используя только предоставленный контекст.

Контекст:
{context}

Вопрос: {question}

Ответ:"""
        
        response = config["llm"].invoke(prompt)
        answer = response.content
        
        latency = time.time() - start
        
        metrics["latency"].append(latency)
        metrics["num_tokens"].append(len(prompt.split()) + len(answer.split()))
        metrics["answer_length"].append(len(answer))
    
    # Средние значения
    return {
        "name": config_dict["name"],
        "avg_latency": sum(metrics["latency"]) / len(metrics["latency"]),
        "avg_tokens": sum(metrics["num_tokens"]) / len(metrics["num_tokens"]),
        "avg_answer_length": sum(metrics["answer_length"]) / len(metrics["answer_length"]),
        "num_chunks": config["num_chunks"],
    }


# ============================================================
# A/B Testing
# ============================================================
print("\n" + "="*60)
print("🧪 A/B TESTING")
print("="*60)

test_questions = [
    "Кто такой Гарри Поттер?",
    "Что такое Хогвартс?",
    "Кто такие Дурсли?",
]

print(f"\n   Тестовые вопросы ({len(test_questions)}):")
for q in test_questions:
    print(f"   • {q}")

print("\n   Оценка конфигураций...\n")

results = []
for config in configs:
    print(f"   ⏳ Тестирую {config['name']}...")
    result = evaluate_config(config, test_questions)
    results.append(result)

# Сортируем по latency
results.sort(key=lambda x: x["avg_latency"])


# ============================================================
# Результаты сравнения
# ============================================================
print("\n" + "="*60)
print("📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
print("="*60)

print("\n   Сравнительная таблица:")
print("   " + "─" * 80)
print(f"   {'Конфигурация':<20} {'Latency (s)':<15} {'Tokens':<12} {'Chunks':<10}")
print("   " + "─" * 80)

for result in results:
    print(f"   {result['name']:<20} {result['avg_latency']:<15.3f} {result['avg_tokens']:<12.0f} {result['num_chunks']:<10}")

print("   " + "─" * 80)

# Находим лучшую конфигурацию
best = results[0]
print(f"\n   🏆 Лучшая конфигурация: {best['name']}")
print(f"      • Latency: {best['avg_latency']:.3f}s")
print(f"      • Tokens: {best['avg_tokens']:.0f}")
print(f"      • Chunks: {best['num_chunks']}")


# ============================================================
# Детальный анализ
# ============================================================
print("\n" + "="*60)
print("📈 ДЕТАЛЬНЫЙ АНАЛИЗ")
print("="*60)

print("\n   Trade-offs:")

# Baseline vs Large chunks
baseline = next(r for r in results if r["name"] == "baseline")
large = next(r for r in results if r["name"] == "large_chunks")

print(f"\n   1. Baseline vs Large Chunks:")
print(f"      Baseline: {baseline['avg_latency']:.3f}s, {baseline['num_chunks']} chunks")
print(f"      Large: {large['avg_latency']:.3f}s, {large['num_chunks']} chunks")
if large['avg_latency'] < baseline['avg_latency']:
    print(f"      ✅ Large chunks быстрее на {((baseline['avg_latency'] - large['avg_latency']) / baseline['avg_latency'] * 100):.1f}%")
else:
    print(f"      ❌ Baseline быстрее на {((large['avg_latency'] - baseline['avg_latency']) / large['avg_latency'] * 100):.1f}%")

# Baseline vs More context
more_ctx = next(r for r in results if r["name"] == "more_context")
print(f"\n   2. Baseline vs More Context:")
print(f"      Baseline: {baseline['avg_latency']:.3f}s, top_k={3}")
print(f"      More Context: {more_ctx['avg_latency']:.3f}s, top_k={5}")
if more_ctx['avg_latency'] > baseline['avg_latency']:
    print(f"      ⚠️ Больше контекста = медленнее (но может быть точнее)")

# GPT-3.5 vs GPT-4
gpt4 = next(r for r in results if r["name"] == "gpt4")
print(f"\n   3. GPT-3.5 vs GPT-4:")
print(f"      GPT-3.5: {baseline['avg_latency']:.3f}s")
print(f"      GPT-4: {gpt4['avg_latency']:.3f}s")
if gpt4['avg_latency'] > baseline['avg_latency']:
    print(f"      ⚠️ GPT-4 медленнее на {((gpt4['avg_latency'] - baseline['avg_latency']) / baseline['avg_latency'] * 100):.1f}%")
    print(f"      💰 Но может быть точнее (нужна оценка качества)")


# ============================================================
# Рекомендации
# ============================================================
print("\n" + "="*60)
print("💡 РЕКОМЕНДАЦИИ ПО A/B TESTING")
print("="*60)
print("""
1. Что тестировать:
   • Размеры чанков (500, 1000, 1500)
   • Количество документов (top_k: 3, 5, 10)
   • Модели LLM (GPT-3.5, GPT-4, Claude)
   • Методы поиска (BM25, Semantic, Hybrid)
   • Reranking (с/без)

2. Метрики для сравнения:
   • Latency (время ответа)
   • Quality (точность, релевантность)
   • Cost (стоимость API вызовов)
   • Throughput (запросов в секунду)

3. Статистическая значимость:
   • Минимум 100 запросов на конфигурацию
   • Используйте t-test для сравнения
   • Учитывайте доверительные интервалы

4. Процесс:
   • Создайте тестовый набор вопросов
   • Запустите все конфигурации
   • Сравните метрики
   • Выберите оптимальную
   • Разверните в продакшен
   • Продолжайте мониторинг
""")

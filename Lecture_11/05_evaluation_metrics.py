"""
Лекция 11: Метрики оценки качества RAG
======================================
Как измерить качество RAG системы?
Метрики для оценки retrieval и generation.
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import time
from typing import List, Dict

load_dotenv()

# ============================================================
# Подготовка данных
# ============================================================
print("="*60)
print("📖 Подготовка данных")
print("="*60)

text = Path("../Lecture_07/data/Rouling_Djoann_[Garri_Potter#1]_Garri_Potter_i_Filosofskiy_kamen.txt").read_text(encoding='utf-8')

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = splitter.split_text(text)

print(f"   Загружено: {len(chunks)} чанков\n")

if os.getenv("OPENAI_API_KEY"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print("   ✅ Индексы созданы\n")
else:
    vectorstore = None
    llm = None
    print("   ⚠️ OPENAI_API_KEY не найден\n")


# ============================================================
# Метрики Retrieval
# ============================================================
print("="*60)
print("1️⃣ МЕТРИКИ RETRIEVAL")
print("="*60)

def calculate_retrieval_metrics(
    retrieved_docs: List[Document],
    relevant_doc_ids: List[int],
    k: int
) -> Dict[str, float]:
    """
    Вычисляет метрики качества retrieval.
    
    Args:
        retrieved_docs: Полученные документы
        relevant_doc_ids: ID релевантных документов (ground truth)
        k: Количество полученных документов
    
    Returns:
        Словарь с метриками
    """
    # Для простоты считаем, что chunk_id = индекс в списке chunks
    retrieved_ids = set(range(len(retrieved_docs)))
    relevant_ids = set(relevant_doc_ids)
    
    # Precision@K: Доля релевантных среди полученных
    if len(retrieved_ids) > 0:
        precision = len(retrieved_ids & relevant_ids) / len(retrieved_ids)
    else:
        precision = 0.0
    
    # Recall@K: Доля найденных релевантных
    if len(relevant_ids) > 0:
        recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
    else:
        recall = 0.0
    
    # F1 Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    # MRR (Mean Reciprocal Rank): Средний обратный ранг первого релевантного
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            mrr = 1.0 / rank
            break
    
    return {
        "precision@k": precision,
        "recall@k": recall,
        "f1_score": f1,
        "mrr": mrr,
    }


# Демонстрация
if vectorstore:
    query = "Как выглядит шрам Гарри Поттера?"
    
    # Получаем документы
    results = vectorstore.similarity_search_with_score(query, k=5)
    retrieved_docs = [Document(page_content=doc.page_content) for doc, _ in results]
    
    # Предположим, что документы 0, 2, 4 релевантны (для демонстрации)
    relevant_doc_ids = [0, 2, 4]
    
    metrics = calculate_retrieval_metrics(retrieved_docs, relevant_doc_ids, k=5)
    
    print(f"   Запрос: «{query}»\n")
    print(f"   Метрики retrieval:")
    print(f"   • Precision@5: {metrics['precision@k']:.3f}")
    print(f"   • Recall@5: {metrics['recall@k']:.3f}")
    print(f"   • F1 Score: {metrics['f1_score']:.3f}")
    print(f"   • MRR: {metrics['mrr']:.3f}")


# ============================================================
# Метрики Generation
# ============================================================
print("\n" + "="*60)
print("2️⃣ МЕТРИКИ GENERATION")
print("="*60)

def calculate_answer_relevance(answer: str, query: str) -> float:
    """
    Оценивает релевантность ответа к вопросу.
    Упрощённая версия (в реальности нужен LLM или специальная модель).
    """
    # Простая эвристика: проверяем наличие ключевых слов из запроса
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    if len(query_words) == 0:
        return 0.0
    
    overlap = len(query_words & answer_words) / len(query_words)
    return min(overlap, 1.0)


def calculate_faithfulness(answer: str, context: str) -> float:
    """
    Оценивает, основан ли ответ на предоставленном контексте.
    Упрощённая версия.
    """
    # Простая эвристика: проверяем наличие общих фраз
    answer_sentences = answer.split('.')
    context_sentences = context.split('.')
    
    matches = 0
    for ans_sent in answer_sentences:
        for ctx_sent in context_sentences:
            # Проверяем наличие общих слов
            ans_words = set(ans_sent.lower().split())
            ctx_words = set(ctx_sent.lower().split())
            if len(ans_words & ctx_words) > 3:  # Порог совпадения
                matches += 1
                break
    
    if len(answer_sentences) == 0:
        return 0.0
    
    return min(matches / len(answer_sentences), 1.0)


# Демонстрация
if llm and vectorstore:
    query = "Кто такой Гарри Поттер?"
    
    # RAG pipeline
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Ответь на вопрос, используя только предоставленный контекст.

Контекст:
{context}

Вопрос: {query}

Ответ:"""
    
    start = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start
    
    answer = response.content
    
    # Метрики
    relevance = calculate_answer_relevance(answer, query)
    faithfulness = calculate_faithfulness(answer, context)
    
    print(f"   Запрос: «{query}»\n")
    print(f"   Ответ: {answer[:150]}...\n")
    print(f"   Метрики generation:")
    print(f"   • Answer Relevance: {relevance:.3f}")
    print(f"   • Faithfulness: {faithfulness:.3f}")
    print(f"   • Latency: {latency:.3f}s")


# ============================================================
# Полная оценка RAG
# ============================================================
print("\n" + "="*60)
print("3️⃣ ПОЛНАЯ ОЦЕНКА RAG СИСТЕМЫ")
print("="*60)

def evaluate_rag_system(
    vectorstore,
    llm,
    test_questions: List[str],
    ground_truth: Dict[str, Dict]  # question -> {relevant_docs: [...], expected_answer: "..."}
) -> Dict[str, float]:
    """
    Полная оценка RAG системы.
    
    Args:
        vectorstore: Векторное хранилище
        llm: LLM модель
        test_questions: Список тестовых вопросов
        ground_truth: Ground truth данные
    
    Returns:
        Словарь с метриками
    """
    all_metrics = {
        "retrieval_precision": [],
        "retrieval_recall": [],
        "retrieval_f1": [],
        "answer_relevance": [],
        "faithfulness": [],
        "latency": [],
    }
    
    for question in test_questions:
        start = time.time()
        
        # Retrieval
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generation
        prompt = f"""Ответь на вопрос, используя только предоставленный контекст.

Контекст:
{context}

Вопрос: {question}

Ответ:"""
        response = llm.invoke(prompt)
        answer = response.content
        
        latency = time.time() - start
        
        # Метрики retrieval
        if question in ground_truth:
            gt = ground_truth[question]
            relevant_docs = gt.get("relevant_docs", [])
            
            ret_metrics = calculate_retrieval_metrics(docs, relevant_docs, k=5)
            all_metrics["retrieval_precision"].append(ret_metrics["precision@k"])
            all_metrics["retrieval_recall"].append(ret_metrics["recall@k"])
            all_metrics["retrieval_f1"].append(ret_metrics["f1_score"])
        
        # Метрики generation
        all_metrics["answer_relevance"].append(calculate_answer_relevance(answer, question))
        all_metrics["faithfulness"].append(calculate_faithfulness(answer, context))
        all_metrics["latency"].append(latency)
    
    # Средние значения
    return {
        metric: sum(values) / len(values) if values else 0.0
        for metric, values in all_metrics.items()
    }


# Демонстрация
if vectorstore and llm:
    test_questions = [
        "Кто такой Гарри Поттер?",
        "Что такое Хогвартс?",
    ]
    
    # Упрощённый ground truth (в реальности нужна разметка)
    ground_truth = {
        "Кто такой Гарри Поттер?": {
            "relevant_docs": [0, 1, 2],
            "expected_answer": "Гарри Поттер - мальчик-волшебник..."
        },
        "Что такое Хогвартс?": {
            "relevant_docs": [5, 6, 7],
            "expected_answer": "Хогвартс - школа магии..."
        },
    }
    
    print("   Оценка RAG системы на тестовых вопросах:\n")
    
    metrics = evaluate_rag_system(vectorstore, llm, test_questions, ground_truth)
    
    print("   Средние метрики:")
    print(f"   • Retrieval Precision: {metrics['retrieval_precision']:.3f}")
    print(f"   • Retrieval Recall: {metrics['retrieval_recall']:.3f}")
    print(f"   • Retrieval F1: {metrics['retrieval_f1']:.3f}")
    print(f"   • Answer Relevance: {metrics['answer_relevance']:.3f}")
    print(f"   • Faithfulness: {metrics['faithfulness']:.3f}")
    print(f"   • Latency: {metrics['latency']:.3f}s")


"""
Лекция 14: Self-Correcting Agent (Самокоррекция)
==================================================
Агент генерирует ответ → Критик проверяет → При необходимости переделывает.
НОВОЕ: Чего не было в AgentExecutor!
"""

from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден")
    exit(1)

# ============================================================
# 1. Состояние
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ============================================================
# 2. LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 3. Узлы графа
# ============================================================

def generator_node(state: AgentState) -> dict:
    """Узел: генерация ответа."""
    print("   📝 [Generator] Генерирую ответ...")
    system = SystemMessage(content="Ты помощник. Отвечай кратко и по делу. На русском.")
    response = llm.invoke([system] + list(state["messages"]))
    return {"messages": [response]}


def critic_node(state: AgentState) -> dict:
    """Узел: критика ответа."""
    last_msg = state["messages"][-1]
    user_q = state["messages"][0].content if state["messages"] else ""
    
    print("   🔍 [Critic] Проверяю качество ответа...")
    
    critic_prompt = f"""Проверь ответ на вопрос.

Вопрос: {user_q}
Ответ: {last_msg.content}

Оцени по шкале 1-10:
- 10 = отличный, полный, точный ответ
- 1 = плохой, неполный или неверный ответ

Ответь ТОЛЬКО числом от 1 до 10."""
    
    score_msg = llm.invoke([HumanMessage(content=critic_prompt)])
    score_text = score_msg.content.strip()
    
    # Пытаемся извлечь число
    score = 5
    for char in score_text:
        if char.isdigit():
            score = min(10, max(1, int(char)))
            break
    
    print(f"   🔍 [Critic] Оценка: {score}/10")
    
    # Добавляем оценку в сообщения для следующего узла
    return {"messages": [HumanMessage(content=f"[Оценка критика: {score}/10]")]}


def should_retry(state: AgentState) -> str:
    """Условный переход: переделывать или конец."""
    # Ищем сообщение с оценкой
    for msg in reversed(list(state["messages"])):
        if hasattr(msg, 'content') and "Оценка критика" in str(msg.content):
            score = 5
            for part in str(msg.content).split():
                if part.isdigit():
                    score = int(part)
                    break
            if score < 7:
                return "retry"
            break
    return "end"


def retry_node(state: AgentState) -> dict:
    """Узел: перегенерация с учётом критики."""
    print("   🔄 [Retry] Переделываю ответ с учётом критики...")
    # Берём оригинальный вопрос (первое сообщение)
    original_question = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and "[Оценка критика" not in str(msg.content):
            original_question = msg.content
            break
    
    if not original_question:
        original_question = "Ответь на вопрос"
    
    system = SystemMessage(content="""Ты помощник. Предыдущий ответ получил низкую оценку.
Дай ЛУЧШИЙ, более полный и точный ответ. На русском.""")
    
    response = llm.invoke([
        system,
        HumanMessage(content=original_question),
    ])
    return {"messages": [response]}


# Упрощённая версия: фиксированное количество итераций
def generator_then_critic(state: AgentState) -> dict:
    """Комбинированный узел: генерация + критика."""
    # Генерация
    print("   📝 [Generator] Генерирую ответ...")
    system = SystemMessage(content="Ты помощник. Отвечай кратко. На русском.")
    response = llm.invoke([system] + list(state["messages"]))
    
    # Критика
    print("   🔍 [Critic] Проверяю...")
    critic_prompt = f"Вопрос: {state['messages'][0].content}\nОтвет: {response.content}\nОценка 1-10 (только число):"
    score_msg = llm.invoke([HumanMessage(content=critic_prompt)])
    score = 5
    for c in score_msg.content.strip():
        if c.isdigit():
            score = min(10, max(1, int(c)))
            break
    print(f"   🔍 [Critic] Оценка: {score}/10")
    
    if score < 7:
        print("   🔄 [Retry] Переделываю...")
        retry_response = llm.invoke([
            SystemMessage(content="Дай лучший, более полный ответ. На русском."),
            HumanMessage(content=state["messages"][0].content),
        ])
        return {"messages": [retry_response]}
    
    return {"messages": [response]}

# ============================================================
# 4. Граф (упрощённый: один узел с генерацией+критикой+retry)
# ============================================================

workflow = StateGraph(AgentState)
workflow.add_node("generate", generator_then_critic)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()

# ============================================================
# 5. Использование
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🔄 Self-Correcting Agent Demo")
    print("="*60)
    print("""
   Агент генерирует ответ → Критик проверяет → При низкой оценке переделывает.
   
   Введите 'exit' для выхода.
""")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🤔 Ваш вопрос: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input:
                continue
            
            print()
            result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
            print(f"\n✅ Финальный ответ: {result['messages'][-1].content}")
        except KeyboardInterrupt:
            break

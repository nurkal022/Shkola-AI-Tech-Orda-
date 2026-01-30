"""
Лекция 14: ReAct агент на LangGraph
====================================
Тот же ReAct что в Лекции 13, но на LangGraph.
Показывает: agent → tools → agent (цикл) с условными переходами.
"""

import json
from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден")
    exit(1)

# ============================================================
# 1. Инструменты
# ============================================================

@tool
def calculate(expression: str) -> str:
    """Вычисляет математическое выражение."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool
def get_current_date() -> str:
    """Возвращает текущую дату."""
    from datetime import datetime
    return f"Текущая дата: {datetime.now().strftime('%d.%m.%Y')}"


tools = [calculate, get_current_date]
tools_by_name = {t.name: t for t in tools}

# ============================================================
# 2. Состояние агента
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ============================================================
# 3. LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = llm.bind_tools(tools)

# ============================================================
# 4. Узлы графа
# ============================================================

def call_model(state: AgentState) -> dict:
    """Узел: вызов LLM."""
    system = SystemMessage(content="Ты помощник. Используй инструменты для вычислений. Отвечай на русском.")
    response = model.invoke([system] + list(state["messages"]))
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Узел: выполнение инструментов."""
    last_message = state["messages"][-1]
    outputs = []
    for tool_call in last_message.tool_calls:
        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(ToolMessage(
            content=str(result),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        ))
    return {"messages": outputs}


def should_continue(state: AgentState) -> str:
    """Условный переход: tool или end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

# ============================================================
# 5. Создание графа
# ============================================================

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# ============================================================
# 6. Использование
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🤖 ReAct агент на LangGraph")
    print("="*60)
    print()
    
    # Демонстрация
    query = "Сколько будет 25 * 17?"
    print(f"Вопрос: {query}\n")
    
    for event in graph.stream({"messages": [HumanMessage(content=query)]}, stream_mode="values"):
        for message in event.get("messages", []):
            if isinstance(message, AIMessage) and message.content:
                print(f"🤖 Ответ: {message.content}")
            elif isinstance(message, ToolMessage):
                print(f"🔧 Tool: {message.content[:50]}...")
    
    # Финальный результат
    result = graph.invoke({"messages": [HumanMessage(content=query)]})
    last_msg = result["messages"][-1]
    print(f"\n✅ Финальный ответ: {last_msg.content}")
    
    # Интерактивный режим
    print("\n" + "="*60)
    print("💬 Интерактивный режим (exit для выхода)")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🤔 Ваш вопрос: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input:
                continue
            
            result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
            print(f"\n✅ {result['messages'][-1].content}")
        except KeyboardInterrupt:
            break

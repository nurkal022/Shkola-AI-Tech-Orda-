"""
Лекция 14: Human-in-the-Loop
=============================
Агент спрашивает подтверждение у человека перед выполнением действий.
НОВОЕ: Чего не было в AgentExecutor!
"""

import json
from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY не найден")
    exit(1)

# ============================================================
# 1. Инструменты (включая "опасный")
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
def send_email(to: str, subject: str, body: str) -> str:
    """Отправляет email. ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ!"""
    return f"[СИМУЛЯЦИЯ] Email отправлен: to={to}, subject={subject}"


tools = [calculate, send_email]
tools_by_name = {t.name: t for t in tools}

# ============================================================
# 2. Состояние
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ============================================================
# 3. LLM и узлы
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = llm.bind_tools(tools)


def call_model(state: AgentState) -> dict:
    system = SystemMessage(content="""Ты помощник. Используй инструменты.
Для send_email - предупреждай что нужна проверка.
Отвечай на русском.""")
    response = model.invoke([system] + list(state["messages"]))
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    outputs = []
    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        
        # Human-in-the-loop: для send_email спрашиваем подтверждение
        if name == "send_email":
            print("\n" + "─"*60)
            print("⚠️  HUMAN-IN-THE-LOOP: Требуется подтверждение!")
            print("─"*60)
            print(f"   Действие: Отправить email")
            print(f"   Кому: {args.get('to', '?')}")
            print(f"   Тема: {args.get('subject', '?')}")
            print(f"   Текст: {args.get('body', '?')[:50]}...")
            print("─"*60)
            confirm = input("   Подтвердить? (y/n): ").strip().lower()
            if confirm != "y":
                outputs.append(ToolMessage(
                    content="Пользователь отклонил отправку email.",
                    name=name,
                    tool_call_id=tool_call["id"],
                ))
                continue
        
        result = tools_by_name[name].invoke(args)
        outputs.append(ToolMessage(content=str(result), name=name, tool_call_id=tool_call["id"]))
    
    return {"messages": outputs}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else "end"

# ============================================================
# 4. Граф
# ============================================================

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# ============================================================
# 5. Использование
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("👤 Human-in-the-Loop Demo")
    print("="*60)
    print("""
   Агент спрашивает подтверждение перед важными действиями.
   Примеры:
   • "Сколько будет 25 * 17?" - без подтверждения
   • "Отправь email Ивану с темой Привет" - с подтверждением
   
   Введите 'exit' для выхода.
""")
    print("="*60)
    
    config = {"configurable": {"thread_id": "human_loop_demo"}}
    
    while True:
        try:
            user_input = input("\n🤔 Ваш запрос: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input:
                continue
            
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            print(f"\n✅ Ответ: {result['messages'][-1].content}")
        except KeyboardInterrupt:
            break

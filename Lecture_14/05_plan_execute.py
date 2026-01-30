"""
Лекция 14: Plan-and-Execute
============================
Агент сначала ПЛАНИРУЕТ шаги, потом ВЫПОЛНЯЕТ по одному.
НОВОЕ: Чего не было в AgentExecutor!
"""

from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import json
import re

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
        return str(result)
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool
def get_current_date() -> str:
    """Возвращает текущую дату."""
    from datetime import datetime
    return datetime.now().strftime("%d.%m.%Y")


tools = [calculate, get_current_date]
tools_by_name = {t.name: t for t in tools}

# ============================================================
# 2. Состояние
# ============================================================

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: list
    current_step: int

# ============================================================
# 3. LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model_with_tools = llm.bind_tools(tools)

# ============================================================
# 4. Узлы графа
# ============================================================

def planner_node(state: AgentState) -> dict:
    """Узел: создание плана."""
    print("   📋 [Planner] Создаю план...")
    
    question = state["messages"][0].content if state["messages"] else ""
    
    plan_prompt = f"""Задача: {question}

Создай план из 1-5 шагов. Каждый шаг - одно действие.
Формат (каждый шаг с новой строки):
1. [действие]
2. [действие]
...

Только план, без объяснений."""

    response = llm.invoke([HumanMessage(content=plan_prompt)])
    plan_text = response.content
    
    # Парсим план
    plan = []
    for line in plan_text.strip().split("\n"):
        line = line.strip()
        if re.match(r"^\d+[\.\)]\s*", line):
            step = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if step:
                plan.append(step)
    
    if not plan:
        plan = [question]
    
    print("   📋 [Planner] План:")
    for i, step in enumerate(plan, 1):
        print(f"      {i}. {step[:50]}...")
    
    return {"plan": plan, "current_step": 0, "messages": [response]}


def executor_node(state: AgentState) -> dict:
    """Узел: выполнение текущего шага плана."""
    plan = state.get("plan") or []
    step_idx = state.get("current_step") or 0
    
    if step_idx >= len(plan):
        return {"current_step": step_idx}
    
    current_step = plan[step_idx]
    print(f"\n   ⚙️  [Executor] Шаг {step_idx + 1}/{len(plan)}: {current_step[:50]}...")
    
    # Используем LLM с инструментами для выполнения шага
    system = SystemMessage(content="""Выполни этот шаг плана. Используй инструменты если нужно.
Отвечай кратко - только результат шага.""")
    
    response = model_with_tools.invoke([
        system,
        HumanMessage(content=f"Выполни: {current_step}"),
    ])
    
    # Если есть tool_calls - выполняем
    if response.tool_calls:
        from langchain_core.messages import ToolMessage
        tool_results = []
        for tc in response.tool_calls:
            result = tools_by_name[tc["name"]].invoke(tc["args"])
            tool_results.append(ToolMessage(content=str(result), name=tc["name"], tool_call_id=tc["id"]))
        # Повторный вызов с результатами
        final = model_with_tools.invoke([system, HumanMessage(content=f"Выполни: {current_step}")] + [response] + tool_results)
        step_result = final.content
    else:
        step_result = response.content
    
    print(f"   ⚙️  [Executor] Результат: {step_result[:80]}...")
    
    return {
        "current_step": step_idx + 1,
        "messages": [AIMessage(content=f"Шаг {step_idx + 1}: {step_result}")],
    }


def should_continue_execution(state: AgentState) -> str:
    """Условный переход: следующий шаг или конец."""
    plan = state.get("plan") or []
    step_idx = state.get("current_step") or 0
    if step_idx < len(plan):
        return "execute"
    return "end"


def synthesizer_node(state: AgentState) -> dict:
    """Узел: синтез финального ответа."""
    print("\n   📝 [Synthesizer] Формирую финальный ответ...")
    
    question = state["messages"][0].content if state["messages"] else ""
    steps_results = [m.content for m in state["messages"] if isinstance(m, AIMessage) and "Шаг" in str(m.content)]
    
    synth_prompt = f"""Задача: {question}

Результаты шагов:
{chr(10).join(steps_results)}

Дай краткий итоговый ответ пользователю."""

    response = llm.invoke([HumanMessage(content=synth_prompt)])
    return {"messages": [response]}

# ============================================================
# 5. Граф
# ============================================================

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", should_continue_execution, {"execute": "executor", "end": "synthesizer"})
workflow.add_edge("synthesizer", END)

graph = workflow.compile()

# ============================================================
# 6. Использование
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("📋 Plan-and-Execute Demo")
    print("="*60)
    print("""
   Агент: 1) Планирует шаги 2) Выполняет по одному 3) Синтезирует ответ
   
   Примеры:
   • Какая сегодня дата? Умножь день на 10
   • Посчитай 25*17 и прибавь 100
   
   Введите 'exit' для выхода.
""")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🤔 Ваша задача: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input:
                continue
            
            print()
            result = graph.invoke({"messages": [HumanMessage(content=user_input)], "plan": [], "current_step": 0})
            print(f"\n✅ Итог: {result['messages'][-1].content}")
        except KeyboardInterrupt:
            break

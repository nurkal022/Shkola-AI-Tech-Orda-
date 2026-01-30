"""
Лекция 14: Простой граф LangGraph
=================================
Базовое понимание StateGraph, узлов и рёбер.
"""

from langgraph.graph import START, END, StateGraph
from typing import TypedDict

# ============================================================
# 1. Определение состояния
# ============================================================

class State(TypedDict):
    text: str

# ============================================================
# 2. Узлы графа
# ============================================================

def node_a(state: State) -> dict:
    return {"text": state["text"] + "a"}

def node_b(state: State) -> dict:
    return {"text": state["text"] + "b"}

# ============================================================
# 3. Создание и компиляция графа
# ============================================================

graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", END)

compiled = graph.compile()

# ============================================================
# 4. Выполнение
# ============================================================

if __name__ == "__main__":
    result = compiled.invoke({"text": ""})
    print(f"Результат: {result}")  # {'text': 'ab'}

"""
Лекция 14: Live Demo - Пошаговое создание графа
================================================
Упрощённая версия для переписывания на лекции.
Раскомментируйте по этапам.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# ЭТАП 1: Импорты
# ============================================================
# from langgraph.graph import START, END, StateGraph
# from typing import TypedDict

# ============================================================
# ЭТАП 2: Состояние
# ============================================================
# class State(TypedDict):
#     text: str

# ============================================================
# ЭТАП 3: Узлы
# ============================================================
# def node_a(state: State) -> dict:
#     return {"text": state["text"] + "a"}
#
# def node_b(state: State) -> dict:
#     return {"text": state["text"] + "b"}

# ============================================================
# ЭТАП 4: Граф
# ============================================================
# graph = StateGraph(State)
# graph.add_node("node_a", node_a)
# graph.add_node("node_b", node_b)
# graph.add_edge(START, "node_a")
# graph.add_edge("node_a", "node_b")
# graph.add_edge("node_b", END)

# ============================================================
# ЭТАП 5: Выполнение
# ============================================================
# compiled = graph.compile()
# result = compiled.invoke({"text": ""})
# print(result)  # {'text': 'ab'}

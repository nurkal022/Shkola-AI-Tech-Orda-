"""
Лекция 14: Live Demo - Простой граф LangGraph
==============================================
Пошаговая демонстрация создания простого графа.
Можно переписывать на лекции по этапам.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# ЭТАП 1: Импорты
# ============================================================
print("="*60)
print("📦 ЭТАП 1: ИМПОРТЫ")
print("="*60)
print()

from langgraph.graph import START, END, StateGraph
from typing import TypedDict

print("✅ Импортировали:")
print("   • START, END - специальные узлы графа")
print("   • StateGraph - граф состояний")
print("   • TypedDict - для определения состояния")
print()

# ============================================================
# ЭТАП 2: Определение состояния (State)
# ============================================================
print("="*60)
print("📋 ЭТАП 2: ОПРЕДЕЛЕНИЕ СОСТОЯНИЯ")
print("="*60)
print()

class State(TypedDict):
    """Состояние графа - что передаётся между узлами."""
    text: str

print("✅ State определён:")
print("   • text: str - строка, которая будет модифицироваться")
print("   • Каждый узел получает state и возвращает обновления")
print()

# ============================================================
# ЭТАП 3: Определение узлов (Nodes)
# ============================================================
print("="*60)
print("🔧 ЭТАП 3: ОПРЕДЕЛЕНИЕ УЗЛОВ")
print("="*60)
print()

def node_a(state: State) -> dict:
    """Узел A: добавляет 'a' к тексту."""
    return {"text": state["text"] + "a"}

def node_b(state: State) -> dict:
    """Узел B: добавляет 'b' к тексту."""
    return {"text": state["text"] + "b"}

print("✅ Узлы созданы:")
print("   • node_a: state['text'] + 'a'")
print("   • node_b: state['text'] + 'b'")
print("   • Узел = функция(state) -> dict с обновлениями")
print()

# ============================================================
# ЭТАП 4: Создание графа
# ============================================================
print("="*60)
print("🕸️  ЭТАП 4: СОЗДАНИЕ ГРАФА")
print("="*60)
print()

graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)

print("✅ Граф создан:")
print("   • StateGraph(State) - граф с нашим состоянием")
print("   • add_node('node_a', node_a) - добавляем узел")
print("   • add_node('node_b', node_b) - добавляем узел")
print()

# ============================================================
# ЭТАП 5: Добавление рёбер (Edges)
# ============================================================
print("="*60)
print("🔗 ЭТАП 5: ДОБАВЛЕНИЕ РЁБЕР")
print("="*60)
print()

graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", END)

print("✅ Рёбра добавлены:")
print("   • START → node_a (вход в граф)")
print("   • node_a → node_b (последовательное выполнение)")
print("   • node_b → END (выход из графа)")
print()
print("   Схема: START → node_a → node_b → END")
print()

# ============================================================
# ЭТАП 6: Компиляция и выполнение
# ============================================================
print("="*60)
print("🚀 ЭТАП 6: КОМПИЛЯЦИЯ И ВЫПОЛНЕНИЕ")
print("="*60)
print()

compiled = graph.compile()

print("✅ Граф скомпилирован: graph.compile()")
print()
print("⏳ Выполняем: invoke({'text': ''})")
print("   Ожидаемый путь: '' → 'a' → 'ab'")
print()

result = compiled.invoke({"text": ""})

print("─" * 60)
print(f"✅ РЕЗУЛЬТАТ: {result}")
print("─" * 60)
print()

# ============================================================
# ЭТАП 7: Визуализация графа
# ============================================================
print("="*60)
print("📊 ЭТАП 7: ВИЗУАЛИЗАЦИЯ")
print("="*60)
print()

try:
    mermaid_code = compiled.get_graph().draw_mermaid()
    print("Mermaid код графа:")
    print("─" * 60)
    print(mermaid_code)
    print("─" * 60)
    print("   Скопируйте в https://mermaid.live для визуализации")
except Exception as e:
    print(f"   (Визуализация недоступна: {e})")

print()
print("="*60)
print("✅ LIVE DEMO ЗАВЕРШЕНА")
print("="*60)
print("""
   Ключевые концепции LangGraph:
   1. State - что передаётся между узлами
   2. Node - функция обработки
   3. Edge - переход между узлами
   4. START/END - вход и выход графа
   5. compile() - компиляция перед выполнением
""")

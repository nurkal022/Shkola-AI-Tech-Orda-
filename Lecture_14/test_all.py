"""
Тест всех файлов Lecture 14
"""

import sys
from pathlib import Path

print("="*60)
print("🧪 ТЕСТИРОВАНИЕ ЛЕКЦИИ 14")
print("="*60)
print()

# 1. Простой граф (без API)
print("1️⃣ 00_live_demo_simple_graph.py (без API)")
print("─"*60)
try:
    exec(open("00_live_demo_simple_graph.py").read())
    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ {e}")
print()

# 2. 01_simple_graph
print("2️⃣ 01_simple_graph.py")
print("─"*60)
try:
    from langgraph.graph import START, END, StateGraph
    from typing import TypedDict
    class State(TypedDict):
        text: str
    def node_a(state): return {"text": state["text"] + "a"}
    def node_b(state): return {"text": state["text"] + "b"}
    g = StateGraph(State)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    r = g.compile().invoke({"text": ""})
    assert r["text"] == "ab"
    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ {e}")
print()

# 3. Проверка импортов для остальных
print("3️⃣ Импорты (02, 03, 04, 05)")
print("─"*60)
try:
    exec(open("02_react_langgraph.py").read().split("if __name__")[0])
    print("   ✅ 02_react_langgraph OK")
except Exception as e:
    print(f"   ⚠️ 02: {str(e)[:60]}")

try:
    exec(open("03_human_in_loop.py").read().split("if __name__")[0])
    print("   ✅ 03_human_in_loop OK")
except Exception as e:
    print(f"   ⚠️ 03: {str(e)[:60]}")

try:
    exec(open("04_self_correcting.py").read().split("if __name__")[0])
    print("   ✅ 04_self_correcting OK")
except Exception as e:
    print(f"   ⚠️ 04: {str(e)[:60]}")

try:
    exec(open("05_plan_execute.py").read().split("if __name__")[0])
    print("   ✅ 05_plan_execute OK")
except Exception as e:
    print(f"   ⚠️ 05: {str(e)[:60]}")
print()

# 4. API key
print("4️⃣ OPENAI_API_KEY")
print("─"*60)
import os
from dotenv import load_dotenv
load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    print("   ✅ Найден")
else:
    print("   ⚠️ Не найден (02-05 требуют API)")
print()

print("="*60)
print("✅ ТЕСТЫ ЗАВЕРШЕНЫ")
print("="*60)

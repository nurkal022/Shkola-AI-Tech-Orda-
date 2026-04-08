"""
Лекция 24: Безопасность в продакшене
04 — Model Context Protocol (MCP)
====================================================
MCP — открытый стандарт от Anthropic для безопасного
подключения LLM к внешним инструментам и данным.

Этот файл объясняет концепцию MCP и показывает
как строить MCP-совместимые серверы на Python.
"""

# ============================================================
# 1. Что такое MCP?
# ============================================================

print("=" * 60)
print("1. ЧТО ТАКОЕ MODEL CONTEXT PROTOCOL (MCP)?")
print("=" * 60)
print()
print("  MCP — открытый протокол для подключения LLM к внешним ресурсам.")
print()
print("  Проблема без MCP:")
print("    - Каждый LLM-провайдер имеет свой формат tool calling")
print("    - Каждая интеграция (GitHub, Slack, DB) — кастомный код")
print("    - Нет стандарта безопасности для доступа к данным")
print()
print("  Решение MCP:")
print("    - Единый протокол для ВСЕХ LLM и ВСЕХ инструментов")
print("    - Как USB для AI — подключил и работает")
print("    - Встроенные механизмы безопасности")
print()
print("  Архитектура MCP:")
print()
print("    ┌──────────┐     ┌──────────────┐     ┌──────────────┐")
print("    │ LLM App  │────>│  MCP Client  │────>│  MCP Server  │")
print("    │(Claude,  │     │ (в приложении)│     │(инструменты, │")
print("    │ GPT,etc) │<────│              │<────│  данные)     │")
print("    └──────────┘     └──────────────┘     └──────────────┘")
print()
print("  MCP Server предоставляет:")
print("    - Tools    — функции, которые LLM может вызывать")
print("    - Resources — данные, которые LLM может читать")
print("    - Prompts   — шаблоны промптов")
print()

# ============================================================
# 2. Пример MCP Server (концептуальный)
# ============================================================

print("=" * 60)
print("2. ПРИМЕР MCP SERVER")
print("=" * 60)
print()
print("  MCP Server — это JSON-RPC 2.0 сервер.")
print("  Ниже — концептуальная реализация.")
print()

import json
from datetime import datetime


class MCPServer:
    """
    Концептуальный MCP Server.
    В реальности используйте: pip install mcp

    Каждый сервер объявляет свои capabilities:
    - tools: функции для вызова
    - resources: данные для чтения
    """

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.tools: dict[str, dict] = {}
        self.resources: dict[str, dict] = {}

    def tool(self, name: str, description: str, parameters: dict):
        """Регистрирует инструмент."""
        def decorator(func):
            self.tools[name] = {
                "name": name,
                "description": description,
                "inputSchema": {
                    "type": "object",
                    "properties": parameters,
                },
                "handler": func,
            }
            return func
        return decorator

    def resource(self, uri: str, description: str):
        """Регистрирует ресурс."""
        def decorator(func):
            self.resources[uri] = {
                "uri": uri,
                "description": description,
                "handler": func,
            }
            return func
        return decorator

    def handle_request(self, method: str, params: dict = None) -> dict:
        """Обрабатывает JSON-RPC запрос."""
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": self.name, "version": self.version},
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"listChanged": True},
                },
            }
        elif method == "tools/list":
            return {
                "tools": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "inputSchema": t["inputSchema"],
                    }
                    for t in self.tools.values()
                ]
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            if tool_name not in self.tools:
                return {"error": f"Tool '{tool_name}' not found"}
            result = self.tools[tool_name]["handler"](**args)
            return {"content": [{"type": "text", "text": str(result)}]}
        elif method == "resources/list":
            return {
                "resources": [
                    {"uri": r["uri"], "description": r["description"]}
                    for r in self.resources.values()
                ]
            }
        elif method == "resources/read":
            uri = params.get("uri")
            if uri not in self.resources:
                return {"error": f"Resource '{uri}' not found"}
            result = self.resources[uri]["handler"]()
            return {"contents": [{"uri": uri, "text": str(result)}]}
        else:
            return {"error": f"Method '{method}' not supported"}


# ============================================================
# 3. Создаём MCP Server для банковского приложения
# ============================================================

print("=" * 60)
print("3. MCP SERVER: БАНКОВСКОЕ ПРИЛОЖЕНИЕ")
print("=" * 60)
print()

server = MCPServer(name="bank-tools", version="1.0.0")

# Эмуляция БД
ACCOUNTS = {
    "KZ001": {"name": "Алмас", "balance": 150_000, "currency": "KZT"},
    "KZ002": {"name": "Айгерим", "balance": 75_000, "currency": "KZT"},
}
TRANSACTIONS = [
    {"from": "KZ001", "to": "KZ002", "amount": 5000, "date": "2025-04-01"},
    {"from": "KZ002", "to": "KZ001", "amount": 2000, "date": "2025-04-05"},
]


@server.tool(
    name="get_balance",
    description="Получает баланс счёта по номеру",
    parameters={"account_id": {"type": "string", "description": "Номер счёта (KZ001, KZ002)"}},
)
def get_balance(account_id: str) -> str:
    if account_id not in ACCOUNTS:
        return f"Счёт {account_id} не найден"
    acc = ACCOUNTS[account_id]
    return f"Счёт {account_id} ({acc['name']}): {acc['balance']:,} {acc['currency']}"


@server.tool(
    name="get_transactions",
    description="Получает историю транзакций для счёта",
    parameters={"account_id": {"type": "string", "description": "Номер счёта"}},
)
def get_transactions(account_id: str) -> str:
    txns = [t for t in TRANSACTIONS if t["from"] == account_id or t["to"] == account_id]
    if not txns:
        return f"Нет транзакций для {account_id}"
    lines = [f"  {t['date']}: {t['from']} -> {t['to']}: {t['amount']:,} KZT" for t in txns]
    return f"Транзакции {account_id}:\n" + "\n".join(lines)


@server.tool(
    name="transfer",
    description="Переводит деньги между счетами",
    parameters={
        "from_account": {"type": "string"},
        "to_account": {"type": "string"},
        "amount": {"type": "number"},
    },
)
def transfer(from_account: str, to_account: str, amount: float) -> str:
    if from_account not in ACCOUNTS or to_account not in ACCOUNTS:
        return "Один из счетов не найден"
    if amount <= 0:
        return "Сумма должна быть положительной"
    if ACCOUNTS[from_account]["balance"] < amount:
        return "Недостаточно средств"

    ACCOUNTS[from_account]["balance"] -= amount
    ACCOUNTS[to_account]["balance"] += amount
    TRANSACTIONS.append({
        "from": from_account, "to": to_account,
        "amount": amount, "date": datetime.now().strftime("%Y-%m-%d"),
    })
    return f"Перевод {amount:,.0f} KZT: {from_account} -> {to_account}. Успешно!"


@server.resource(
    uri="bank://accounts/list",
    description="Список всех счетов",
)
def list_accounts():
    lines = [f"  {k}: {v['name']} ({v['balance']:,} {v['currency']})" for k, v in ACCOUNTS.items()]
    return "Счета:\n" + "\n".join(lines)


# ============================================================
# 4. Тестируем MCP Server
# ============================================================

print("  Тестируем MCP Server:")
print()

# Initialize
result = server.handle_request("initialize")
print(f"  initialize: {result['serverInfo']['name']} v{result['serverInfo']['version']}")

# List tools
result = server.handle_request("tools/list")
print(f"  tools/list: {len(result['tools'])} инструментов")
for tool in result["tools"]:
    print(f"    - {tool['name']}: {tool['description']}")

# Call tool: get_balance
result = server.handle_request("tools/call", {"name": "get_balance", "arguments": {"account_id": "KZ001"}})
print(f"\n  get_balance(KZ001): {result['content'][0]['text']}")

# Call tool: get_transactions
result = server.handle_request("tools/call", {"name": "get_transactions", "arguments": {"account_id": "KZ001"}})
print(f"\n  get_transactions(KZ001):\n{result['content'][0]['text']}")

# Call tool: transfer
result = server.handle_request("tools/call", {
    "name": "transfer",
    "arguments": {"from_account": "KZ001", "to_account": "KZ002", "amount": 10000},
})
print(f"\n  transfer(KZ001 -> KZ002, 10000): {result['content'][0]['text']}")

# Check balance after transfer
result = server.handle_request("tools/call", {"name": "get_balance", "arguments": {"account_id": "KZ001"}})
print(f"  get_balance(KZ001) after: {result['content'][0]['text']}")

# List resources
result = server.handle_request("resources/list")
print(f"\n  resources/list: {len(result['resources'])} ресурсов")

# Read resource
result = server.handle_request("resources/read", {"uri": "bank://accounts/list"})
print(f"\n  resource bank://accounts/list:\n{result['contents'][0]['text']}")

# ============================================================
# 5. Безопасность в MCP
# ============================================================

print()
print("=" * 60)
print("5. БЕЗОПАСНОСТЬ В MCP")
print("=" * 60)
print()
print("  MCP включает механизмы безопасности:")
print()
print("  1. Consent (согласие пользователя)")
print("     - LLM ДОЛЖНА спрашивать разрешение перед вызовом tool")
print("     - Пользователь видит: 'AI хочет вызвать transfer(10000 KZT)'")
print("     - Пользователь: [Разрешить] / [Запретить]")
print()
print("  2. Scoping (ограничение доступа)")
print("     - Server объявляет capabilities")
print("     - Client может ограничить доступные tools")
print("     - Принцип наименьших привилегий")
print()
print("  3. Sandboxing (изоляция)")
print("     - MCP Server работает в отдельном процессе")
print("     - Не имеет доступа к памяти клиента")
print("     - Общение только через JSON-RPC")
print()
print("  4. Validation (валидация)")
print("     - inputSchema проверяет параметры")
print("     - Server валидирует ВСЕ входные данные")
print("     - Нельзя вызвать tool с невалидными аргументами")
print()
print("  5. Audit (аудит)")
print("     - Все вызовы логируются")
print("     - Можно отслеживать кто, что и когда вызывал")
print()

# ============================================================
# Итоги
# ============================================================

print("=" * 60)
print("ИТОГИ: MODEL CONTEXT PROTOCOL")
print("=" * 60)
print()
print("  MCP = стандартизированный протокол LLM <-> инструменты")
print()
print("  Преимущества:")
print("    - Один стандарт для всех LLM и всех инструментов")
print("    - Встроенная безопасность (consent, scoping, sandboxing)")
print("    - Разделение ответственности (client != server)")
print("    - Экосистема готовых серверов (GitHub, Slack, DB, и т.д.)")
print()
print("  Ресурсы:")
print("    - Спецификация: https://modelcontextprotocol.io")
print("    - Python SDK: pip install mcp")
print("    - Готовые серверы: https://github.com/modelcontextprotocol/servers")

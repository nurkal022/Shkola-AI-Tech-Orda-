# Лекция 24: Безопасность в продакшене и лучшие практики

## Файлы

| # | Файл | Тема |
|---|------|------|
| 0 | `00_live_demo.py` | Live Demo: prompt injection атака и защита |
| 1 | `01_prompt_injection.py` | 4 типа атак + 4-уровневая защита (regex, prompt, LLM-judge, output filter) |
| 2 | `02_input_validation.py` | Pydantic: Field, Enum, validators, cross-field validation |
| 3 | `03_rate_limiting.py` | Sliding window, token bucket, per-user tiered limits |
| 4 | `04_mcp_protocol.py` | Model Context Protocol: концепция, MCP Server, безопасность |
| 5 | `05_secure_api.py` | Полный проект: все практики вместе + audit log |

## Запуск

```bash
pip install -r requirements.txt

python 00_live_demo.py           # Live demo (TestClient)
python 01_prompt_injection.py    # Атаки и защита
python 02_input_validation.py    # Валидация (TestClient)
python 03_rate_limiting.py       # Rate limiting (TestClient)
python 04_mcp_protocol.py        # MCP (концептуальный)
python 05_secure_api.py          # Полный API (порт 8005)
```

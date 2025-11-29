"""
Пример 7: Ollama (Локальные модели)
Запуск LLM локально без API ключей и интернета
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Создаем экземпляр локальной модели
# Убедитесь что Ollama запущен: ollama serve
# И модель скачана: ollama pull gemma3:1b
llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.7,
)

# Простой вызов
print("=== Простой вызов ===")
response = llm.invoke("Привет! Кто ты?")
print(response.content)

print("\n=== С шаблоном промпта ===")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты Python эксперт. Отвечай кратко на русском."),
    ("human", "{question}")
])

chain = prompt | llm

response = chain.invoke({"question": "Что такое list comprehension?"})
print(response.content)

print("\n=== Streaming ===")

for chunk in chain.stream({"question": "Напиши простой пример функции"}):
    print(chunk.content, end="", flush=True)
print()


"""
Пример 6: Streaming (Потоковый вывод)
Получаем ответ по частям, как в ChatGPT
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты рассказчик историй."),
    ("human", "{request}")
])

chain = prompt | llm

print("Генерация истории (streaming):\n")

# Используем stream вместо invoke
for chunk in chain.stream({"request": "Расскажи очень короткую историю про робота"}):
    # Каждый chunk содержит часть ответа
    print(chunk.content, end="", flush=True)

print("\n")  # Новая строка в конце


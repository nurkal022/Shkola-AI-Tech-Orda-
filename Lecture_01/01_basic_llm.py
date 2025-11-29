"""
Пример 1: Базовый вызов LLM
Самый простой способ использования LangChain - прямой вызов модели
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Создаем экземпляр модели
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Простой вызов
response = llm.invoke("Привет! Расскажи короткий факт о Python")
print("Ответ модели:")
print(response.content)


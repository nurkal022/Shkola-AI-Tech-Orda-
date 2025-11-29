"""
Пример 2: Prompt Templates
Шаблоны промптов позволяют создавать переиспользуемые промпты с переменными
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Создаем шаблон промпта
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты эксперт по {topic}. Отвечай кратко и по делу."),
    ("human", "{question}")
])

# Форматируем промпт с конкретными значениями
messages = prompt.invoke({
    "topic": "программированию на Python",
    "question": "Что такое декоратор?"
})

# Отправляем в модель
response = llm.invoke(messages)
print("Вопрос: Что такое декоратор?")
print("\nОтвет:")
print(response.content)


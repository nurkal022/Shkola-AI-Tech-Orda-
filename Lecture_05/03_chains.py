"""
Пример 3: Chains (Цепочки)
Цепочки позволяют объединять несколько компонентов в пайплайн
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Шаблон для генерации шутки
joke_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты комик. Придумывай короткие и смешные шутки."),
    ("human", "Расскажи шутку про {topic}")
])

# Создаем цепочку с помощью LCEL (LangChain Expression Language)
# prompt -> llm -> parser
chain = joke_prompt | llm | StrOutputParser()

# Запускаем цепочку
result = chain.invoke({"topic": "программистов"})
print("Шутка про программистов:")
print(result)

print("\n" + "="*50 + "\n")

# Еще одна тема
result2 = chain.invoke({"topic": "кофе"})
print("Шутка про кофе:")
print(result2)


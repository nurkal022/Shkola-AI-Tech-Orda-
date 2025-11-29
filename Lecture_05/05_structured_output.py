"""
Пример 5: Output Parsers (Парсеры вывода)
Встроенные в LangChain парсеры для обработки ответов модели
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
    
)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ===========================================
# 1. StrOutputParser - просто текст
# ===========================================
print("=== StrOutputParser ===")

str_parser = StrOutputParser()

chain = llm | str_parser  # Возвращает строку вместо AIMessage

result = chain.invoke("Скажи 'Привет' одним словом")
print(f"Тип: {type(result)}")  # <class 'str'>
print(f"Результат: {result}\n")


# ===========================================
# 2. CommaSeparatedListOutputParser - список
# ===========================================
print("=== CommaSeparatedListOutputParser ===")

list_parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_template(
    "Назови 5 языков программирования.\n{format_instructions}"
)

# Парсер сам генерирует инструкции для модели
formatted_prompt = prompt.invoke({
    "format_instructions": list_parser.get_format_instructions()
})
print(f"Инструкции для модели: {list_parser.get_format_instructions()}")

chain = prompt | llm | list_parser

result = chain.invoke({
    "format_instructions": list_parser.get_format_instructions()
})
print(f"Тип: {type(result)}")  # <class 'list'>
print(f"Результат: {result}\n")


# ===========================================
# 3. JsonOutputParser - JSON словарь
# ===========================================
print("=== JsonOutputParser ===")

json_parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    """Дай информацию о фильме в JSON формате с полями: title, year, director.
    
{format_instructions}

Фильм: {movie}"""
)

chain = prompt | llm | json_parser

result = chain.invoke({
    "format_instructions": json_parser.get_format_instructions(),
    "movie": "Интерстеллар"
})
print(f"Тип: {type(result)}")  # <class 'dict'>
print(f"Результат: {result}")
print(f"Доступ к полю: {result.get('title')}")

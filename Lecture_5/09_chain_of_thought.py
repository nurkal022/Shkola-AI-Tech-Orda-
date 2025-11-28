"""
Пример 9: Chain of Thought (Цепочка рассуждений)
Один запрос → несколько этапов обработки
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# Этап 1: Придумать тему
step1 = ChatPromptTemplate.from_template(
    "Придумай одну интересную тему для статьи про {subject}. Только тему, без объяснений."
) | llm | parser

# Этап 2: Написать план
step2 = ChatPromptTemplate.from_template(
    "Напиши короткий план статьи (3 пункта) на тему: {topic}"
) | llm | parser

# Этап 3: Написать введение
step3 = ChatPromptTemplate.from_template(
    "Напиши короткое введение (2-3 предложения) для статьи с планом:\n{plan}"
) | llm | parser


# Запускаем цепочку
print("=== Chain of Thought ===\n")

subject = "Python"

# Шаг 1
topic = step1.invoke({"subject": subject})
print(f"1️⃣ Тема: {topic}\n")

# Шаг 2
plan = step2.invoke({"topic": topic})
print(f"2️⃣ План:\n{plan}\n")

# Шаг 3
intro = step3.invoke({"plan": plan})
print(f"3️⃣ Введение:\n{intro}")

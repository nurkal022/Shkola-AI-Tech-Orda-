"""
Пример 4: Memory (Память)
Память позволяет модели помнить предыдущие сообщения в диалоге
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Промпт с местом для истории сообщений
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты дружелюбный ассистент. Отвечай кратко."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# Храним историю сообщений
history = []

def chat(user_input):
    """Функция для общения с сохранением истории"""
    response = chain.invoke({
        "history": history,
        "input": user_input
    })
    
    # Добавляем сообщения в историю
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))
    
    return response.content

# Демонстрация диалога с памятью
print("Диалог с памятью:\n")

print("User: Меня зовут Алмас")
print(f"AI: {chat('Меня зовут Алмас')}\n")

print("User: Какой язык программирования лучше учить первым?")
print(f"AI: {chat('Какой язык программирования лучше учить первым?')}\n")

print("User: Как меня зовут?")
print(f"AI: {chat('Как меня зовут?')}\n")  # Модель должна помнить имя!


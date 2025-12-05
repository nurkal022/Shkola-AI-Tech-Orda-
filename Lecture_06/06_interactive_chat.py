"""
Шаг 6: Интерактивный чат с RAG системой
=======================================
Полноценный чат-бот для общения по книгам Гарри Поттера
с историей диалога и стримингом ответов.
"""

import os
from pathlib import Path
from typing import List, Optional, Generator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()


class HarryPotterChatBot:
    """
    Интерактивный чат-бот по книгам Гарри Поттера.
    
    Особенности:
    - Память диалога (помнит предыдущие сообщения)
    - RAG для ответов на основе книг
    - Стриминг ответов
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        vectorstore_path: str = "./faiss_harry_potter",
        memory_window: int = 10
    ):
        print("🧙‍♂️ Инициализация Harry Potter ChatBot...")
        
        # LLM с поддержкой стриминга
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            streaming=True
        )
        
        # Embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Vector store
        if Path(vectorstore_path).exists():
            self.vectorstore = FAISS.load_local(
                vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"   ✅ База знаний загружена")
        else:
            raise FileNotFoundError(
                f"Индекс не найден: {vectorstore_path}\n"
                "Сначала запустите 04_rag_pipeline.py для создания индекса"
            )
        
        # Память диалога
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Системный промпт
        self.system_prompt = """Ты - экспертный помощник по вселенной Гарри Поттера. 
Ты обладаешь глубокими знаниями всех 7 книг серии.

Твоя задача:
1. Отвечать на вопросы о персонажах, событиях, магии и мире Гарри Поттера
2. Использовать контекст из книг для точных ответов
3. Быть дружелюбным и увлекательным собеседником
4. Если не уверен - честно признавать это

Всегда отвечай на русском языке."""

        print("   ✅ ChatBot готов к работе!\n")
    
    def retrieve_context(self, query: str, k: int = 4) -> str:
        """Получение релевантного контекста из книг"""
        docs = self.vectorstore.similarity_search(query, k=k)
        
        context_parts = []
        for doc in docs:
            title = doc.metadata.get('title', 'Неизвестно')
            context_parts.append(f"[{title}]\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def chat(self, user_input: str, stream: bool = True) -> str:
        """
        Основной метод чата.
        
        Args:
            user_input: Сообщение пользователя
            stream: Включить стриминг ответа
        
        Returns:
            Ответ бота
        """
        # Получаем контекст
        context = self.retrieve_context(user_input)
        
        # Получаем историю
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        # Формируем промпт
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("system", "Контекст из книг:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Создаем сообщения
        messages = prompt.format_messages(
            context=context,
            chat_history=chat_history,
            input=user_input
        )
        
        # Генерируем ответ
        if stream:
            response = ""
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    response += chunk.content
            print()  # Новая строка после стриминга
        else:
            result = self.llm.invoke(messages)
            response = result.content
        
        # Сохраняем в память
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        return response
    
    def clear_memory(self):
        """Очистка истории диалога"""
        self.memory.clear()
        print("🗑️ История диалога очищена")
    
    def get_history(self) -> List[dict]:
        """Получение истории диалога"""
        messages = self.memory.load_memory_variables({})["chat_history"]
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history


def interactive_chat():
    """Запуск интерактивного чата"""
    print("="*60)
    print("🧙‍♂️ HARRY POTTER CHATBOT")
    print("="*60)
    print("Добро пожаловать в чат по вселенной Гарри Поттера!")
    print("Команды:")
    print("  /clear  - очистить историю")
    print("  /history - показать историю")
    print("  /quit   - выход")
    print("="*60 + "\n")
    
    try:
        bot = HarryPotterChatBot()
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        return
    
    while True:
        try:
            user_input = input("👤 Вы: ").strip()
            
            if not user_input:
                continue
            
            # Команды
            if user_input.lower() == "/quit":
                print("👋 До свидания!")
                break
            elif user_input.lower() == "/clear":
                bot.clear_memory()
                continue
            elif user_input.lower() == "/history":
                history = bot.get_history()
                print("\n📜 История диалога:")
                for msg in history:
                    role = "👤" if msg["role"] == "user" else "🤖"
                    print(f"{role}: {msg['content'][:100]}...")
                print()
                continue
            
            # Обычное сообщение
            print("🤖 Бот: ", end="")
            bot.chat(user_input, stream=True)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break


def demo_chat():
    """Демонстрация чата с предопределенными вопросами"""
    print("="*60)
    print("🎬 ДЕМОНСТРАЦИЯ ЧАТА")
    print("="*60)
    
    try:
        bot = HarryPotterChatBot()
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        return
    
    # Демо-диалог
    questions = [
        "Привет! Расскажи кратко о Гарри Поттере",
        "А кто его лучшие друзья?",
        "Расскажи подробнее о Гермионе",
        "В какой факультет они попали?"
    ]
    
    for question in questions:
        print(f"\n👤 Вы: {question}")
        print("🤖 Бот: ", end="")
        bot.chat(question, stream=True)
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_chat()
    else:
        interactive_chat()


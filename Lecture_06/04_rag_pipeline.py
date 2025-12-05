"""
Шаг 4: RAG Pipeline - Retrieval-Augmented Generation
====================================================
Полный pipeline для вопросно-ответной системы на основе документов.
Используем GPT-4.1-mini для генерации ответов.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()


class RAGPipeline:
    """
    RAG Pipeline для вопросно-ответной системы.
    
    Архитектура:
    1. Query → Embedding → Vector Search
    2. Retrieved Documents + Query → LLM
    3. LLM → Answer
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.3,
        vectorstore_path: Optional[str] = None
    ):
        """
        Инициализация RAG pipeline.
        
        Args:
            model_name: Модель для генерации (gpt-4.1-mini, gpt-4o, gpt-4o-mini)
            embedding_model: Модель для embeddings
            temperature: Температура генерации (0 = детерминированно, 1 = креативно)
            vectorstore_path: Путь к сохраненному векторному хранилищу
        """
        print(f"🚀 Инициализация RAG Pipeline...")
        
        # LLM для генерации
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        print(f"   ✅ LLM: {model_name}")
        
        # Embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        print(f"   ✅ Embeddings: {embedding_model}")
        
        # Vector store
        self.vectorstore = None
        if vectorstore_path and Path(vectorstore_path).exists():
            self.load_vectorstore(vectorstore_path)
        
        # Промпт по умолчанию
        self.prompt = self._create_default_prompt()
        
    def _create_default_prompt(self) -> ChatPromptTemplate:
        """Создание промпта для RAG"""
        template = """Ты - эксперт по книгам о Гарри Поттере. Отвечай на вопросы, 
используя ТОЛЬКО информацию из предоставленного контекста.

Правила:
1. Отвечай на русском языке
2. Если информации нет в контексте - честно скажи об этом
3. Цитируй релевантные части текста когда уместно
4. Будь точным и информативным

Контекст из книг:
{context}

Вопрос: {question}

Ответ:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def load_vectorstore(self, path: str) -> None:
        """Загрузка векторного хранилища"""
        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"   ✅ Vector store загружен: {path}")
    
    def create_vectorstore(
        self, 
        documents: List[Document],
        save_path: Optional[str] = None
    ) -> None:
        """Создание векторного хранилища из документов"""
        print(f"🔄 Создание vector store из {len(documents)} документов...")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        if save_path:
            self.vectorstore.save_local(save_path)
            print(f"💾 Vector store сохранен: {save_path}")
        
        print(f"✅ Vector store создан!")
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Поиск релевантных документов.
        
        Args:
            query: Поисковый запрос
            k: Количество документов
        """
        if not self.vectorstore:
            raise ValueError("Vector store не инициализирован!")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def format_docs(self, docs: List[Document]) -> str:
        """Форматирование документов для контекста"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('title', 'Неизвестно')
            formatted.append(f"[Источник {i}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)
    
    def query(
        self, 
        question: str, 
        k: int = 4,
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Основной метод для получения ответа.
        
        Args:
            question: Вопрос пользователя
            k: Количество документов для контекста
            return_sources: Возвращать ли источники
        
        Returns:
            Dict с ответом и опционально источниками
        """
        # 1. Поиск релевантных документов
        docs = self.retrieve(question, k=k)
        
        # 2. Форматирование контекста
        context = self.format_docs(docs)
        
        # 3. Генерация ответа
        chain = self.prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        result = {"answer": answer}
        
        if return_sources:
            result["sources"] = [
                {
                    "title": doc.metadata.get('title'),
                    "book_number": doc.metadata.get('book_number'),
                    "content_preview": doc.page_content[:200] + "..."
                }
                for doc in docs
            ]
        
        return result
    
    def query_with_chain(self, question: str, k: int = 4) -> str:
        """
        Альтернативный метод с использованием LCEL chain.
        Более элегантный, но менее гибкий.
        """
        if not self.vectorstore:
            raise ValueError("Vector store не инициализирован!")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke(question)


def create_documents_from_books(data_dir: str = "data") -> List[Document]:
    """Создание документов из книг"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    documents = []
    data_path = Path(data_dir)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    )
    
    for file_path in sorted(data_path.glob("*.txt")):
        filename = file_path.stem
        
        # Извлекаем метаданные
        book_num = 0
        if '#' in filename:
            try:
                book_num = int(filename.split('#')[1].split(']')[0])
            except:
                pass
        
        parts = filename.split(']_')
        title = parts[1].replace('_', ' ') if len(parts) > 1 else filename
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "book_number": book_num,
                    "title": title,
                    "chunk_id": i
                }
            ))
        
        print(f"📖 {title}: {len(chunks)} чанков")
    
    return documents


def demo_rag():
    """Демонстрация RAG pipeline"""
    print("="*60)
    print("🤖 ДЕМОНСТРАЦИЯ RAG PIPELINE")
    print("="*60)
    
    # Проверяем существование индекса
    index_path = "./faiss_harry_potter"
    
    if Path(index_path).exists():
        # Загружаем существующий индекс
        print("\n📂 Загрузка существующего индекса...")
        rag = RAGPipeline(
            model_name="gpt-4.1-mini",
            vectorstore_path=index_path
        )
    else:
        # Создаем новый индекс
        print("\n🔨 Создание нового индекса...")
        rag = RAGPipeline(model_name="gpt-4.1-mini")
        
        documents = create_documents_from_books()
        rag.create_vectorstore(documents, save_path=index_path)
    
    # Тестовые вопросы
    questions = [
        "Кто такой Волдеморт и почему его боятся?",
        "Как Гарри попал в Хогвартс?",
        "Расскажи про крестражи",
        "Кто такой Северус Снейп?",
    ]
    
    print("\n" + "="*60)
    print("💬 ДИАЛОГ С RAG СИСТЕМОЙ")
    print("="*60)
    
    for question in questions:
        print(f"\n❓ Вопрос: {question}")
        print("-"*50)
        
        result = rag.query(question, k=4, return_sources=True)
        
        print(f"💡 Ответ:\n{result['answer']}")
        
        print(f"\n📚 Источники:")
        for source in result['sources']:
            print(f"   - {source['title']} (книга #{source['book_number']})")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    demo_rag()


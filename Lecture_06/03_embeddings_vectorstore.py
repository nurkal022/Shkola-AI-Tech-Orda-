"""
Шаг 3: Создание Embeddings и Vector Store
=========================================
Преобразование текста в векторы и хранение для семантического поиска.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document

# Загружаем переменные окружения
load_dotenv()


class EmbeddingManager:
    """Менеджер для работы с embeddings"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Инициализация embedding модели.
        
        Args:
            model: Модель для embeddings
                - text-embedding-3-small (дешевле, быстрее)
                - text-embedding-3-large (точнее, дороже)
                - text-embedding-ada-002 (legacy)
        """
        self.embeddings = OpenAIEmbeddings(model=model)
        self.model_name = model
        print(f"✅ Embedding модель инициализирована: {model}")
    
    def embed_query(self, text: str) -> List[float]:
        """Получить embedding для одного текста (запроса)"""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получить embeddings для списка документов"""
        return self.embeddings.embed_documents(texts)


class VectorStoreManager:
    """Менеджер для работы с векторными хранилищами"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.vectorstore = None
    
    def create_faiss_store(
        self, 
        documents: List[Document],
        save_path: Optional[str] = None
    ) -> FAISS:
        """
        Создание FAISS векторного хранилища.
        
        FAISS - быстрый и эффективный для локального использования.
        Плюсы: быстрый поиск, работает локально
        Минусы: не персистентный по умолчанию
        """
        print(f"🔄 Создание FAISS индекса из {len(documents)} документов...")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_manager.embeddings
        )
        
        if save_path:
            self.save_faiss(save_path)
        
        print(f"✅ FAISS индекс создан!")
        return self.vectorstore
    
    def save_faiss(self, path: str) -> None:
        """Сохранение FAISS индекса на диск"""
        if self.vectorstore and isinstance(self.vectorstore, FAISS):
            self.vectorstore.save_local(path)
            print(f"💾 FAISS индекс сохранен: {path}")
    
    def load_faiss(self, path: str) -> FAISS:
        """Загрузка FAISS индекса с диска"""
        self.vectorstore = FAISS.load_local(
            path, 
            self.embedding_manager.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"📂 FAISS индекс загружен: {path}")
        return self.vectorstore
    
    def create_chroma_store(
        self, 
        documents: List[Document],
        persist_directory: str = "./chroma_db"
    ) -> Chroma:
        """
        Создание Chroma векторного хранилища.
        
        Chroma - персистентное хранилище с метаданными.
        Плюсы: автоматическое сохранение, фильтрация по метаданным
        Минусы: медленнее FAISS
        """
        print(f"🔄 Создание Chroma индекса из {len(documents)} документов...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_manager.embeddings,
            persist_directory=persist_directory
        )
        
        print(f"✅ Chroma индекс создан и сохранен: {persist_directory}")
        return self.vectorstore
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4
    ) -> List[Document]:
        """
        Поиск похожих документов.
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore не инициализирован!")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4
    ) -> List[tuple]:
        """
        Поиск с оценкой релевантности.
        Возвращает пары (документ, score).
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore не инициализирован!")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)


def create_documents_from_books(data_dir: str = "data") -> List[Document]:
    """
    Создание документов из всех книг с метаданными.
    """
    from text_chunking import create_recursive_splitter, ChunkConfig
    
    documents = []
    data_path = Path(data_dir)
    
    config = ChunkConfig(chunk_size=1000, chunk_overlap=200)
    splitter = create_recursive_splitter(config)
    
    for file_path in sorted(data_path.glob("*.txt")):
        # Извлекаем метаданные
        filename = file_path.stem
        book_num = 0
        if '#' in filename:
            try:
                book_num = int(filename.split('#')[1].split(']')[0])
            except:
                pass
        
        parts = filename.split(']_')
        title = parts[1].replace('_', ' ') if len(parts) > 1 else filename
        
        # Читаем и разбиваем текст
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = splitter.split_text(text)
        
        # Создаем документы с метаданными
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "book_number": book_num,
                    "title": title,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        print(f"📖 {title}: {len(chunks)} чанков")
    
    print(f"\n✅ Всего создано {len(documents)} документов")
    return documents


def demo_vectorstore():
    """Демонстрация работы с векторным хранилищем"""
    print("="*60)
    print("🗄️ ДЕМОНСТРАЦИЯ VECTOR STORE")
    print("="*60)
    
    # Создаем embedding manager
    embed_manager = EmbeddingManager(model="text-embedding-3-small")
    
    # Создаем документы
    documents = create_documents_from_books()
    
    # Создаем vector store manager
    vs_manager = VectorStoreManager(embed_manager)
    
    # Создаем FAISS индекс
    vs_manager.create_faiss_store(documents, save_path="./faiss_index")
    
    # Тестовые запросы
    test_queries = [
        "Кто такой Волдеморт?",
        "Расскажи про Хогвартс",
        "Как Гарри узнал что он волшебник?",
        "Кто друзья Гарри Поттера?"
    ]
    
    print("\n" + "="*60)
    print("🔍 ТЕСТИРОВАНИЕ ПОИСКА")
    print("="*60)
    
    for query in test_queries:
        print(f"\n❓ Запрос: {query}")
        print("-"*40)
        
        results = vs_manager.similarity_search_with_score(query, k=2)
        
        for doc, score in results:
            print(f"📄 [{doc.metadata['title']}] (score: {score:.4f})")
            print(f"   {doc.page_content[:200]}...")
            print()


if __name__ == "__main__":
    demo_vectorstore()


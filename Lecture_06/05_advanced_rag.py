"""
Шаг 5: Продвинутые техники RAG
==============================
Улучшение качества RAG: переранжирование, гибридный поиск, 
self-query и другие продвинутые техники.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()


class AdvancedRAG:
    """
    Продвинутый RAG с различными стратегиями улучшения качества.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        vectorstore_path: Optional[str] = None
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        self.vectorstore = None
        if vectorstore_path and Path(vectorstore_path).exists():
            self.vectorstore = FAISS.load_local(
                vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Vector store загружен: {vectorstore_path}")
    
    # ========================================
    # Стратегия 1: Multi-Query Retrieval
    # ========================================
    def multi_query_retrieve(
        self, 
        question: str, 
        k: int = 4,
        num_queries: int = 3
    ) -> List[Document]:
        """
        Генерирует несколько вариантов запроса для лучшего покрытия.
        
        Идея: один вопрос можно задать по-разному, каждый вариант
        может найти разные релевантные документы.
        """
        # Генерируем альтернативные запросы
        query_prompt = ChatPromptTemplate.from_template(
            """Ты помощник для генерации поисковых запросов.
            Для данного вопроса пользователя, сгенерируй {num_queries} альтернативных 
            формулировок того же вопроса на русском языке.
            Каждый запрос на новой строке, без нумерации.
            
            Оригинальный вопрос: {question}
            
            Альтернативные запросы:"""
        )
        
        chain = query_prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "question": question,
            "num_queries": num_queries
        })
        
        # Парсим запросы
        queries = [question]  # Включаем оригинал
        queries.extend([q.strip() for q in response.strip().split('\n') if q.strip()])
        
        print(f"🔍 Сгенерировано {len(queries)} запросов:")
        for q in queries:
            print(f"   - {q}")
        
        # Собираем уникальные документы
        all_docs = []
        seen_contents = set()
        
        for query in queries:
            docs = self.vectorstore.similarity_search(query, k=k)
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
        
        return all_docs[:k * 2]  # Возвращаем больше документов
    
    # ========================================
    # Стратегия 2: Contextual Compression
    # ========================================
    def compressed_retrieve(
        self, 
        question: str, 
        k: int = 4
    ) -> List[Document]:
        """
        Сжимает найденные документы, оставляя только релевантные части.
        
        Идея: из большого чанка извлечь только ту часть, 
        которая отвечает на вопрос.
        """
        # Создаем компрессор
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        # Создаем compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": k})
        )
        
        docs = compression_retriever.invoke(question)
        
        print(f"📦 Сжато до {len(docs)} релевантных фрагментов")
        return docs
    
    # ========================================
    # Стратегия 3: Reranking (Переранжирование)
    # ========================================
    def rerank_documents(
        self, 
        question: str,
        documents: List[Document],
        top_k: int = 4
    ) -> List[Document]:
        """
        Переранжирование документов с помощью LLM.
        
        Идея: векторный поиск хорош, но LLM может лучше оценить
        релевантность документа к конкретному вопросу.
        """
        if not documents:
            return []
        
        rerank_prompt = ChatPromptTemplate.from_template(
            """Оцени релевантность документа к вопросу по шкале от 0 до 10.
            Верни ТОЛЬКО число.
            
            Вопрос: {question}
            
            Документ: {document}
            
            Оценка (0-10):"""
        )
        
        chain = rerank_prompt | self.llm | StrOutputParser()
        
        scored_docs = []
        for doc in documents:
            try:
                score_str = chain.invoke({
                    "question": question,
                    "document": doc.page_content[:1000]  # Ограничиваем длину
                })
                score = float(score_str.strip())
            except:
                score = 5.0  # Дефолтная оценка
            
            scored_docs.append((doc, score))
        
        # Сортируем по убыванию оценки
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📊 Переранжировано {len(documents)} документов")
        for doc, score in scored_docs[:top_k]:
            print(f"   Score {score:.1f}: {doc.metadata.get('title', 'N/A')}")
        
        return [doc for doc, _ in scored_docs[:top_k]]
    
    # ========================================
    # Стратегия 4: Parent Document Retrieval
    # ========================================
    def retrieve_with_context(
        self, 
        question: str, 
        k: int = 4,
        context_size: int = 1
    ) -> List[Document]:
        """
        Получение документов с окружающим контекстом.
        
        Идея: маленькие чанки хороши для поиска, но для ответа
        может понадобиться больше контекста.
        """
        # Находим релевантные чанки
        docs = self.vectorstore.similarity_search(question, k=k)
        
        expanded_docs = []
        for doc in docs:
            chunk_id = doc.metadata.get('chunk_id', 0)
            source = doc.metadata.get('source', '')
            
            # Пытаемся найти соседние чанки
            expanded_content = doc.page_content
            
            # Ищем предыдущий чанк
            if chunk_id > 0:
                prev_docs = self.vectorstore.similarity_search(
                    f"chunk_id:{chunk_id - 1} source:{source}",
                    k=1
                )
                if prev_docs:
                    expanded_content = prev_docs[0].page_content + "\n\n" + expanded_content
            
            # Ищем следующий чанк
            next_docs = self.vectorstore.similarity_search(
                f"chunk_id:{chunk_id + 1} source:{source}",
                k=1
            )
            if next_docs:
                expanded_content = expanded_content + "\n\n" + next_docs[0].page_content
            
            expanded_doc = Document(
                page_content=expanded_content,
                metadata=doc.metadata
            )
            expanded_docs.append(expanded_doc)
        
        return expanded_docs
    
    # ========================================
    # Полный Advanced RAG Pipeline
    # ========================================
    def advanced_query(
        self,
        question: str,
        k: int = 4,
        use_multi_query: bool = True,
        use_reranking: bool = True,
        use_compression: bool = False
    ) -> Dict[str, Any]:
        """
        Полный продвинутый RAG pipeline.
        """
        print(f"\n{'='*50}")
        print(f"🚀 Advanced RAG Query")
        print(f"{'='*50}")
        print(f"❓ Вопрос: {question}\n")
        
        # Шаг 1: Retrieval
        if use_multi_query:
            print("📍 Шаг 1: Multi-Query Retrieval")
            docs = self.multi_query_retrieve(question, k=k)
        else:
            print("📍 Шаг 1: Standard Retrieval")
            docs = self.vectorstore.similarity_search(question, k=k*2)
        
        # Шаг 2: Reranking
        if use_reranking:
            print("\n📍 Шаг 2: Reranking")
            docs = self.rerank_documents(question, docs, top_k=k)
        
        # Шаг 3: Compression (опционально)
        if use_compression:
            print("\n📍 Шаг 3: Compression")
            docs = self.compressed_retrieve(question, k=k)
        
        # Шаг 4: Generation
        print("\n📍 Шаг 4: Generation")
        
        context = "\n\n---\n\n".join([
            f"[{doc.metadata.get('title', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        answer_prompt = ChatPromptTemplate.from_template(
            """Ты эксперт по книгам о Гарри Поттере. Ответь на вопрос,
            используя предоставленный контекст.
            
            Контекст:
            {context}
            
            Вопрос: {question}
            
            Дай подробный и информативный ответ на русском языке:"""
        )
        
        chain = answer_prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        return {
            "answer": answer,
            "num_docs_retrieved": len(docs),
            "sources": [doc.metadata.get('title') for doc in docs]
        }


def demo_advanced_rag():
    """Демонстрация продвинутого RAG"""
    print("="*60)
    print("🔬 ДЕМОНСТРАЦИЯ ADVANCED RAG")
    print("="*60)
    
    index_path = "./faiss_harry_potter"
    
    if not Path(index_path).exists():
        print("❌ Индекс не найден. Сначала запустите 04_rag_pipeline.py")
        return
    
    rag = AdvancedRAG(
        model_name="gpt-4.1-mini",
        vectorstore_path=index_path
    )
    
    # Сложный вопрос для тестирования
    question = "Какова связь между Гарри Поттером и Волдемортом? Почему именно Гарри был избран?"
    
    # Стандартный RAG
    print("\n" + "="*50)
    print("📊 СРАВНЕНИЕ СТРАТЕГИЙ")
    print("="*50)
    
    # Advanced RAG
    result = rag.advanced_query(
        question,
        k=4,
        use_multi_query=True,
        use_reranking=True
    )
    
    print("\n" + "="*50)
    print("💡 ФИНАЛЬНЫЙ ОТВЕТ:")
    print("="*50)
    print(result['answer'])
    print(f"\n📚 Использовано источников: {result['num_docs_retrieved']}")
    print(f"📖 Книги: {', '.join(set(result['sources']))}")


if __name__ == "__main__":
    demo_advanced_rag()


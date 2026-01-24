from supabase import create_client, Client
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# print(supabase.table("documents").select("*").execute())


openai_client = OpenAI()

# response = openai_client.embeddings.create(
#     model="text-embedding-3-small",
#     input="Hello, world!"
# )
# print(response.data[0].embedding)



# with open("data/конституция.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100,
#     length_function=len,
#     separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
# )

# chunks = splitter.split_text(text)
# print("Количество чанков:", len(chunks))

# # print(chunks[0])
# # print("-"*100)
# print(chunks[1])
# print("-"*100)
# print(chunks[2])
# print("-"*100)
# print(chunks[3])
# print("-"*100)
# print(chunks[4])


def get_embeddings(texts: list[str], batch_size: int = 100):
    """Создать эмбеддинги батчами"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"   Обработано: {min(i+batch_size, len(texts))}/{len(texts)}")
    return all_embeddings


# embeddings = get_embeddings(chunks, batch_size=100)
# print("Количество эмбеддингов:", len(embeddings))
# print(embeddings[0])
# print("-"*100)



# batch_size = 20
# loaded_count = 0

# for i in range(0, len(chunks), batch_size):
#     batch_chunks = chunks[i:i+batch_size]
#     batch_embeddings = embeddings[i:i+batch_size]

#     records = []
#     for chunk, emb in zip(batch_chunks, batch_embeddings):
#         records.append({
#             "content": chunk,
#             "source": "конституция",
#             "chunk_id": i + len(records),
#             "embedding": emb
#         })
#     supabase.table("documents").insert(records).execute()
#     loaded_count += len(batch_chunks)



# print(f"Загружено {loaded_count} документов в Supabase!")




def search_documents(query: str, top_k: int = 3):
    query_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    
    try:
        result = supabase.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_count': top_k,
        }).execute()
        
 
        return result.data
    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        raise e



def rag_pipeline(query: str, top_k: int = 3):
    """RAG пайплайн"""
    context = search_documents(query, top_k)

    context_text = "\n\n---\n\n".join([
    f"[{doc['book']}]\n{doc['content']}"
    for doc in context
])

    prompt = f"""Ответь на вопрос, используя ТОЛЬКО предоставленный контекст из Конституции Республики Казахстан.

        КОНТЕКСТ:
        {context_text}

        ВОПРОС: {query}

        ОТВЕТ (кратко, 2-3 предложения):"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты - специалист по Конституции Республики Казахстан."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


print(rag_pipeline("Какие права и свободы гарантирует Конституция?", 3))



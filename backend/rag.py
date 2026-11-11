import os
import chromadb
import ollama

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
CHAT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

ollama_client = ollama.Client(host=OLLAMA_URL)

def _embed(text: str) -> list[float]:
    resp = ollama_client.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]

def index_chunks(doc_id: str, chunks: list[str]):
    if not chunks:
        return 0
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    embeddings = [_embed(c) for c in chunks]
    # ВАЖНО: сохраняем doc_id в метаданные чанка
    metadatas = [{"doc_id": doc_id} for _ in chunks]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)

def search(question: str, doc_id: str = None, n_results: int = 3) -> list[str]:
    q_emb = _embed(question)
    
    # Фильтр по конкретному документу, если он передан
    where_filter = {"doc_id": doc_id} if doc_id else None
    
    results = collection.query(
        query_embeddings=[q_emb], 
        n_results=n_results,
        where=where_filter
    )
    docs = results.get("documents") or []
    return docs[0] if docs else []

def answer(question: str, context_chunks: list[str]) -> str:
    if context_chunks:
        context = "\n\n---\n\n".join(context_chunks)
        prompt = (
            "Ты — ассистент по работе с документами. Ответь на вопрос пользователя, "
            "используя ТОЛЬКО приведённый контекст. Если ответа в контексте нет — скажи, что не нашёл. "
            "Не выдумывай факты, опирайся только на текст.\n\n"
            f"Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:"
        )
    else:
        prompt = question
    resp = ollama_client.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()
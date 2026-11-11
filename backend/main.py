import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from pipeline import run_pipeline
from storage import upload_file
from database import init_db, create_document, get_document, list_documents
import rag

app = FastAPI(title="Document AI MVP")

@app.on_event("startup")
def startup():
    init_db()

class QuestionRequest(BaseModel):
    question: str
    doc_id: str | None = None  # <-- Опциональный ID документа для точного поиска

class IngestRequest(BaseModel):
    doc_id: str

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "Пустой файл")
    doc_id = str(uuid.uuid4())
    create_document(doc_id, file.filename, status="processing")
    try:
        upload_file(f"{doc_id}_{file.filename}", content)
        text, chunks = run_pipeline(content)
        n = rag.index_chunks(doc_id, chunks)
        create_document(doc_id, file.filename, status="processed", chunks_count=n)
    except Exception as e:
        create_document(doc_id, file.filename, status=f"error: {e}", chunks_count=0)
    doc = get_document(doc_id)
    return {"doc_id": doc_id, "status": doc["status"], "chunks": doc["chunks_count"]}

@app.post("/ingest")
async def ingest(payload: IngestRequest):
    doc = get_document(payload.doc_id)
    if not doc:
        raise HTTPException(404, "Документ не найден")
    return {"doc_id": doc["id"], "status": doc["status"]}

@app.post("/ask")
async def ask(payload: QuestionRequest):
    # Ищем фрагменты (только по doc_id, если он передан)
    chunks = rag.search(payload.question, doc_id=payload.doc_id)
    answer = rag.answer(payload.question, chunks)
    return {"answer": answer, "sources": chunks}

@app.get("/documents")
def docs():
    return list_documents()

@app.get("/health")
def health():
    return {"status": "ok"}
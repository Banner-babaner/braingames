import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from pydantic import BaseModel

from pipeline import run_pipeline
from storage import upload_file
from database import init_db, create_document, get_document, list_documents
from access import check_subscription, get_owned_document
import rag

app = FastAPI(title="Document AI MVP")

@app.on_event("startup")
def startup():
    init_db()

class QuestionRequest(BaseModel):
    question: str
    doc_id: str | None = None

class IngestRequest(BaseModel):
    doc_id: str

# --- временная аутентификация: X-User-Id ---
# На MVP хватает. Позже замените на JWT/OAuth2.
def current_user(x_user_id: str = Header(default="demo-user")) -> str:
    return x_user_id

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    user_id: str = Depends(current_user),
):
    content = await file.read()
    if not content:
        raise HTTPException(400, "Пустой файл")

    # 1. Подписка (403 — нет доступа, 402 — лимит исчерпан)
    check_subscription(user_id, len(content))

    # 2. Canny только для фото с телефона. Для планшетных сканов — выкл.
    use_canny = file.content_type in ("image/jpeg", "image/jpg")

    doc_id = str(uuid.uuid4())
    create_document(doc_id, file.filename, status="processing",
                    owner_id=user_id, file_size=len(content))
    try:
        # 3. Изоляция в хранилище: префикс tenants/<user_id>/
        upload_file(f"tenants/{user_id}/{doc_id}_{file.filename}", content)
        text, chunks = run_pipeline(content, use_canny=use_canny)
        n = rag.index_chunks(doc_id, chunks, owner_id=user_id)
        create_document(doc_id, file.filename, status="processed",
                        chunks_count=n, owner_id=user_id, file_size=len(content))
    except Exception as e:
        create_document(doc_id, file.filename, status=f"error: {e}",
                        chunks_count=0, owner_id=user_id, file_size=len(content))
    doc = get_document(doc_id)
    return {"doc_id": doc_id, "status": doc["status"], "chunks": doc["chunks_count"]}

@app.post("/ingest")
async def ingest(payload: IngestRequest, user_id: str = Depends(current_user)):
    doc = get_owned_document(user_id, payload.doc_id)  # 404 на чужой
    return {"doc_id": doc["id"], "status": doc["status"]}

@app.post("/ask")
async def ask(payload: QuestionRequest, user_id: str = Depends(current_user)):
    if payload.doc_id:
        get_owned_document(user_id, payload.doc_id)
        chunks = rag.search(payload.question, doc_id=payload.doc_id, owner_id=user_id)
    else:
        chunks = rag.search(payload.question, owner_id=user_id)
    answer = rag.answer(payload.question, chunks)
    return {"answer": answer, "sources": chunks}

@app.get("/documents")
def docs(user_id: str = Depends(current_user)):
    return list_documents(owner_id=user_id)

@app.get("/health")
def health():
    return {"status": "ok"}
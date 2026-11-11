import os
import sqlite3
from datetime import datetime

DB_PATH = os.getenv("METADATA_DB_PATH", "/app/data/metadata.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                chunks_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

def create_document(doc_id: str, filename: str, status: str = "processed", chunks_count: int = 0):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, filename, status, chunks_count, created_at) VALUES (?, ?, ?, ?, ?)",
            (doc_id, filename, status, chunks_count, datetime.utcnow().isoformat())
        )

def get_document(doc_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

def list_documents():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.getenv("METADATA_DB_PATH", "/app/data/metadata.db")

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        # documents: owner_id и file_size
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                chunks_count INTEGER DEFAULT 0,
                owner_id TEXT NOT NULL DEFAULT 'demo-user',
                file_size INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        # users: план обязателен, expires_at обязателен. Бесплатного тарифа нет.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                plan TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        _migrate(conn)

def _migrate(conn):
    """Добавляет новые колонки, если БД создана старой версией init_db."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "owner_id" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'demo-user'")
    if "file_size" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN file_size INTEGER NOT NULL DEFAULT 0")

def create_document(doc_id: str, filename: str, status: str = "processed",
                    chunks_count: int = 0, owner_id: str = "demo-user",
                    file_size: int = 0):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO documents
               (id, filename, status, chunks_count, owner_id, file_size, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, filename, status, chunks_count, owner_id, file_size, _now())
        )

def get_document(doc_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

def list_documents(owner_id: str | None = None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if owner_id:
            rows = conn.execute(
                "SELECT * FROM documents WHERE owner_id = ? ORDER BY created_at DESC",
                (owner_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

# --- подписка ---
# Тарифы только платные. free нет.
PLAN_LIMITS = {
    "start":      {"docs_per_month": 100,   "max_file_mb": 10},
    "pro":        {"docs_per_month": 1000,  "max_file_mb": 20},
    "enterprise": {"docs_per_month": -1,    "max_file_mb": 100},  # -1 = без лимита
}

def get_user(user_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

def count_docs_this_month(owner_id: str) -> int:
    now = datetime.now(timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM documents "
            "WHERE owner_id = ? AND created_at >= ? "
            "AND status = 'processed'",
            (owner_id, start.isoformat())
        ).fetchone()
        return row[0] if row else 0
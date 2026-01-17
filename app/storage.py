import sqlite3
import time
from pathlib import Path
from typing import Optional, Dict, Any

class JobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
              reference_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL,
              result_path TEXT,
              error TEXT,
              processing_ms INTEGER
            )
            """)
            c.commit()

    def create(self, reference_id: str):
        now = int(time.time() * 1000)
        with self._conn() as c:
            c.execute(
                "INSERT INTO jobs(reference_id,status,created_at_ms,updated_at_ms) VALUES(?,?,?,?)",
                (reference_id, "pending", now, now),
            )
            c.commit()

    def set_status(self, reference_id: str, status: str, **kwargs):
        now = int(time.time() * 1000)
        fields = ["status=?", "updated_at_ms=?"]
        values = [status, now]
        for k, v in kwargs.items():
            fields.append(f"{k}=?")
            values.append(v)
        values.append(reference_id)
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE reference_id=?", values)
            c.commit()

    def get(self, reference_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute("SELECT reference_id,status,result_path,error,processing_ms FROM jobs WHERE reference_id=?",
                            (reference_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "reference_id": row[0],
                "status": row[1],
                "result_path": row[2],
                "error": row[3],
                "processing_ms": row[4],
            }

"""
Document Registry Management for SIH 26117.

SQLite-backed persistent registry of processed files, hashes, categories,
and OCR status. Enables incremental indexing, duplicate skipping, modified-file
reindexing, and deleted-file cleanup.
"""

from contextlib import contextmanager
from datetime import datetime
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from .config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "document_registry.db"


class DocumentRegistry:

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table_and_migrate()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _create_table_and_migrate(self):
        with self._get_connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    file_type TEXT,
                    category TEXT,
                    document_type TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Safe non-breaking schema migration for optional extra columns
            cursor = connection.execute("PRAGMA table_info(documents)")
            columns = {row[1] for row in cursor.fetchall()}

            if "ocr_used" not in columns:
                try:
                    connection.execute("ALTER TABLE documents ADD COLUMN ocr_used INTEGER DEFAULT 0")
                except Exception:
                    pass

            if "chunk_count" not in columns:
                try:
                    connection.execute("ALTER TABLE documents ADD COLUMN chunk_count INTEGER DEFAULT 0")
                except Exception:
                    pass

            connection.commit()

    def get_by_path(self, path: Path) -> Optional[Dict[str, Any]]:
        path_str = str(Path(path).resolve())
        with self._get_connection() as connection:
            cursor = connection.execute(
                """
                SELECT
                    document_id, filename, path, file_hash, file_type,
                    category, document_type, status, created_at, updated_at,
                    ocr_used, chunk_count
                FROM documents
                WHERE path = ?
                """,
                (path_str,)
            )
            row = cursor.fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as connection:
            cursor = connection.execute(
                """
                SELECT
                    document_id, filename, path, file_hash, file_type,
                    category, document_type, status, created_at, updated_at,
                    ocr_used, chunk_count
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,)
            )
            row = cursor.fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    def add(
        self,
        document_id: str,
        filename: str,
        path: Path,
        file_hash: str,
        file_type: str,
        category: str,
        document_type: str,
        status: str = "indexed",
        ocr_used: bool = False,
        chunk_count: int = 0,
    ):
        now = datetime.now().isoformat()
        path_str = str(Path(path).resolve())

        with self._get_connection() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, filename, path, file_hash, file_type,
                    category, document_type, status, created_at, updated_at,
                    ocr_used, chunk_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    path_str,
                    file_hash,
                    file_type,
                    category,
                    document_type,
                    status,
                    now,
                    now,
                    1 if ocr_used else 0,
                    chunk_count,
                )
            )
            connection.commit()

    def update(
        self,
        document_id: str,
        file_hash: str,
        file_type: Optional[str] = None,
        category: Optional[str] = None,
        document_type: Optional[str] = None,
        status: str = "indexed",
        ocr_used: Optional[bool] = None,
        chunk_count: Optional[int] = None,
    ):
        now = datetime.now().isoformat()

        with self._get_connection() as connection:
            connection.execute(
                """
                UPDATE documents
                SET
                    file_hash = ?,
                    file_type = COALESCE(?, file_type),
                    category = COALESCE(?, category),
                    document_type = COALESCE(?, document_type),
                    status = ?,
                    updated_at = ?,
                    ocr_used = COALESCE(?, ocr_used),
                    chunk_count = COALESCE(?, chunk_count)
                WHERE document_id = ?
                """,
                (
                    file_hash,
                    file_type,
                    category,
                    document_type,
                    status,
                    now,
                    1 if ocr_used is True else (0 if ocr_used is False else None),
                    chunk_count,
                    document_id,
                )
            )
            connection.commit()

    def delete(self, document_id: str):
        with self._get_connection() as connection:
            connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,)
            )
            connection.commit()

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as connection:
            cursor = connection.execute(
                """
                SELECT
                    document_id, filename, path, file_hash, file_type,
                    category, document_type, status, created_at, updated_at,
                    ocr_used, chunk_count
                FROM documents
                ORDER BY filename
                """
            )
            rows = cursor.fetchall()

        return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row) -> Dict[str, Any]:
        return {
            "document_id": row[0],
            "filename": row[1],
            "path": row[2],
            "file_hash": row[3],
            "file_type": row[4],
            "category": row[5],
            "document_type": row[6],
            "status": row[7],
            "created_at": row[8],
            "updated_at": row[9],
            "ocr_used": bool(row[10]) if len(row) > 10 and row[10] is not None else False,
            "chunk_count": row[11] if len(row) > 11 and row[11] is not None else 0,
        }
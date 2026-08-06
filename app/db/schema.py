"""Conexión, transacciones, metadatos y esquema (DDL) de la base SQLite.

Este módulo es la base del paquete `app.db`: el resto de módulos
(`fts`, `links`, `documents`, `conversations`, `projects`) importan sus
helpers y constantes, así que no debe importar nada de ellos a nivel de
módulo para evitar un ciclo de imports.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from app.config import settings

DEFAULT_PROJECT_ID = "default"
DEFAULT_PROJECT_NAME = "General"
DEFAULT_VERIFIER_CONTEXT_TOKENS = 8192
DEFAULT_WRITER_CONTEXT_TOKENS = 16384
MIN_AGENT_CONTEXT_TOKENS = 4096
MAX_AGENT_CONTEXT_TOKENS = 262144
FTS_INDEX_VERSION = "1"
FTS_VERSION_META_KEY = "fts_index_version"
FTS_ERROR_META_KEY = "fts_index_error"
GRAPH_INDEX_VERSION = "1"
GRAPH_VERSION_META_KEY = "document_links_version"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    connection = get_connection()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _document_status(metadata: dict[str, Any]) -> str:
    value = metadata.get("estado", metadata.get("status", ""))
    return str(value).strip().casefold()


def _fts_table_exists(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = 'chunks_fts'
        """
    ).fetchone()
    return row is not None


def _set_meta_with_connection(
    connection: sqlite3.Connection,
    key: str,
    value: str,
) -> None:
    connection.execute(
        """
        INSERT INTO app_meta(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def init_db() -> None:
    with transaction() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                mtime REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                links_json TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunk_parents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                parent_index INTEGER NOT NULL,
                heading TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
                    ON DELETE CASCADE,
                UNIQUE(document_id, parent_index)
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                heading TEXT NOT NULL,
                content TEXT NOT NULL,
                parent_id INTEGER,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
                    ON DELETE CASCADE,
                FOREIGN KEY(parent_id) REFERENCES chunk_parents(id)
                    ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS document_links (
                source_document_id INTEGER NOT NULL
                    REFERENCES documents(id) ON DELETE CASCADE,
                target_document_id INTEGER
                    REFERENCES documents(id) ON DELETE CASCADE,
                target_raw TEXT NOT NULL,
                section TEXT NOT NULL DEFAULT '',
                embedded INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                verification_enabled INTEGER NOT NULL DEFAULT 1,
                project_memory_enabled INTEGER NOT NULL DEFAULT 1,
                verifier_context_tokens INTEGER NOT NULL DEFAULT 8192,
                writer_context_tokens INTEGER NOT NULL DEFAULT 16384,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                title TEXT NOT NULL,
                memory_enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                chat_model TEXT NOT NULL,
                agent_metrics_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS conversation_memory (
                conversation_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL DEFAULT '',
                summarized_until_turn_id INTEGER NOT NULL DEFAULT 0,
                summarized_turn_count INTEGER NOT NULL DEFAULT 0,
                summary_model TEXT NOT NULL DEFAULT '',
                last_error TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS project_memory (
                project_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL DEFAULT '',
                summarized_until_turn_id INTEGER NOT NULL DEFAULT 0,
                summarized_turn_count INTEGER NOT NULL DEFAULT 0,
                summary_model TEXT NOT NULL DEFAULT '',
                last_error TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS staged_documents (
                run_id TEXT NOT NULL,
                path TEXT NOT NULL,
                title TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                mtime REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                links_json TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                PRIMARY KEY(run_id, path)
            );

            CREATE TABLE IF NOT EXISTS staged_chunk_parents (
                run_id TEXT NOT NULL,
                document_path TEXT NOT NULL,
                parent_index INTEGER NOT NULL,
                heading TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY(run_id, document_path)
                    REFERENCES staged_documents(run_id, path)
                    ON DELETE CASCADE,
                PRIMARY KEY(run_id, document_path, parent_index)
            );

            CREATE TABLE IF NOT EXISTS staged_chunks (
                run_id TEXT NOT NULL,
                document_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                heading TEXT NOT NULL,
                content TEXT NOT NULL,
                parent_index INTEGER,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                FOREIGN KEY(run_id, document_path)
                    REFERENCES staged_documents(run_id, path)
                    ON DELETE CASCADE,
                PRIMARY KEY(run_id, document_path, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS staged_document_links (
                run_id TEXT NOT NULL,
                source_document_path TEXT NOT NULL,
                target_document_path TEXT,
                target_raw TEXT NOT NULL,
                section TEXT NOT NULL DEFAULT '',
                embedded INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(run_id, source_document_path)
                    REFERENCES staged_documents(run_id, path)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_documents_title
                ON documents(title);
            CREATE INDEX IF NOT EXISTS idx_links_source
                ON document_links(source_document_id);
            CREATE INDEX IF NOT EXISTS idx_links_target
                ON document_links(target_document_id);
            CREATE INDEX IF NOT EXISTS idx_staged_chunks_run
                ON staged_chunks(run_id);
            CREATE INDEX IF NOT EXISTS idx_staged_links_run
                ON staged_document_links(run_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation
                ON conversation_turns(conversation_id, id);
            """
        )
        chunk_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(chunks)"
            ).fetchall()
        }
        if "parent_id" not in chunk_columns:
            connection.execute(
                """
                ALTER TABLE chunks
                ADD COLUMN parent_id INTEGER
                    REFERENCES chunk_parents(id) ON DELETE CASCADE
                """
            )
        staged_chunk_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(staged_chunks)"
            ).fetchall()
        }
        if "parent_index" not in staged_chunk_columns:
            connection.execute(
                """
                ALTER TABLE staged_chunks
                ADD COLUMN parent_index INTEGER
                """
            )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunk_parents_document
            ON chunk_parents(document_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_parent
            ON chunks(parent_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_staged_parents_run
            ON staged_chunk_parents(run_id)
            """
        )
        now = utc_now()
        connection.execute(
            """
            INSERT INTO projects(id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME, now, now),
        )
        project_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(projects)"
            ).fetchall()
        }
        project_migrations = {
            "verification_enabled": (
                "INTEGER NOT NULL DEFAULT 1"
            ),
            "project_memory_enabled": (
                "INTEGER NOT NULL DEFAULT 1"
            ),
            "verifier_context_tokens": (
                "INTEGER NOT NULL DEFAULT 8192"
            ),
            "writer_context_tokens": (
                "INTEGER NOT NULL DEFAULT 16384"
            ),
        }
        for column, definition in project_migrations.items():
            if column not in project_columns:
                connection.execute(
                    f"ALTER TABLE projects ADD COLUMN {column} {definition}"
                )
        conversation_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(conversations)"
            ).fetchall()
        }
        if "project_id" not in conversation_columns:
            connection.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN project_id TEXT
                    REFERENCES projects(id) ON DELETE SET NULL
                """
            )
        if "memory_enabled" not in conversation_columns:
            connection.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN memory_enabled INTEGER NOT NULL DEFAULT 1
                """
            )
        turn_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(conversation_turns)"
            ).fetchall()
        }
        if "agent_metrics_json" not in turn_columns:
            connection.execute(
                """
                ALTER TABLE conversation_turns
                ADD COLUMN agent_metrics_json TEXT NOT NULL DEFAULT '{}'
                """
            )
        connection.execute(
            """
            UPDATE conversations
            SET project_id = ?
            WHERE project_id IS NULL
            """,
            (DEFAULT_PROJECT_ID,),
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_project_updated
            ON conversations(project_id, updated_at DESC)
            """
        )
        # Import diferido: fts.py y links.py importan helpers de este mismo
        # módulo a nivel de módulo, así que un import al principio de este
        # archivo crearía un ciclo. Sólo init_db necesita estas dos llamadas.
        from app.db.fts import _ensure_fts_index
        from app.db.links import _ensure_document_links

        _ensure_fts_index(connection)
        _ensure_document_links(connection)


def get_meta(key: str) -> str | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT value FROM app_meta WHERE key = ?", (key,)
        ).fetchone()
    return str(row["value"]) if row else None

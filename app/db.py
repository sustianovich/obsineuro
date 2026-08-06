from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence
from uuid import uuid4

import numpy as np

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


def _delete_fts_chunks(
    connection: sqlite3.Connection,
    chunk_ids: Sequence[int],
) -> None:
    if not chunk_ids or not _fts_table_exists(connection):
        return
    connection.executemany(
        "DELETE FROM chunks_fts WHERE rowid = ?",
        [(int(chunk_id),) for chunk_id in chunk_ids],
    )


def _insert_fts_chunk(
    connection: sqlite3.Connection,
    *,
    chunk_id: int,
    title: str,
    heading: str,
    content: str,
    path: str,
    document_status: str,
) -> None:
    if not _fts_table_exists(connection):
        return
    connection.execute(
        """
        INSERT INTO chunks_fts(
            rowid, title, heading, content, path, document_status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            title,
            heading,
            content,
            path,
            document_status,
        ),
    )


def _rebuild_fts_index(connection: sqlite3.Connection) -> None:
    if not _fts_table_exists(connection):
        return
    connection.execute("DELETE FROM chunks_fts")
    rows = connection.execute(
        """
        SELECT
            c.id AS chunk_id,
            c.heading,
            c.content,
            d.title,
            d.path,
            d.metadata_json
        FROM chunks AS c
        JOIN documents AS d ON d.id = c.document_id
        ORDER BY c.id
        """
    ).fetchall()
    values = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        values.append(
            (
                int(row["chunk_id"]),
                str(row["title"]),
                str(row["heading"]),
                str(row["content"]),
                str(row["path"]),
                _document_status(
                    metadata if isinstance(metadata, dict) else {}
                ),
            )
        )
    connection.executemany(
        """
        INSERT INTO chunks_fts(
            rowid, title, heading, content, path, document_status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        values,
    )


def _ensure_fts_index(connection: sqlite3.Connection) -> None:
    connection.execute("SAVEPOINT ensure_fts")
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                title,
                heading,
                content,
                path,
                document_status UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2'
            )
            """
        )
        chunk_count = int(
            connection.execute(
                "SELECT COUNT(*) AS total FROM chunks"
            ).fetchone()["total"]
        )
        fts_count = int(
            connection.execute(
                "SELECT COUNT(*) AS total FROM chunks_fts"
            ).fetchone()["total"]
        )
        version_row = connection.execute(
            "SELECT value FROM app_meta WHERE key = ?",
            (FTS_VERSION_META_KEY,),
        ).fetchone()
        version = str(version_row["value"]) if version_row else None
        if (
            version != FTS_INDEX_VERSION
            or fts_count != chunk_count
        ):
            _rebuild_fts_index(connection)
        _set_meta_with_connection(
            connection,
            FTS_VERSION_META_KEY,
            FTS_INDEX_VERSION,
        )
        _set_meta_with_connection(connection, FTS_ERROR_META_KEY, "")
        connection.execute("RELEASE SAVEPOINT ensure_fts")
    except sqlite3.Error as exc:
        connection.execute("ROLLBACK TO SAVEPOINT ensure_fts")
        connection.execute("RELEASE SAVEPOINT ensure_fts")
        _set_meta_with_connection(
            connection,
            FTS_ERROR_META_KEY,
            " ".join(str(exc).split())[:500],
        )


def _decoded_links(value: Any) -> list[dict[str, Any]]:
    try:
        links = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(links, list):
        return []
    return [link for link in links if isinstance(link, dict)]


def _resolved_link_values(
    documents: Sequence[dict[str, Any]],
    *,
    identity_key: str,
) -> list[tuple[Any, Any | None, str, str, int]]:
    from app.rag.graph import build_alias_index, resolve_document_target

    aliases = build_alias_index(documents)
    values: list[tuple[Any, Any | None, str, str, int]] = []
    for document in documents:
        for link in _decoded_links(document.get("links_json")):
            target_raw = str(link.get("target", "")).strip()
            if not target_raw:
                continue
            values.append(
                (
                    document[identity_key],
                    resolve_document_target(aliases, target_raw),
                    target_raw,
                    str(link.get("section", "")).strip(),
                    int(bool(link.get("embedded", False))),
                )
            )
    return values


def _active_documents(
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, path, title, links_json
        FROM documents
        ORDER BY id
        """
    ).fetchall()
    return [
        {
            "document_id": int(row["id"]),
            "path": str(row["path"]),
            "title": str(row["title"]),
            "links_json": row["links_json"],
        }
        for row in rows
    ]


def _rebuild_document_links(connection: sqlite3.Connection) -> int:
    """Resuelve de nuevo todo el grafo tras cambiar el directorio de notas.

    Los alias son globales: modificar una sola nota puede cambiar el destino
    de enlaces escritos en otras. La reconstrucción completa dentro de la
    transacción evita backlinks huérfanos u obsoletos.
    """
    values = _resolved_link_values(
        _active_documents(connection),
        identity_key="document_id",
    )
    connection.execute("DELETE FROM document_links")
    connection.executemany(
        """
        INSERT INTO document_links(
            source_document_id, target_document_id,
            target_raw, section, embedded
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        values,
    )
    _set_meta_with_connection(
        connection,
        GRAPH_VERSION_META_KEY,
        GRAPH_INDEX_VERSION,
    )
    return len(values)


def _lightweight_alias_index(connection: sqlite3.Connection) -> dict[str, Any]:
    """Índice de alias sin decodificar `links_json` de cada nota.

    `_rebuild_document_links` lee y decodifica los enlaces de todo el vault
    porque reconstruye el grafo entero. Una sustitución incremental sólo
    necesita saber a qué documento resuelve cada alias, no releer los
    enlaces de notas que no cambiaron.
    """
    from app.rag.graph import build_alias_index

    rows = connection.execute(
        "SELECT id, path, title FROM documents ORDER BY id"
    ).fetchall()
    return build_alias_index(
        {
            "document_id": int(row["id"]),
            "path": str(row["path"]),
            "title": str(row["title"]),
        }
        for row in rows
    )


def _insert_resolved_links(
    connection: sqlite3.Connection,
    values: list[tuple[Any, Any | None, str, str, int]],
) -> None:
    if not values:
        return
    connection.executemany(
        """
        INSERT INTO document_links(
            source_document_id, target_document_id,
            target_raw, section, embedded
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        values,
    )


def _replace_outgoing_document_links(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    links: list[dict[str, Any]],
    aliases: dict[str, Any],
) -> None:
    """Sustituye únicamente las aristas salientes de un documento."""
    from app.rag.graph import resolve_document_target

    connection.execute(
        "DELETE FROM document_links WHERE source_document_id = ?",
        (document_id,),
    )
    values = [
        (
            document_id,
            resolve_document_target(aliases, target_raw),
            target_raw,
            str(link.get("section", "")).strip(),
            int(bool(link.get("embedded", False))),
        )
        for link in links
        for target_raw in [str(link.get("target", "")).strip()]
        if target_raw
    ]
    _insert_resolved_links(connection, values)


def _reinsert_captured_links(
    connection: sqlite3.Connection,
    rows: Sequence[sqlite3.Row],
    aliases: dict[str, Any],
) -> None:
    """Reinserta aristas que un borrado en cascada se llevó por delante,
    resolviendo su destino contra un índice de alias ya actualizado."""
    from app.rag.graph import resolve_document_target

    values = [
        (
            int(row["source_document_id"]),
            resolve_document_target(aliases, str(row["target_raw"])),
            str(row["target_raw"]),
            str(row["section"]),
            int(row["embedded"]),
        )
        for row in rows
    ]
    _insert_resolved_links(connection, values)


def _reresolve_broken_links(
    connection: sqlite3.Connection,
    aliases: dict[str, Any],
) -> None:
    """Reintenta resolver los enlaces rotos por si la nota tocada los
    completa; no toca las aristas que ya estaban resueltas."""
    from app.rag.graph import resolve_document_target

    rows = connection.execute(
        """
        SELECT rowid, target_raw
        FROM document_links
        WHERE target_document_id IS NULL
        """
    ).fetchall()
    updates = [
        (
            resolve_document_target(aliases, str(row["target_raw"])),
            int(row["rowid"]),
        )
        for row in rows
    ]
    updates = [
        (target, rowid) for target, rowid in updates if target is not None
    ]
    if updates:
        connection.executemany(
            "UPDATE document_links SET target_document_id = ? WHERE rowid = ?",
            updates,
        )


def _ensure_document_links(connection: sqlite3.Connection) -> None:
    """Migra `links_json` sin tocar fragmentos, embeddings ni su huella."""
    version_row = connection.execute(
        "SELECT value FROM app_meta WHERE key = ?",
        (GRAPH_VERSION_META_KEY,),
    ).fetchone()
    version = str(version_row["value"]) if version_row else None
    if version != GRAPH_INDEX_VERSION:
        _rebuild_document_links(connection)


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
        _ensure_fts_index(connection)
        _ensure_document_links(connection)


def get_meta(key: str) -> str | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT value FROM app_meta WHERE key = ?", (key,)
        ).fetchone()
    return str(row["value"]) if row else None


def reset_staged_index() -> None:
    """Elimina preparaciones antiguas sin tocar el índice activo."""
    with transaction() as connection:
        connection.execute("DELETE FROM staged_document_links")
        connection.execute("DELETE FROM staged_chunks")
        connection.execute("DELETE FROM staged_chunk_parents")
        connection.execute("DELETE FROM staged_documents")


def discard_staged_index(run_id: str) -> None:
    with transaction() as connection:
        connection.execute(
            "DELETE FROM staged_document_links WHERE run_id = ?",
            (run_id,),
        )
        connection.execute(
            "DELETE FROM staged_chunks WHERE run_id = ?",
            (run_id,),
        )
        connection.execute(
            "DELETE FROM staged_chunk_parents WHERE run_id = ?",
            (run_id,),
        )
        connection.execute(
            "DELETE FROM staged_documents WHERE run_id = ?",
            (run_id,),
        )


def get_document_hashes() -> dict[str, str]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT path, sha256 FROM documents"
        ).fetchall()
    return {str(row["path"]): str(row["sha256"]) for row in rows}


def delete_documents_not_in(existing_paths: set[str]) -> int:
    with transaction() as connection:
        rows = connection.execute("SELECT path FROM documents").fetchall()
        deleted = 0
        for row in rows:
            path = str(row["path"])
            if path not in existing_paths:
                chunk_rows = connection.execute(
                    """
                    SELECT c.id
                    FROM chunks AS c
                    JOIN documents AS d ON d.id = c.document_id
                    WHERE d.path = ?
                    """,
                    (path,),
                ).fetchall()
                _delete_fts_chunks(
                    connection,
                    [int(chunk_row["id"]) for chunk_row in chunk_rows],
                )
                connection.execute(
                    "DELETE FROM documents WHERE path = ?", (path,)
                )
                deleted += 1
        if deleted:
            _rebuild_document_links(connection)
    return deleted


def replace_document(
    *,
    path: str,
    title: str,
    sha256: str,
    mtime: float,
    metadata: dict[str, Any],
    links: list[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    parents: Sequence[dict[str, Any]] | None = None,
) -> None:
    with transaction() as connection:
        previous = connection.execute(
            "SELECT id, title FROM documents WHERE path = ?",
            (path,),
        ).fetchone()
        previous_document_id = int(previous["id"]) if previous else None
        # Sólo un cambio de título (o una nota nueva) puede alterar a qué
        # resuelve un alias en cualquier otro documento. Una edición de
        # contenido con la misma ruta y el mismo título no puede afectar a
        # nadie más: basta con refrescar los enlaces propios de la nota.
        identity_changed = (
            previous is None or str(previous["title"]) != title
        )

        # El id es AUTOINCREMENT y cambia en cada sustitución aunque la nota
        # sea la misma. El borrado más abajo se lleva en cascada las aristas
        # que otras notas tenían apuntando a ella; se capturan antes para
        # reinsertarlas apuntando al id nuevo.
        incoming_edges: list[sqlite3.Row] = []
        if previous_document_id is not None:
            incoming_edges = connection.execute(
                """
                SELECT source_document_id, target_raw, section, embedded
                FROM document_links
                WHERE target_document_id = ?
                  AND source_document_id != ?
                """,
                (previous_document_id, previous_document_id),
            ).fetchall()

        old_chunk_rows = connection.execute(
            """
            SELECT c.id
            FROM chunks AS c
            JOIN documents AS d ON d.id = c.document_id
            WHERE d.path = ?
            """,
            (path,),
        ).fetchall()
        _delete_fts_chunks(
            connection,
            [int(row["id"]) for row in old_chunk_rows],
        )
        connection.execute("DELETE FROM documents WHERE path = ?", (path,))
        cursor = connection.execute(
            """
            INSERT INTO documents(
                path, title, sha256, mtime,
                metadata_json, links_json, indexed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                path,
                title,
                sha256,
                mtime,
                json.dumps(metadata, ensure_ascii=False, default=str),
                json.dumps(links, ensure_ascii=False, default=str),
                utc_now(),
            ),
        )
        document_id = int(cursor.lastrowid)

        parent_ids: dict[int, int] = {}
        for parent in parents or ():
            parent_index = int(parent["parent_index"])
            parent_cursor = connection.execute(
                """
                INSERT INTO chunk_parents(
                    document_id, parent_index, heading, content
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    document_id,
                    parent_index,
                    str(parent["heading"]),
                    str(parent["content"]),
                ),
            )
            parent_ids[parent_index] = int(parent_cursor.lastrowid)

        for chunk in chunks:
            vector = np.asarray(chunk["embedding"], dtype=np.float32)
            chunk_heading = str(chunk["heading"])
            chunk_content = str(chunk["content"])
            chunk_parent_index = chunk.get("parent_index")
            parent_id = (
                parent_ids[int(chunk_parent_index)]
                if chunk_parent_index is not None
                else None
            )
            chunk_cursor = connection.execute(
                """
                INSERT INTO chunks(
                    document_id, chunk_index, heading,
                    content, parent_id, embedding, embedding_dim
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    int(chunk["chunk_index"]),
                    chunk_heading,
                    chunk_content,
                    parent_id,
                    vector.tobytes(),
                    int(vector.size),
                ),
            )
            _insert_fts_chunk(
                connection,
                chunk_id=int(chunk_cursor.lastrowid),
                title=title,
                heading=chunk_heading,
                content=chunk_content,
                path=path,
                document_status=_document_status(metadata),
            )

        # Nunca recorre el grafo entero: como mucho toca las aristas propias
        # de esta nota, las que otras notas tenían apuntando a ella y, si
        # cambió de identidad, las que estaban rotas.
        aliases = _lightweight_alias_index(connection)
        _replace_outgoing_document_links(
            connection,
            document_id=document_id,
            links=links,
            aliases=aliases,
        )
        _reinsert_captured_links(connection, incoming_edges, aliases)
        if identity_changed:
            _reresolve_broken_links(connection, aliases)


def stage_document(
    *,
    run_id: str,
    path: str,
    title: str,
    sha256: str,
    mtime: float,
    metadata: dict[str, Any],
    links: list[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    parents: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Guarda un documento en la preparación, nunca en el índice activo."""
    with transaction() as connection:
        connection.execute(
            """
            DELETE FROM staged_documents
            WHERE run_id = ? AND path = ?
            """,
            (run_id, path),
        )
        connection.execute(
            """
            INSERT INTO staged_documents(
                run_id, path, title, sha256, mtime,
                metadata_json, links_json, indexed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                path,
                title,
                sha256,
                mtime,
                json.dumps(metadata, ensure_ascii=False, default=str),
                json.dumps(links, ensure_ascii=False, default=str),
                utc_now(),
            ),
        )

        for parent in parents or ():
            connection.execute(
                """
                INSERT INTO staged_chunk_parents(
                    run_id, document_path, parent_index, heading, content
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    path,
                    int(parent["parent_index"]),
                    str(parent["heading"]),
                    str(parent["content"]),
                ),
            )

        for chunk in chunks:
            vector = np.asarray(chunk["embedding"], dtype=np.float32)
            connection.execute(
                """
                INSERT INTO staged_chunks(
                    run_id, document_path, chunk_index, heading,
                    content, parent_index, embedding, embedding_dim
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    path,
                    int(chunk["chunk_index"]),
                    str(chunk["heading"]),
                    str(chunk["content"]),
                    chunk.get("parent_index"),
                    vector.tobytes(),
                    int(vector.size),
                ),
            )


def resolve_staged_document_links(run_id: str) -> int:
    """Materializa el grafo preparado cuando ya existen todas las notas."""
    with transaction() as connection:
        rows = connection.execute(
            """
            SELECT path, title, links_json
            FROM staged_documents
            WHERE run_id = ?
            ORDER BY path
            """,
            (run_id,),
        ).fetchall()
        documents = [
            {
                "path": str(row["path"]),
                "title": str(row["title"]),
                "links_json": row["links_json"],
            }
            for row in rows
        ]
        values = _resolved_link_values(documents, identity_key="path")
        connection.execute(
            "DELETE FROM staged_document_links WHERE run_id = ?",
            (run_id,),
        )
        connection.executemany(
            """
            INSERT INTO staged_document_links(
                run_id, source_document_path, target_document_path,
                target_raw, section, embedded
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (run_id, source, target, target_raw, section, embedded)
                for source, target, target_raw, section, embedded in values
            ],
        )
        return len(values)


def commit_staged_index(
    *,
    run_id: str,
    fingerprint_key: str,
    fingerprint_value: str,
    expected_documents: int,
    expected_chunks: int,
    expected_links: int,
    expected_parents: int | None = None,
) -> None:
    """Sustituye índice y huella juntos dentro de una única transacción."""
    with transaction() as connection:
        staged_documents = int(
            connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM staged_documents
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()["total"]
        )
        staged_chunks = int(
            connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM staged_chunks
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()["total"]
        )
        staged_links = int(
            connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM staged_document_links
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()["total"]
        )
        staged_parents = int(
            connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM staged_chunk_parents
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()["total"]
        )
        orphaned_children = int(
            connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM staged_chunks AS child
                LEFT JOIN staged_chunk_parents AS parent
                  ON parent.run_id = child.run_id
                 AND parent.document_path = child.document_path
                 AND parent.parent_index = child.parent_index
                WHERE child.run_id = ?
                  AND child.parent_index IS NOT NULL
                  AND parent.parent_index IS NULL
                """,
                (run_id,),
            ).fetchone()["total"]
        )
        if (
            staged_documents != expected_documents
            or staged_chunks != expected_chunks
            or staged_links != expected_links
            or (
                expected_parents is not None
                and staged_parents != expected_parents
            )
            or orphaned_children != 0
        ):
            raise ValueError(
                "La preparación del índice está incompleta; "
                "se conserva el índice anterior."
            )

        if _fts_table_exists(connection):
            connection.execute("DELETE FROM chunks_fts")
        connection.execute("DELETE FROM documents")
        connection.execute(
            """
            INSERT INTO documents(
                path, title, sha256, mtime,
                metadata_json, links_json, indexed_at
            )
            SELECT
                path, title, sha256, mtime,
                metadata_json, links_json, indexed_at
            FROM staged_documents
            WHERE run_id = ?
            ORDER BY path
            """,
            (run_id,),
        )
        connection.execute(
            """
            INSERT INTO chunk_parents(
                document_id, parent_index, heading, content
            )
            SELECT
                d.id, parent.parent_index, parent.heading, parent.content
            FROM staged_chunk_parents AS parent
            JOIN documents AS d ON d.path = parent.document_path
            WHERE parent.run_id = ?
            ORDER BY parent.document_path, parent.parent_index
            """,
            (run_id,),
        )
        connection.execute(
            """
            INSERT INTO chunks(
                document_id, chunk_index, heading,
                content, parent_id, embedding, embedding_dim
            )
            SELECT
                d.id, sc.chunk_index, sc.heading,
                sc.content, parent.id, sc.embedding, sc.embedding_dim
            FROM staged_chunks AS sc
            JOIN documents AS d ON d.path = sc.document_path
            LEFT JOIN chunk_parents AS parent
              ON parent.document_id = d.id
             AND parent.parent_index = sc.parent_index
            WHERE sc.run_id = ?
            ORDER BY sc.document_path, sc.chunk_index
            """,
            (run_id,),
        )
        connection.execute(
            """
            INSERT INTO document_links(
                source_document_id, target_document_id,
                target_raw, section, embedded
            )
            SELECT
                source.id, target.id,
                link.target_raw, link.section, link.embedded
            FROM staged_document_links AS link
            JOIN documents AS source
                ON source.path = link.source_document_path
            LEFT JOIN documents AS target
                ON target.path = link.target_document_path
            WHERE link.run_id = ?
            ORDER BY link.rowid
            """,
            (run_id,),
        )
        _rebuild_fts_index(connection)
        active_documents = int(
            connection.execute(
                "SELECT COUNT(*) AS total FROM documents"
            ).fetchone()["total"]
        )
        active_chunks = int(
            connection.execute(
                "SELECT COUNT(*) AS total FROM chunks"
            ).fetchone()["total"]
        )
        active_parents = int(
            connection.execute(
                "SELECT COUNT(*) AS total FROM chunk_parents"
            ).fetchone()["total"]
        )
        active_fts_chunks = (
            int(
                connection.execute(
                    "SELECT COUNT(*) AS total FROM chunks_fts"
                ).fetchone()["total"]
            )
            if _fts_table_exists(connection)
            else active_chunks
        )
        active_links = int(
            connection.execute(
                "SELECT COUNT(*) AS total FROM document_links"
            ).fetchone()["total"]
        )
        if (
            active_documents != expected_documents
            or active_chunks != expected_chunks
            or active_fts_chunks != expected_chunks
            or active_links != expected_links
            or (
                expected_parents is not None
                and active_parents != expected_parents
            )
        ):
            raise RuntimeError(
                "La activación del índice no produjo todos los documentos "
                "y fragmentos esperados; se revierte la transacción."
            )

        connection.execute(
            """
            INSERT INTO app_meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (fingerprint_key, fingerprint_value),
        )
        _set_meta_with_connection(
            connection,
            GRAPH_VERSION_META_KEY,
            GRAPH_INDEX_VERSION,
        )
        connection.execute(
            "DELETE FROM staged_document_links WHERE run_id = ?",
            (run_id,),
        )
        connection.execute(
            "DELETE FROM staged_chunks WHERE run_id = ?",
            (run_id,),
        )
        connection.execute(
            "DELETE FROM staged_chunk_parents WHERE run_id = ?",
            (run_id,),
        )
        connection.execute(
            "DELETE FROM staged_documents WHERE run_id = ?",
            (run_id,),
        )


def _get_fts5_status_with_connection(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    available = _fts_table_exists(connection)
    expected_chunks = int(
        connection.execute(
            "SELECT COUNT(*) AS total FROM chunks"
        ).fetchone()["total"]
    )
    indexed_chunks = 0
    if available:
        try:
            indexed_chunks = int(
                connection.execute(
                    "SELECT COUNT(*) AS total FROM chunks_fts"
                ).fetchone()["total"]
            )
        except sqlite3.Error:
            available = False
    version_row = connection.execute(
        "SELECT value FROM app_meta WHERE key = ?",
        (FTS_VERSION_META_KEY,),
    ).fetchone()
    error_row = connection.execute(
        "SELECT value FROM app_meta WHERE key = ?",
        (FTS_ERROR_META_KEY,),
    ).fetchone()
    version = str(version_row["value"]) if version_row else None
    error = str(error_row["value"]).strip() if error_row else ""
    return {
        "available": available,
        "version": version,
        "indexed_chunks": indexed_chunks,
        "expected_chunks": expected_chunks,
        "synchronized": (
            available
            and version == FTS_INDEX_VERSION
            and indexed_chunks == expected_chunks
            and not error
        ),
        "error": error or None,
    }


def get_fts5_status() -> dict[str, Any]:
    with get_connection() as connection:
        return _get_fts5_status_with_connection(connection)


def _ambiguous_alias_count(connection: sqlite3.Connection) -> int:
    """Cuenta alias (título/ruta/nombre de archivo) reclamados por más de
    una nota, p.ej. dos "Index.md" en carpetas distintas."""
    from app.rag.graph import count_ambiguous_aliases

    rows = connection.execute(
        "SELECT id, path, title FROM documents"
    ).fetchall()
    return count_ambiguous_aliases(
        {
            "document_id": int(row["id"]),
            "path": str(row["path"]),
            "title": str(row["title"]),
        }
        for row in rows
    )


def get_document_graph_status() -> dict[str, Any]:
    with get_connection() as connection:
        available = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'document_links'
            """
        ).fetchone() is not None
        if not available:
            return {
                "available": False,
                "total_edges": 0,
                "resolved_edges": 0,
                "broken_edges": 0,
                "orphan_documents": 0,
                "ambiguous_aliases": 0,
            }

        counts = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(target_document_id) AS resolved,
                SUM(CASE WHEN target_document_id IS NULL THEN 1 ELSE 0 END)
                    AS broken
            FROM document_links
            """
        ).fetchone()
        orphan_documents = int(
            connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM documents AS document
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM document_links AS link
                    WHERE link.target_document_id IS NOT NULL
                      AND (
                        link.source_document_id = document.id
                        OR link.target_document_id = document.id
                      )
                )
                """
            ).fetchone()["total"]
        )
        ambiguous_aliases = _ambiguous_alias_count(connection)
    return {
        "available": True,
        "total_edges": int(counts["total"]),
        "resolved_edges": int(counts["resolved"]),
        "broken_edges": int(counts["broken"] or 0),
        "orphan_documents": orphan_documents,
        "ambiguous_aliases": ambiguous_aliases,
    }


def search_chunks_fts(
    match_query: str,
    *,
    limit: int,
    status: str | None = None,
) -> list[dict[str, Any]]:
    if not match_query.strip() or limit <= 0:
        return []
    with get_connection() as connection:
        if not _get_fts5_status_with_connection(connection)["synchronized"]:
            return []
        parameters: list[Any] = [match_query]
        status_filter = ""
        if status:
            status_filter = "AND document_status = ?"
            parameters.append(status.strip().casefold())
        parameters.append(limit)
        rows = connection.execute(
            f"""
            SELECT
                rowid AS chunk_id,
                bm25(chunks_fts, 5.0, 3.0, 1.0, 2.0, 0.0)
                    AS bm25_score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
                {status_filter}
            ORDER BY bm25_score
            LIMIT ?
            """,
            parameters,
        ).fetchall()
    return [
        {
            "chunk_id": int(row["chunk_id"]),
            "bm25_score": float(row["bm25_score"]),
        }
        for row in rows
    ]


def get_stats() -> dict[str, int]:
    with get_connection() as connection:
        documents = connection.execute(
            "SELECT COUNT(*) AS total FROM documents"
        ).fetchone()["total"]
        chunks = connection.execute(
            "SELECT COUNT(*) AS total FROM chunks"
        ).fetchone()["total"]
        parents = connection.execute(
            "SELECT COUNT(*) AS total FROM chunk_parents"
        ).fetchone()["total"]
    return {
        "documents": int(documents),
        "chunks": int(chunks),
        "parents": int(parents),
    }


def get_document_statuses() -> list[dict[str, Any]]:
    """Estados documentales realmente presentes en el vault, con recuento.

    Permite que el desplegable de la interfaz refleje el vocabulario del
    vault del cliente en lugar de una lista fija escrita a mano.
    """
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT metadata_json FROM documents"
        ).fetchall()

    counts: dict[str, int] = {}
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        value = _document_status(metadata)
        if value:
            counts[value] = counts.get(value, 0) + 1

    return [
        {"value": value, "documents": count}
        for value, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def get_embedding_dimensions() -> list[int]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT embedding_dim
            FROM chunks
            ORDER BY embedding_dim
            """
        ).fetchall()
    return [int(row["embedding_dim"]) for row in rows]


def conversation_title(question: str, max_length: int = 72) -> str:
    normalized = " ".join(question.split()).strip()
    if not normalized:
        return "Nueva conversación"
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def conversation_exists(conversation_id: str) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return row is not None


def _memory_status_from_connection(
    connection: sqlite3.Connection,
    conversation_id: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
            c.memory_enabled,
            COUNT(t.id) AS total_turns,
            COALESCE(m.summary, '') AS summary,
            COALESCE(m.summarized_until_turn_id, 0)
                AS summarized_until_turn_id,
            COALESCE(m.summarized_turn_count, 0)
                AS summarized_turn_count,
            COALESCE(m.summary_model, '') AS summary_model,
            m.last_error,
            m.updated_at AS memory_updated_at
        FROM conversations AS c
        LEFT JOIN conversation_turns AS t
            ON t.conversation_id = c.id
        LEFT JOIN conversation_memory AS m
            ON m.conversation_id = c.id
        WHERE c.id = ?
        GROUP BY c.id
        """,
        (conversation_id,),
    ).fetchone()
    if row is None:
        return None
    total_turns = int(row["total_turns"])
    summarized_turns = int(row["summarized_turn_count"])
    return {
        "enabled": bool(row["memory_enabled"]),
        "has_summary": bool(str(row["summary"]).strip()),
        "summary": str(row["summary"]),
        "summarized_until_turn_id": int(
            row["summarized_until_turn_id"]
        ),
        "summarized_turns": summarized_turns,
        "pending_turns": max(0, total_turns - summarized_turns),
        "total_turns": total_turns,
        "summary_model": str(row["summary_model"]),
        "last_error": (
            str(row["last_error"]) if row["last_error"] else None
        ),
        "updated_at": (
            str(row["memory_updated_at"])
            if row["memory_updated_at"]
            else None
        ),
    }


def get_conversation_memory_status(
    conversation_id: str,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        return _memory_status_from_connection(connection, conversation_id)


def set_conversation_memory_enabled(
    conversation_id: str,
    enabled: bool,
) -> dict[str, Any] | None:
    with transaction() as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET memory_enabled = ?, updated_at = ?
            WHERE id = ?
            """,
            (int(enabled), utc_now(), conversation_id),
        )
        if cursor.rowcount == 0:
            return None
        return _memory_status_from_connection(connection, conversation_id)


def load_conversation_memory_context(
    conversation_id: str,
    *,
    recent_turn_limit: int,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        status = _memory_status_from_connection(
            connection,
            conversation_id,
        )
        if status is None:
            return None
        rows = connection.execute(
            """
            SELECT id, question, answer
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, recent_turn_limit),
        ).fetchall()
    turns = [
        {
            "id": int(row["id"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
        }
        for row in reversed(rows)
    ]
    return {
        "enabled": status["enabled"],
        "summary": status["summary"],
        "turns": turns,
    }


def get_conversation_memory_batch(
    conversation_id: str,
    *,
    batch_size: int,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        conversation = connection.execute(
            """
            SELECT memory_enabled
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None
        memory = connection.execute(
            """
            SELECT
                summary,
                summarized_until_turn_id,
                summarized_turn_count
            FROM conversation_memory
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        summarized_until = (
            int(memory["summarized_until_turn_id"]) if memory else 0
        )
        rows = connection.execute(
            """
            SELECT id, question, answer
            FROM conversation_turns
            WHERE conversation_id = ? AND id > ?
            ORDER BY id
            LIMIT ?
            """,
            (conversation_id, summarized_until, batch_size),
        ).fetchall()
    if not bool(conversation["memory_enabled"]) or len(rows) < batch_size:
        return None
    return {
        "previous_summary": str(memory["summary"]) if memory else "",
        "previous_summarized_turn_count": (
            int(memory["summarized_turn_count"]) if memory else 0
        ),
        "turns": [
            {
                "id": int(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }
            for row in rows
        ],
    }


def save_conversation_memory_summary(
    *,
    conversation_id: str,
    summary: str,
    summarized_until_turn_id: int,
    summarized_turn_count: int,
    summary_model: str,
) -> bool:
    now = utc_now()
    with transaction() as connection:
        current = connection.execute(
            """
            SELECT summarized_until_turn_id
            FROM conversation_memory
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        current_turn_id = (
            int(current["summarized_until_turn_id"]) if current else 0
        )
        if summarized_until_turn_id <= current_turn_id:
            return False
        connection.execute(
            """
            INSERT INTO conversation_memory(
                conversation_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                last_error,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                summary = excluded.summary,
                summarized_until_turn_id =
                    excluded.summarized_until_turn_id,
                summarized_turn_count = excluded.summarized_turn_count,
                summary_model = excluded.summary_model,
                last_error = NULL,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                now,
            ),
        )
        return True


def set_conversation_memory_error(
    conversation_id: str,
    error: str,
) -> None:
    now = utc_now()
    safe_error = " ".join(error.split()).strip()[:500]
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO conversation_memory(
                conversation_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                last_error,
                updated_at
            )
            VALUES (?, '', 0, 0, '', ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            """,
            (conversation_id, safe_error, now),
        )


def project_exists(project_id: str) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
    return row is not None


def _validate_context_tokens(value: int, label: str) -> int:
    normalized = int(value)
    if not MIN_AGENT_CONTEXT_TOKENS <= normalized <= MAX_AGENT_CONTEXT_TOKENS:
        raise ValueError(
            f"{label} debe estar entre {MIN_AGENT_CONTEXT_TOKENS} y "
            f"{MAX_AGENT_CONTEXT_TOKENS} tokens."
        )
    return normalized


def _project_agent_settings_from_row(
    row: sqlite3.Row,
) -> dict[str, Any]:
    return {
        "verification_enabled": bool(row["verification_enabled"]),
        "project_memory_enabled": bool(row["project_memory_enabled"]),
        "verifier_context_tokens": int(
            row["verifier_context_tokens"]
        ),
        "writer_context_tokens": int(row["writer_context_tokens"]),
    }


def get_project_agent_settings(
    project_id: str,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                verification_enabled,
                project_memory_enabled,
                verifier_context_tokens,
                writer_context_tokens
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()
    return _project_agent_settings_from_row(row) if row else None


def update_project_agent_settings(
    project_id: str,
    *,
    verification_enabled: bool,
    project_memory_enabled: bool,
    verifier_context_tokens: int,
    writer_context_tokens: int,
) -> dict[str, Any] | None:
    verifier_tokens = _validate_context_tokens(
        verifier_context_tokens,
        "La ventana del verificador",
    )
    writer_tokens = _validate_context_tokens(
        writer_context_tokens,
        "La ventana del redactor",
    )
    with transaction() as connection:
        cursor = connection.execute(
            """
            UPDATE projects
            SET
                verification_enabled = ?,
                project_memory_enabled = ?,
                verifier_context_tokens = ?,
                writer_context_tokens = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                int(verification_enabled),
                int(project_memory_enabled),
                verifier_tokens,
                writer_tokens,
                utc_now(),
                project_id,
            ),
        )
        if cursor.rowcount == 0:
            return None
    return get_project_agent_settings(project_id)


def _project_memory_status_from_connection(
    connection: sqlite3.Connection,
    project_id: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
            p.project_memory_enabled,
            COUNT(t.id) AS total_turns,
            COALESCE(m.summary, '') AS summary,
            COALESCE(m.summarized_until_turn_id, 0)
                AS summarized_until_turn_id,
            COALESCE(m.summarized_turn_count, 0)
                AS summarized_turn_count,
            COALESCE(m.summary_model, '') AS summary_model,
            m.last_error,
            m.updated_at AS memory_updated_at
        FROM projects AS p
        LEFT JOIN conversations AS c ON c.project_id = p.id
        LEFT JOIN conversation_turns AS t
            ON t.conversation_id = c.id
        LEFT JOIN project_memory AS m ON m.project_id = p.id
        WHERE p.id = ?
        GROUP BY p.id
        """,
        (project_id,),
    ).fetchone()
    if row is None:
        return None
    total_turns = int(row["total_turns"])
    summarized_turns = int(row["summarized_turn_count"])
    return {
        "enabled": bool(row["project_memory_enabled"]),
        "has_summary": bool(str(row["summary"]).strip()),
        "summary": str(row["summary"]),
        "summarized_until_turn_id": int(
            row["summarized_until_turn_id"]
        ),
        "summarized_turns": summarized_turns,
        "pending_turns": max(0, total_turns - summarized_turns),
        "total_turns": total_turns,
        "summary_model": str(row["summary_model"]),
        "last_error": (
            str(row["last_error"]) if row["last_error"] else None
        ),
        "updated_at": (
            str(row["memory_updated_at"])
            if row["memory_updated_at"]
            else None
        ),
    }


def get_project_memory_status(
    project_id: str,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        return _project_memory_status_from_connection(
            connection,
            project_id,
        )


def load_project_memory_context(
    project_id: str,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        status = _project_memory_status_from_connection(
            connection,
            project_id,
        )
    if status is None:
        return None
    return {
        "enabled": status["enabled"],
        "summary": status["summary"],
    }


def get_project_memory_batch(
    project_id: str,
    *,
    batch_size: int,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        project = connection.execute(
            """
            SELECT project_memory_enabled
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()
        if project is None:
            return None
        memory = connection.execute(
            """
            SELECT
                summary,
                summarized_until_turn_id,
                summarized_turn_count
            FROM project_memory
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        summarized_until = (
            int(memory["summarized_until_turn_id"]) if memory else 0
        )
        rows = connection.execute(
            """
            SELECT
                t.id,
                c.title AS conversation_title,
                t.question,
                t.answer
            FROM conversation_turns AS t
            JOIN conversations AS c ON c.id = t.conversation_id
            WHERE c.project_id = ? AND t.id > ?
            ORDER BY t.id
            LIMIT ?
            """,
            (project_id, summarized_until, batch_size),
        ).fetchall()
    if (
        not bool(project["project_memory_enabled"])
        or len(rows) < batch_size
    ):
        return None
    return {
        "previous_summary": str(memory["summary"]) if memory else "",
        "previous_summarized_turn_count": (
            int(memory["summarized_turn_count"]) if memory else 0
        ),
        "turns": [
            {
                "id": int(row["id"]),
                "conversation_title": str(row["conversation_title"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }
            for row in rows
        ],
    }


def save_project_memory_summary(
    *,
    project_id: str,
    summary: str,
    summarized_until_turn_id: int,
    summarized_turn_count: int,
    summary_model: str,
) -> bool:
    now = utc_now()
    with transaction() as connection:
        current = connection.execute(
            """
            SELECT summarized_until_turn_id
            FROM project_memory
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        current_turn_id = (
            int(current["summarized_until_turn_id"]) if current else 0
        )
        if summarized_until_turn_id <= current_turn_id:
            return False
        connection.execute(
            """
            INSERT INTO project_memory(
                project_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                last_error,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                summary = excluded.summary,
                summarized_until_turn_id =
                    excluded.summarized_until_turn_id,
                summarized_turn_count = excluded.summarized_turn_count,
                summary_model = excluded.summary_model,
                last_error = NULL,
                updated_at = excluded.updated_at
            """,
            (
                project_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                now,
            ),
        )
        return True


def set_project_memory_error(project_id: str, error: str) -> None:
    now = utc_now()
    safe_error = " ".join(error.split()).strip()[:500]
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO project_memory(
                project_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                last_error,
                updated_at
            )
            VALUES (?, '', 0, 0, '', ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            """,
            (project_id, safe_error, now),
        )


def _agent_usage_point(
    metrics: dict[str, Any],
    role: str,
) -> dict[str, Any] | None:
    raw = metrics.get(role)
    if not isinstance(raw, dict):
        return None
    if raw.get("status") not in {"completed", "degraded"}:
        return None
    context_window = int(raw.get("context_window_tokens") or 0)
    prompt_tokens = int(raw.get("prompt_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or 0)
    total_tokens = int(
        raw.get("total_tokens")
        or prompt_tokens + completion_tokens
    )
    if context_window <= 0 or total_tokens < 0:
        return None
    return {
        "context_window_tokens": context_window,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "usage_percent": round(
            min(100.0, total_tokens / context_window * 100),
            2,
        ),
        "estimated": bool(raw.get("estimated", False)),
    }


def _project_context_usage(
    metrics_rows: list[sqlite3.Row],
) -> dict[str, Any]:
    role_points: dict[str, list[dict[str, Any]]] = {
        "verifier": [],
        "writer": [],
    }
    latest_metrics: dict[str, Any] = {}
    for row in metrics_rows:
        try:
            metrics = json.loads(row["agent_metrics_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(metrics, dict):
            continue
        latest_metrics = metrics
        for role in role_points:
            point = _agent_usage_point(metrics, role)
            if point is not None:
                role_points[role].append(point)

    output: dict[str, Any] = {}
    latest_percentages: list[float] = []
    maximum_percentages: list[float] = []
    all_percentages: list[float] = []
    for role, points in role_points.items():
        if not points:
            output[role] = {
                "latest": None,
                "maximum_percent": None,
                "average_percent": None,
                "samples": 0,
            }
            continue
        percentages = [
            float(point["usage_percent"]) for point in points
        ]
        latest = _agent_usage_point(latest_metrics, role)
        if latest is not None:
            latest_percentages.append(float(latest["usage_percent"]))
        maximum_percentages.append(max(percentages))
        all_percentages.extend(percentages)
        output[role] = {
            "latest": latest,
            "maximum_percent": round(max(percentages), 2),
            "average_percent": round(
                sum(percentages) / len(percentages),
                2,
            ),
            "samples": len(points),
        }

    output.update(
        {
            "latest_percent": (
                round(max(latest_percentages), 2)
                if latest_percentages
                else None
            ),
            "maximum_percent": (
                round(max(maximum_percentages), 2)
                if maximum_percentages
                else None
            ),
            "average_percent": (
                round(sum(all_percentages) / len(all_percentages), 2)
                if all_percentages
                else None
            ),
        }
    )
    return output


def _validated_project_name(name: str) -> str:
    normalized = " ".join(name.split()).strip()
    if not normalized:
        raise ValueError("El nombre del proyecto no puede estar vacío.")
    if len(normalized) > 80:
        raise ValueError(
            "El nombre del proyecto no puede superar 80 caracteres."
        )
    return normalized


def create_project(name: str) -> dict[str, Any]:
    normalized = _validated_project_name(name)
    project_id = uuid4().hex
    now = utc_now()
    verifier_context_tokens = _validate_context_tokens(
        getattr(
            settings,
            "default_verifier_context_tokens",
            DEFAULT_VERIFIER_CONTEXT_TOKENS,
        ),
        "La ventana del verificador",
    )
    writer_context_tokens = _validate_context_tokens(
        getattr(
            settings,
            "default_writer_context_tokens",
            DEFAULT_WRITER_CONTEXT_TOKENS,
        ),
        "La ventana del redactor",
    )
    try:
        with transaction() as connection:
            connection.execute(
                """
                INSERT INTO projects(
                    id,
                    name,
                    verification_enabled,
                    project_memory_enabled,
                    verifier_context_tokens,
                    writer_context_tokens,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, 1, 1, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    normalized,
                    verifier_context_tokens,
                    writer_context_tokens,
                    now,
                    now,
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"Ya existe un proyecto llamado «{normalized}»."
        ) from exc
    return {
        "id": project_id,
        "name": normalized,
        "created_at": now,
        "updated_at": now,
        "conversation_count": 0,
        "agent_settings": {
            "verification_enabled": True,
            "project_memory_enabled": True,
            "verifier_context_tokens": (
                verifier_context_tokens
            ),
            "writer_context_tokens": writer_context_tokens,
        },
        "context_usage": _project_context_usage([]),
    }


def list_projects() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                p.id,
                p.name,
                p.verification_enabled,
                p.project_memory_enabled,
                p.verifier_context_tokens,
                p.writer_context_tokens,
                p.created_at,
                p.updated_at,
                COUNT(c.id) AS conversation_count
            FROM projects AS p
            LEFT JOIN conversations AS c ON c.project_id = p.id
            GROUP BY p.id
            ORDER BY
                CASE WHEN p.id = ? THEN 0 ELSE 1 END,
                p.name COLLATE NOCASE
            """,
            (DEFAULT_PROJECT_ID,),
        ).fetchall()
        metric_rows = connection.execute(
            """
            SELECT c.project_id, t.agent_metrics_json
            FROM conversation_turns AS t
            JOIN conversations AS c ON c.id = t.conversation_id
            ORDER BY t.id
            """
        ).fetchall()
        memory_rows = connection.execute(
            """
            SELECT
                p.id AS project_id,
                COUNT(t.id) AS total_turns,
                COALESCE(m.summary, '') AS summary,
                COALESCE(m.summarized_turn_count, 0)
                    AS summarized_turn_count,
                m.last_error
            FROM projects AS p
            LEFT JOIN conversations AS c ON c.project_id = p.id
            LEFT JOIN conversation_turns AS t
                ON t.conversation_id = c.id
            LEFT JOIN project_memory AS m ON m.project_id = p.id
            GROUP BY p.id
            """
        ).fetchall()
    metrics_by_project: dict[str, list[sqlite3.Row]] = {}
    for metric_row in metric_rows:
        metrics_by_project.setdefault(
            str(metric_row["project_id"]),
            [],
        ).append(metric_row)
    memory_by_project = {
        str(row["project_id"]): {
            "enabled": True,
            "has_summary": bool(str(row["summary"]).strip()),
            "summarized_turns": int(row["summarized_turn_count"]),
            "pending_turns": max(
                0,
                int(row["total_turns"])
                - int(row["summarized_turn_count"]),
            ),
            "total_turns": int(row["total_turns"]),
            "last_error": (
                str(row["last_error"]) if row["last_error"] else None
            ),
        }
        for row in memory_rows
    }
    projects = []
    for row in rows:
        project_id = str(row["id"])
        project_memory = memory_by_project.get(project_id, {})
        project_memory["enabled"] = bool(
            row["project_memory_enabled"]
        )
        projects.append(
            {
            "id": str(row["id"]),
            "name": str(row["name"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "conversation_count": int(row["conversation_count"]),
            "agent_settings": _project_agent_settings_from_row(row),
            "context_usage": _project_context_usage(
                metrics_by_project.get(project_id, [])
            ),
            "memory": project_memory,
        }
        )
    return projects


def rename_project(project_id: str, name: str) -> dict[str, Any] | None:
    normalized = _validated_project_name(name)
    now = utc_now()
    try:
        with transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE projects
                SET name = ?, updated_at = ?
                WHERE id = ?
                """,
                (normalized, now, project_id),
            )
            if cursor.rowcount == 0:
                return None
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM conversations
                    WHERE project_id = ?
                    """,
                    (project_id,),
                ).fetchone()["total"]
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"Ya existe un proyecto llamado «{normalized}»."
        ) from exc
    return {
        "id": project_id,
        "name": normalized,
        "updated_at": now,
        "conversation_count": count,
    }


def delete_project(project_id: str) -> dict[str, Any] | None:
    if project_id == DEFAULT_PROJECT_ID:
        raise ValueError("El proyecto General no se puede eliminar.")

    now = utc_now()
    with transaction() as connection:
        row = connection.execute(
            "SELECT id, name FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            return None

        conversation_count = int(
            connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM conversations
                WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()["total"]
        )
        connection.execute(
            """
            UPDATE conversations
            SET project_id = ?, updated_at = ?
            WHERE project_id = ?
            """,
            (DEFAULT_PROJECT_ID, now, project_id),
        )
        connection.execute(
            """
            DELETE FROM project_memory
            WHERE project_id = ?
            """,
            (project_id,),
        )
        cursor = connection.execute(
            "DELETE FROM projects WHERE id = ?",
            (project_id,),
        )
        connection.execute(
            """
            UPDATE projects
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, DEFAULT_PROJECT_ID),
        )

    return {
        "deleted": cursor.rowcount > 0,
        "project_id": project_id,
        "project_name": str(row["name"]),
        "moved_conversations": conversation_count,
        "fallback_project_id": DEFAULT_PROJECT_ID,
    }


def save_conversation_turn(
    *,
    conversation_id: str | None,
    project_id: str | None = None,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    chat_model: str,
    memory_enabled: bool = True,
    agent_metrics: dict[str, Any] | None = None,
) -> tuple[str, int]:
    now = utc_now()
    with transaction() as connection:
        current_turns = 0
        if conversation_id:
            row = connection.execute(
                "SELECT id, project_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(
                    f"No existe la conversación: {conversation_id}"
                )
            current_turns = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM conversation_turns
                    WHERE conversation_id = ?
                    """,
                    (conversation_id,),
                ).fetchone()["total"]
            )
        else:
            project_id = project_id or DEFAULT_PROJECT_ID
            project = connection.execute(
                "SELECT id FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if project is None:
                raise KeyError(f"No existe el proyecto: {project_id}")
            conversation_id = uuid4().hex
            connection.execute(
                """
                INSERT INTO conversations(
                    id, project_id, title, memory_enabled,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    project_id,
                    conversation_title(question),
                    int(memory_enabled),
                    now,
                    now,
                ),
            )

        cursor = connection.execute(
            """
            INSERT INTO conversation_turns(
                conversation_id, question, answer,
                sources_json, chat_model, agent_metrics_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                question,
                answer,
                json.dumps(sources, ensure_ascii=False, default=str),
                chat_model,
                json.dumps(
                    agent_metrics or {},
                    ensure_ascii=False,
                    default=str,
                ),
                now,
            ),
        )
        if current_turns == 0:
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (conversation_title(question), now, conversation_id),
            )
        else:
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (now, conversation_id),
            )
        connection.execute(
            """
            UPDATE conversations
            SET memory_enabled = ?
            WHERE id = ?
            """,
            (int(memory_enabled), conversation_id),
        )
        connection.execute(
            """
            UPDATE projects
            SET updated_at = ?
            WHERE id = (
                SELECT project_id
                FROM conversations
                WHERE id = ?
            )
            """,
            (now, conversation_id),
        )
        return conversation_id, int(cursor.lastrowid)


def list_conversations(
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.id,
                c.project_id,
                c.title,
                c.memory_enabled,
                c.created_at,
                c.updated_at,
                COUNT(t.id) AS turn_count
            FROM conversations AS c
            LEFT JOIN conversation_turns AS t
                ON t.conversation_id = c.id
            WHERE (? IS NULL OR c.project_id = ?)
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """,
            (project_id, project_id),
        ).fetchall()
    return [
        {
            "id": str(row["id"]),
            "project_id": str(row["project_id"]),
            "title": str(row["title"]),
            "memory_enabled": bool(row["memory_enabled"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "turn_count": int(row["turn_count"]),
        }
        for row in rows
    ]


_AGENT_CONTEXT_FIELDS = (
    "model",
    "model_context_window",
    "configured_context_window",
    "effective_context_window",
    "context_source",
)


def _normalize_agent_role_metrics(raw: Any) -> dict[str, Any] | None:
    """Decora una entrada verifier/writer con una marca de completitud.

    No reescribe nada en la base de datos: un turno guardado antes de que
    existieran estos campos (o guardado con un modelo distinto al activo
    ahora) sigue mostrando exactamente lo que se guardó, sólo que
    marcado como incompleto en vez de mostrar huecos silenciosos.
    """
    if not isinstance(raw, dict):
        return None
    normalized = dict(raw)
    normalized["incomplete_context_data"] = any(
        normalized.get(field) is None for field in _AGENT_CONTEXT_FIELDS
    )
    return normalized


def _normalize_agent_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    normalized = dict(metrics)
    for role in ("verifier", "writer"):
        role_metrics = _normalize_agent_role_metrics(metrics.get(role))
        if role_metrics is not None:
            normalized[role] = role_metrics
    return normalized


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        conversation = connection.execute(
            """
            SELECT
                id, project_id, title, memory_enabled,
                created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None
        rows = connection.execute(
            """
            SELECT
                id, question, answer, sources_json,
                chat_model, agent_metrics_json, created_at
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()

    turns: list[dict[str, Any]] = []
    for row in rows:
        try:
            sources = json.loads(row["sources_json"] or "[]")
        except json.JSONDecodeError:
            sources = []
        try:
            agent_metrics = json.loads(
                row["agent_metrics_json"] or "{}"
            )
        except json.JSONDecodeError:
            agent_metrics = {}
        turns.append(
            {
                "id": int(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "sources": sources if isinstance(sources, list) else [],
                "chat_model": str(row["chat_model"]),
                "agent_metrics": _normalize_agent_metrics(agent_metrics),
                "created_at": str(row["created_at"]),
            }
        )
    memory = get_conversation_memory_status(conversation_id)
    return {
        "id": str(conversation["id"]),
        "project_id": str(conversation["project_id"]),
        "title": str(conversation["title"]),
        "memory": memory,
        "created_at": str(conversation["created_at"]),
        "updated_at": str(conversation["updated_at"]),
        "turns": turns,
    }


def delete_conversation(conversation_id: str) -> bool:
    with transaction() as connection:
        row = connection.execute(
            "SELECT project_id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return False
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM project_memory WHERE project_id = ?",
            (str(row["project_id"]),),
        )
        return cursor.rowcount > 0

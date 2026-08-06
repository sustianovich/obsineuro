"""Índice FTS5: reconstrucción, sincronización y búsqueda léxica."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Sequence

from app.db.schema import (
    FTS_ERROR_META_KEY,
    FTS_INDEX_VERSION,
    FTS_VERSION_META_KEY,
    _document_status,
    _fts_table_exists,
    _set_meta_with_connection,
    get_connection,
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

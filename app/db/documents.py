"""Ciclo de vida del índice de documentos: preparación (staged) y activo."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Sequence

import numpy as np

from app.db.fts import _delete_fts_chunks, _insert_fts_chunk, _rebuild_fts_index
from app.db.links import (
    _lightweight_alias_index,
    _rebuild_document_links,
    _reinsert_captured_links,
    _replace_outgoing_document_links,
    _reresolve_broken_links,
    _resolved_link_values,
)
from app.db.schema import (
    GRAPH_INDEX_VERSION,
    GRAPH_VERSION_META_KEY,
    _document_status,
    _fts_table_exists,
    _set_meta_with_connection,
    get_connection,
    transaction,
    utc_now,
)


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

"""Grafo de wikienlaces: resolución de alias, backlinks y su estado."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Sequence

from app.db.schema import (
    GRAPH_INDEX_VERSION,
    GRAPH_VERSION_META_KEY,
    _set_meta_with_connection,
    get_connection,
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

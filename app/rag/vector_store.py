"""Almacén vectorial en memoria con invalidación automática.

Sustituye al patrón anterior (`load_all_chunks()` en cada consulta), que
leía de SQLite todos los fragmentos con su contenido, sus metadatos JSON y
sus vectores, y recorría la lista fragmento a fragmento en Python.

Aquí se mantiene una única matriz `float32` normalizada en memoria y la
búsqueda semántica se resuelve con un producto matricial de NumPy. El
contenido y los metadatos sólo se leen de SQLite para los candidatos
finales, no para todo el corpus.

La caché se invalida sola: en cada consulta se comprueba una firma barata
(número de fragmentos, mayor identificador y huella del índice). Tras
reindexar, la primera consulta recarga la matriz sin intervención.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterable

import numpy as np

from app.db import get_connection
from app.rag.document_meta import document_tags, document_vigencia


INDEX_FINGERPRINT_KEY = "index_fingerprint"


@dataclass(frozen=True)
class IndexSignature:
    chunk_count: int
    max_chunk_id: int
    fingerprint: str

    @classmethod
    def empty(cls) -> "IndexSignature":
        return cls(chunk_count=-1, max_chunk_id=-1, fingerprint="")


def _read_signature(connection: sqlite3.Connection) -> IndexSignature:
    row = connection.execute(
        "SELECT COUNT(*) AS total, COALESCE(MAX(id), 0) AS top FROM chunks"
    ).fetchone()
    fingerprint_row = connection.execute(
        "SELECT value FROM app_meta WHERE key = ?",
        (INDEX_FINGERPRINT_KEY,),
    ).fetchone()
    return IndexSignature(
        chunk_count=int(row["total"]),
        max_chunk_id=int(row["top"]),
        fingerprint=(
            str(fingerprint_row["value"]) if fingerprint_row else ""
        ),
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normaliza L2 por filas; las filas nulas quedan a cero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32, copy=False)


class VectorStore:
    """Matriz de embeddings compartida, segura entre hilos."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._signature = IndexSignature.empty()
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._chunk_ids = np.zeros(0, dtype=np.int64)
        self._document_ids = np.zeros(0, dtype=np.int64)
        self._statuses: list[str] = []
        self._vigencias: list[str] = []
        self._tags: list[frozenset[str]] = []
        self._tag_document_counts: dict[str, int] = {}
        self._status_masks: dict[str, np.ndarray] = {}
        self._vigencia_masks: dict[str, np.ndarray] = {}
        self._tag_masks: dict[str, np.ndarray] = {}
        self._row_by_chunk_id: dict[int, int] = {}
        self._dimension = 0

    # ------------------------------------------------------------------
    # Carga y sincronización
    # ------------------------------------------------------------------
    def invalidate(self) -> None:
        """Fuerza la recarga en la siguiente consulta (tras indexar)."""
        with self._lock:
            self._signature = IndexSignature.empty()

    def _load(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.embedding,
                c.embedding_dim,
                d.id AS document_id,
                d.metadata_json
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.id
            """
        ).fetchall()

        if not rows:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._chunk_ids = np.zeros(0, dtype=np.int64)
            self._document_ids = np.zeros(0, dtype=np.int64)
            self._statuses = []
            self._vigencias = []
            self._tags = []
            self._tag_document_counts = {}
            self._status_masks = {}
            self._vigencia_masks = {}
            self._tag_masks = {}
            self._row_by_chunk_id = {}
            self._dimension = 0
            return

        dimensions = {int(row["embedding_dim"]) for row in rows}
        # Un índice sano tiene una única dimensión. Si conviven varias
        # (reindexado a medias), se conserva la mayoritaria y se descarta
        # el resto en lugar de fallar durante una demostración.
        if len(dimensions) == 1:
            dimension = dimensions.pop()
            usable = rows
        else:
            counts: dict[int, int] = {}
            for row in rows:
                key = int(row["embedding_dim"])
                counts[key] = counts.get(key, 0) + 1
            dimension = max(counts, key=lambda key: counts[key])
            usable = [
                row for row in rows if int(row["embedding_dim"]) == dimension
            ]

        matrix = np.empty((len(usable), dimension), dtype=np.float32)
        chunk_ids = np.empty(len(usable), dtype=np.int64)
        document_ids = np.empty(len(usable), dtype=np.int64)
        statuses: list[str] = []
        vigencias: list[str] = []
        tags: list[frozenset[str]] = []

        valid = 0
        for row in usable:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size != dimension:
                continue
            matrix[valid] = vector
            chunk_ids[valid] = int(row["chunk_id"])
            document_ids[valid] = int(row["document_id"])
            metadata = _parse_metadata(row["metadata_json"])
            statuses.append(_status_from(metadata))
            vigencias.append(document_vigencia(metadata))
            tags.append(frozenset(document_tags(metadata)))
            valid += 1

        self._matrix = np.ascontiguousarray(
            _normalize_rows(matrix[:valid])
        )
        self._chunk_ids = chunk_ids[:valid]
        self._document_ids = document_ids[:valid]
        self._statuses = statuses
        self._vigencias = vigencias
        self._tags = tags
        tag_documents: dict[str, set[int]] = {}
        for document_id, group in zip(self._document_ids, self._tags):
            for tag in group:
                tag_documents.setdefault(tag, set()).add(int(document_id))
        self._tag_document_counts = {
            tag: len(document_ids)
            for tag, document_ids in tag_documents.items()
        }
        self._status_masks = {}
        self._vigencia_masks = {}
        self._tag_masks = {}
        self._row_by_chunk_id = {
            int(chunk_id): row for row, chunk_id in enumerate(self._chunk_ids)
        }
        self._dimension = dimension

    def ensure_loaded(self) -> None:
        with self._lock:
            with get_connection() as connection:
                signature = _read_signature(connection)
                if signature == self._signature and self._matrix.size:
                    return
                if (
                    signature == self._signature
                    and signature.chunk_count == 0
                ):
                    return
                self._load(connection)
                self._signature = signature

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    def _status_mask(self, status: str) -> np.ndarray:
        cached = self._status_masks.get(status)
        if cached is None:
            cached = np.fromiter(
                (value == status for value in self._statuses),
                dtype=bool,
                count=len(self._statuses),
            )
            self._status_masks[status] = cached
        return cached

    def _vigencia_mask(self, vigencia: str) -> np.ndarray:
        cached = self._vigencia_masks.get(vigencia)
        if cached is None:
            if vigencia == "no_caducada":
                # Lo útil por defecto en un corpus normativo: todo salvo
                # lo que ya dejó de regir.
                predicate = [value != "caducada" for value in self._vigencias]
            else:
                predicate = [value == vigencia for value in self._vigencias]
            cached = np.fromiter(
                predicate, dtype=bool, count=len(self._vigencias)
            )
            self._vigencia_masks[vigencia] = cached
        return cached

    def _tag_mask(self, tag: str) -> np.ndarray:
        cached = self._tag_masks.get(tag)
        if cached is None:
            cached = np.fromiter(
                (tag in value for value in self._tags),
                dtype=bool,
                count=len(self._tags),
            )
            self._tag_masks[tag] = cached
        return cached

    def search(
        self,
        query_vector: np.ndarray,
        *,
        limit: int,
        min_similarity: float,
        status: str | None = None,
        vigencia: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Devuelve los `limit` fragmentos más próximos por coseno."""
        self.ensure_loaded()
        with self._lock:
            if not self._matrix.size or limit <= 0:
                return []

            query = np.asarray(query_vector, dtype=np.float32).ravel()
            if query.size != self._dimension:
                # Índice y consulta con modelos distintos: abstenerse en
                # lugar de devolver ruido.
                return []
            norm = float(np.linalg.norm(query))
            if norm == 0.0:
                return []
            query = query / norm

            scores = self._matrix @ query

            mask = scores >= min_similarity
            if status:
                mask = mask & self._status_mask(status)
            if vigencia:
                mask = mask & self._vigencia_mask(vigencia)
            for tag in tags or []:
                mask = mask & self._tag_mask(tag)
            rows = np.flatnonzero(mask)

            if rows.size == 0:
                return []

            take = min(limit, rows.size)
            candidate_scores = scores[rows]
            # argpartition evita ordenar todo el corpus.
            partitioned = np.argpartition(-candidate_scores, take - 1)[:take]
            ordered = partitioned[np.argsort(-candidate_scores[partitioned])]
            selected_rows = rows[ordered]

            return [
                {
                    "chunk_id": int(self._chunk_ids[row]),
                    "document_id": int(self._document_ids[row]),
                    "row": int(row),
                    "semantic_score": float(scores[row]),
                }
                for row in selected_rows
            ]

    def vectors_for_rows(self, rows: Iterable[int]) -> np.ndarray:
        with self._lock:
            index = np.fromiter(rows, dtype=np.int64)
            if index.size == 0 or not self._matrix.size:
                return np.zeros((0, self._dimension or 1), dtype=np.float32)
            return self._matrix[index]

    def best_chunks_for_documents(
        self,
        query_vector: np.ndarray,
        document_ids: Iterable[int],
    ) -> dict[int, dict[str, Any]]:
        """Elige por coseno el mejor fragmento de cada documento.

        La propagación trabaja a nivel de nota, pero el redactor consume
        fragmentos. Resolver aquí la segunda etapa permite puntuar todos sus
        fragmentos con la matriz ya residente, sin lecturas ni bucles de
        contenido en Python.
        """
        self.ensure_loaded()
        wanted = np.fromiter(
            (int(value) for value in document_ids),
            dtype=np.int64,
        )
        with self._lock:
            if wanted.size == 0 or not self._matrix.size:
                return {}
            query = np.asarray(query_vector, dtype=np.float32).ravel()
            if query.size != self._dimension:
                return {}
            norm = float(np.linalg.norm(query))
            if norm == 0.0:
                return {}
            scores = self._matrix @ (query / norm)
            rows = np.flatnonzero(np.isin(self._document_ids, wanted))

            best_rows: dict[int, int] = {}
            for row in rows:
                document_id = int(self._document_ids[row])
                previous = best_rows.get(document_id)
                if previous is None or scores[row] > scores[previous]:
                    best_rows[document_id] = int(row)

            return {
                document_id: {
                    "chunk_id": int(self._chunk_ids[row]),
                    "document_id": document_id,
                    "row": row,
                    "semantic_score": float(scores[row]),
                }
                for document_id, row in best_rows.items()
            }

    def row_for_chunk(self, chunk_id: int) -> int | None:
        with self._lock:
            return self._row_by_chunk_id.get(int(chunk_id))

    def stats(self) -> dict[str, Any]:
        # Fuerza la carga: si no, /api/status informa de un vocabulario
        # vacío justo después de indexar y antes de la primera consulta.
        self.ensure_loaded()
        with self._lock:
            bytes_used = int(self._matrix.nbytes)
            return {
                "loaded_chunks": int(self._matrix.shape[0]),
                "dimension": self._dimension,
                "memory_mb": round(bytes_used / (1024 * 1024), 2),
                "fingerprint_known": bool(self._signature.fingerprint),
                "vigencias": sorted(set(self._vigencias)) if self._vigencias else [],
                "tags": sorted({tag for group in self._tags for tag in group}),
                "tag_counts": [
                    {"tag": tag, "documents": documents}
                    for tag, documents in sorted(self._tag_document_counts.items())
                ],
            }


def _parse_metadata(metadata_json: Any) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_json or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _status_from(metadata: dict[str, Any]) -> str:
    value = metadata.get("estado", metadata.get("status", ""))
    return str(value).strip().casefold()


vector_store = VectorStore()


def hydrate_chunks(chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Lee contenido y metadatos sólo de los candidatos seleccionados."""
    if not chunk_ids:
        return {}
    unique_ids = list(dict.fromkeys(int(value) for value in chunk_ids))
    placeholders = ",".join("?" * len(unique_ids))
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                c.chunk_index,
                c.heading,
                c.content,
                c.parent_id,
                c.embedding_dim,
                parent.parent_index,
                parent.content AS parent_content,
                d.id AS document_id,
                d.path,
                d.title,
                d.metadata_json,
                d.links_json
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN chunk_parents AS parent ON parent.id = c.parent_id
            WHERE c.id IN ({placeholders})
            """,
            unique_ids,
        ).fetchall()

    return {
        int(row["chunk_id"]): {
            "chunk_id": int(row["chunk_id"]),
            "chunk_index": int(row["chunk_index"]),
            "heading": str(row["heading"]),
            "content": str(row["parent_content"] or row["content"]),
            "matched_content": (
                str(row["content"])
                if row["parent_id"] is not None
                else None
            ),
            "parent_id": (
                int(row["parent_id"])
                if row["parent_id"] is not None
                else None
            ),
            "parent_index": (
                int(row["parent_index"])
                if row["parent_index"] is not None
                else None
            ),
            "context_expanded": row["parent_id"] is not None,
            "embedding_dim": int(row["embedding_dim"]),
            "document_id": int(row["document_id"]),
            "path": str(row["path"]),
            "title": str(row["title"]),
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "links": json.loads(row["links_json"] or "[]"),
        }
        for row in rows
    }


def document_chunks(document_id: int) -> list[dict[str, Any]]:
    """Fragmentos de un documento, para la expansión de enlaces."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.chunk_index,
                c.heading,
                c.content,
                c.parent_id,
                parent.parent_index,
                parent.content AS parent_content,
                d.id AS document_id,
                d.path,
                d.title,
                d.metadata_json,
                d.links_json
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN chunk_parents AS parent ON parent.id = c.parent_id
            WHERE d.id = ?
            ORDER BY c.chunk_index
            """,
            (int(document_id),),
        ).fetchall()
    return [
        {
            "chunk_id": int(row["chunk_id"]),
            "chunk_index": int(row["chunk_index"]),
            "heading": str(row["heading"]),
            "content": str(row["parent_content"] or row["content"]),
            "matched_content": (
                str(row["content"])
                if row["parent_id"] is not None
                else None
            ),
            "parent_id": (
                int(row["parent_id"])
                if row["parent_id"] is not None
                else None
            ),
            "parent_index": (
                int(row["parent_index"])
                if row["parent_index"] is not None
                else None
            ),
            "context_expanded": row["parent_id"] is not None,
            "document_id": int(row["document_id"]),
            "path": str(row["path"]),
            "title": str(row["title"]),
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "links": json.loads(row["links_json"] or "[]"),
        }
        for row in rows
    ]


def document_directory() -> list[dict[str, Any]]:
    """Índice ligero de documentos (título y ruta) para resolver enlaces."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, path, title FROM documents"
        ).fetchall()
    return [
        {
            "document_id": int(row["id"]),
            "path": str(row["path"]),
            "title": str(row["title"]),
        }
        for row in rows
    ]

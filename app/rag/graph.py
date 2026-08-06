"""Resolución de wikienlaces y recuperación por propagación en el grafo.

El grafo complementa la evidencia semántica: no genera semillas propias.
Esto conserva la abstención del recuperador y evita que la mera cercanía
estructural se confunda con relevancia para la pregunta.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import sqlite3
from threading import RLock
from typing import Any, Iterable, Sequence

import numpy as np

from app.config import settings


def normalize(value: str) -> str:
    return value.replace("\\", "/").strip().casefold()


def document_aliases(document: dict[str, Any]) -> set[str]:
    """Devuelve las formas que Obsidian puede usar para nombrar una nota."""
    path = normalize(str(document["path"]))
    without_suffix = path[:-3] if path.endswith(".md") else path
    stem = without_suffix.rsplit("/", 1)[-1]
    return {
        normalize(str(document["title"])),
        without_suffix,
        stem,
    }


def _index_aliases(
    documents: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    aliases: dict[str, Any] = {}
    claimants: dict[str, set[Any]] = defaultdict(set)
    for document in documents:
        identity = document.get("document_id", document.get("path"))
        for alias in document_aliases(document):
            aliases.setdefault(alias, identity)
            claimants[alias].add(identity)
    ambiguous = sum(1 for identities in claimants.values() if len(identities) > 1)
    return aliases, ambiguous


def build_alias_index(
    documents: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Construye un índice estable y conserva la primera colisión.

    Es la misma regla que empleaba la expansión de wikienlaces. Centralizarla
    impide que una nota se resuelva a destinos distintos durante la
    indexación y durante una consulta.
    """
    aliases, _ = _index_aliases(documents)
    return aliases


def count_ambiguous_aliases(documents: Iterable[dict[str, Any]]) -> int:
    """Cuenta alias reclamados por más de una nota (p.ej. dos "Index.md").

    `build_alias_index` conserva silenciosamente a la primera nota que
    reclama cada alias; esto expone cuántos alias tuvieron más de un
    reclamante, para distinguir en el estado del grafo un enlace ambiguo
    de uno simplemente roto.
    """
    _, ambiguous = _index_aliases(documents)
    return ambiguous


def target_aliases(target: str) -> tuple[str, ...]:
    normalized = normalize(target)
    without_suffix = (
        normalized[:-3] if normalized.endswith(".md") else normalized
    )
    stem = without_suffix.rsplit("/", 1)[-1]
    return tuple(dict.fromkeys((normalized, without_suffix, stem)))


def resolve_document_target(
    aliases: dict[str, Any],
    target: str,
) -> Any | None:
    for candidate in target_aliases(target):
        resolved = aliases.get(candidate)
        if resolved is not None:
            return resolved
    return None


def _propagate_with_hops(
    seed_weights: dict[int, float],
    edges: Sequence[tuple[int, int]],
    *,
    max_hops: int,
    decay: float,
    backlink_weight: float,
) -> tuple[dict[int, float], dict[int, int]]:
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for source, target in edges:
        source_id = int(source)
        target_id = int(target)
        # Recorrer el enlace escrito es una salida; volver por él es un
        # backlink y se penaliza para no sobrevalorar notas muy citadas.
        adjacency[source_id].append((target_id, 1.0))
        adjacency[target_id].append((source_id, backlink_weight))

    seeds = {int(key): float(value) for key, value in seed_weights.items()}
    best_seen = dict(seeds)
    reached: dict[int, float] = {}
    reached_hops: dict[int, int] = {}
    frontier = dict(seeds)

    for hop in range(1, max_hops + 1):
        next_frontier: dict[int, float] = {}
        for document_id, current_weight in frontier.items():
            for neighbor, direction_weight in adjacency.get(document_id, []):
                propagated = current_weight * decay * direction_weight
                if propagated <= best_seen.get(neighbor, -1.0):
                    continue
                best_seen[neighbor] = propagated
                next_frontier[neighbor] = max(
                    propagated,
                    next_frontier.get(neighbor, -1.0),
                )
                if neighbor not in seeds:
                    reached[neighbor] = propagated
                    reached_hops[neighbor] = hop
        frontier = next_frontier
        if not frontier:
            break

    return reached, reached_hops


def propagate_document_scores(
    seed_weights: dict[int, float],
    edges: Sequence[tuple[int, int]],
    *,
    max_hops: int,
    decay: float,
    backlink_weight: float,
) -> dict[int, float]:
    """Propaga el máximo por documento, nunca la suma de caminos."""
    scores, _ = _propagate_with_hops(
        seed_weights,
        edges,
        max_hops=max_hops,
        decay=decay,
        backlink_weight=backlink_weight,
    )
    return scores


def _semantic_document_seeds(
    semantic: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[int, float]:
    seeds: dict[int, float] = {}
    for item in semantic:
        document_id = int(item["document_id"])
        if document_id in seeds:
            continue
        rank = len(seeds) + 1
        seeds[document_id] = 1.0 / rank
        if len(seeds) >= limit:
            break
    return seeds


@dataclass(frozen=True)
class _EdgeSignature:
    """Firma barata para detectar cambios sin releer las aristas.

    `COUNT` y `MAX(rowid)` bastan para altas y bajas, pero una resolución
    de enlace roto (`UPDATE ... SET target_document_id`) no cambia ni el
    número de filas ni el rowid máximo. La suma de destinos sí se mueve en
    ese caso, así que la firma la incluye para no servir una arista recién
    resuelta como si siguiera rota.
    """

    edge_count: int
    max_rowid: int
    target_checksum: int

    @classmethod
    def empty(cls) -> "_EdgeSignature":
        return cls(edge_count=-1, max_rowid=-1, target_checksum=-1)


def _read_edge_signature(connection: sqlite3.Connection) -> _EdgeSignature:
    row = connection.execute(
        """
        SELECT
            COUNT(*) AS total,
            COALESCE(MAX(rowid), 0) AS top,
            COALESCE(SUM(target_document_id), 0) AS target_checksum
        FROM document_links
        """
    ).fetchone()
    return _EdgeSignature(
        edge_count=int(row["total"]),
        max_rowid=int(row["top"]),
        target_checksum=int(row["target_checksum"]),
    )


def _load_edges(connection: sqlite3.Connection) -> list[tuple[int, int]]:
    rows = connection.execute(
        """
        SELECT source_document_id, target_document_id
        FROM document_links
        WHERE target_document_id IS NOT NULL
        """
    ).fetchall()
    return [
        (int(row["source_document_id"]), int(row["target_document_id"]))
        for row in rows
    ]


class GraphEdgeStore:
    """Aristas resueltas en memoria, con el mismo patrón que `vector_store`
    aplica a la matriz de embeddings: se evita repetir una consulta SQL
    completa en cada búsqueda mientras el grafo no haya cambiado."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._signature = _EdgeSignature.empty()
        self._edges: list[tuple[int, int]] = []

    def invalidate(self) -> None:
        """Fuerza la recarga en la siguiente consulta (tras indexar)."""
        with self._lock:
            self._signature = _EdgeSignature.empty()

    def edges(self) -> list[tuple[int, int]]:
        from app.db import get_connection

        with self._lock:
            try:
                with get_connection() as connection:
                    signature = _read_edge_signature(connection)
                    if signature != self._signature:
                        self._edges = _load_edges(connection)
                        self._signature = signature
            except sqlite3.Error:
                return []
            return self._edges


graph_store = GraphEdgeStore()


def _resolved_edges() -> list[tuple[int, int]]:
    return graph_store.edges()


def graph_candidates(
    semantic: list[dict[str, Any]],
    query_vector: np.ndarray,
    *,
    use_graph: bool | None = None,
    seed_documents: int | None = None,
    max_hops: int | None = None,
    decay: float | None = None,
    backlink_weight: float | None = None,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Selecciona un único fragmento semánticamente óptimo por nota alcanzada.

    Los parámetros nombrados son opcionales para no romper llamadas
    existentes: si se omiten, se resuelven contra `settings` igual que
    antes. Quien enrute la consulta (`app.rag.query_routing`) puede pasar
    una política explícita sin tocar `settings`, lo que le permite activar
    el grafo para una consulta relacional aunque `RAG_GRAPH_SEARCH` esté
    desactivado globalmente.
    """
    enabled = settings.graph_search_enabled if use_graph is None else use_graph
    if not enabled or not semantic:
        return []

    seeds = _semantic_document_seeds(
        semantic,
        limit=(
            settings.graph_seed_documents
            if seed_documents is None
            else seed_documents
        ),
    )
    scores, hops = _propagate_with_hops(
        seeds,
        _resolved_edges(),
        max_hops=(
            settings.graph_max_hops if max_hops is None else max_hops
        ),
        decay=settings.graph_decay if decay is None else decay,
        backlink_weight=(
            settings.graph_backlink_weight
            if backlink_weight is None
            else backlink_weight
        ),
    )
    if not scores:
        return []

    from app.rag.vector_store import vector_store

    best_chunks = vector_store.best_chunks_for_documents(
        query_vector,
        scores.keys(),
    )
    candidates = []
    for document_id, graph_score in scores.items():
        best = best_chunks.get(document_id)
        if best is None:
            continue
        candidates.append(
            {
                "chunk_id": best["chunk_id"],
                "document_id": best["document_id"],
                "row": best["row"],
                "graph_chunk_similarity": best["semantic_score"],
                "graph_score": float(graph_score),
                "graph_hop": int(hops[document_id]),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["graph_score"],
            item["graph_chunk_similarity"],
        ),
        reverse=True,
    )
    effective_max_candidates = (
        settings.graph_max_candidates
        if max_candidates is None
        else max_candidates
    )
    return candidates[:effective_max_candidates]


def get_graph_status() -> dict[str, Any]:
    from app.db import get_document_graph_status

    status = get_document_graph_status()
    active = bool(settings.graph_search_enabled and status["available"])
    return {
        **status,
        "configured": settings.graph_search_enabled,
        "active": active,
        "weight": settings.hybrid_graph_weight,
        "max_hops": settings.graph_max_hops,
        "decay": settings.graph_decay,
        "backlink_weight": settings.graph_backlink_weight,
        "seed_documents": settings.graph_seed_documents,
        "max_candidates": settings.graph_max_candidates,
        "wiki_link_expansion_enabled": not settings.graph_search_enabled,
    }

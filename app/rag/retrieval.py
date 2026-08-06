"""Recuperación híbrida: semántica, FTS5 y grafo, con diversificación.

Cambios respecto a la versión anterior:

1. La búsqueda semántica ya no recorre el corpus en Python. Se apoya en
   `vector_store`, que mantiene una matriz normalizada en memoria y
   resuelve la similitud con un único producto matricial.
2. La consulta se codifica con el prefijo de tarea que corresponde al
   modelo de embedding (`search_query:` en nomic). El indexador usa el
   prefijo de documento. Sin esta simetría la recuperación pierde
   precisión de forma medible.
3. Tras la fusión RRF se aplica MMR y un tope de fragmentos por
   documento, para que el contexto no lo monopolice una nota larga.
4. El contenido y los metadatos sólo se leen de SQLite para los
   candidatos finales.
5. El grafo de wikienlaces es una tercera rama opcional sembrada por la
   semántica; por sí solo nunca puede eludir la abstención.

Se conserva intacta la regla de abstención: si ningún fragmento supera el
umbral semántico, la recuperación devuelve una lista vacía y la capa
superior no consulta al modelo.
"""

from __future__ import annotations

import math
import re
import sqlite3
from typing import Any
import unicodedata

import numpy as np

from app.config import settings
from app.db import get_fts5_status, search_chunks_fts
from app.rag.abstention import evaluate_posterior_abstention
from app.rag.document_meta import (
    document_tags,
    document_vigencia,
    vigencia_details,
)
from app.rag.embedding_tasks import resolve_scheme
from app.rag.graph import (
    build_alias_index,
    get_graph_status,
    graph_candidates,
    graph_store,
    resolve_document_target,
)
from app.rag.ollama_client import create_query_embedding
from app.rag.query_routing import RetrievalPolicy, route_query
from app.rag.reranking import get_rerank_status, rerank
from app.rag.vector_store import (
    document_chunks,
    document_directory,
    hydrate_chunks,
    vector_store,
)


SPANISH_STOPWORDS = {
    "a", "al", "algo", "ante", "como", "con", "cual", "cuando", "de",
    "del", "desde", "donde", "el", "ella", "en", "entre", "es", "esta",
    "este", "esto", "hay", "la", "las", "lo", "los", "mas", "me", "para",
    "pero", "por", "porque", "que", "se", "segun", "ser", "si", "sin",
    "sobre", "son", "su", "sus", "un", "una", "y", "ya",
}

MAX_LINKED_ADDITIONS = 3

# Diagnóstico del último reordenado, para exponerlo en la respuesta.
last_rerank_outcome: dict[str, Any] = {}

# Diagnóstico de la última consulta enrutada y de la última decisión de
# abstención posterior. Son sólo lectura para la API de estado: no afectan
# a la política de la siguiente consulta, que siempre se recalcula desde
# cero (ver `app.rag.query_routing`).
last_routing_outcome: dict[str, Any] = {}
last_abstention_outcome: dict[str, Any] = {}


# ----------------------------------------------------------------------
# Utilidades léxicas
# ----------------------------------------------------------------------
def normalize(value: str) -> str:
    return value.replace("\\", "/").strip().casefold()


def metadata_status(metadata: dict[str, Any]) -> str:
    value = metadata.get("estado", metadata.get("status", ""))
    return str(value).strip().casefold()


def normalize_lexical(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )


def extract_lexical_terms(value: str, *, limit: int = 32) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in re.findall(r"\w+", value, flags=re.UNICODE):
        term = raw_term.strip("_").casefold()
        normalized = normalize_lexical(term)
        if (
            not normalized
            or normalized in SPANISH_STOPWORDS
            or (len(normalized) < 2 and not normalized.isdigit())
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def build_fts_match_query(value: str) -> tuple[str, list[str]]:
    terms = extract_lexical_terms(value)
    return " OR ".join(f'"{term}"' for term in terms), terms


def lexical_term_coverage(
    chunk: dict[str, Any],
    terms: list[str],
) -> tuple[float, int]:
    if not terms:
        return 0.0, 0
    haystack = normalize_lexical(
        " ".join(
            [
                str(chunk.get("title", "")),
                str(chunk.get("heading", "")),
                str(
                    chunk.get("matched_content")
                    or chunk.get("content", "")
                ),
                str(chunk.get("path", "")),
            ]
        )
    )
    haystack_terms = {
        term.strip("_")
        for term in re.findall(r"\w+", haystack, flags=re.UNICODE)
        if term.strip("_")
    }
    matched = sum(normalize_lexical(term) in haystack_terms for term in terms)
    return matched / len(terms), matched


def required_lexical_matches(term_count: int) -> int:
    if term_count <= 1:
        return 1
    if term_count <= 10:
        return 2
    return max(2, math.ceil(term_count * 0.2))


# ----------------------------------------------------------------------
# Estado
# ----------------------------------------------------------------------
def get_retrieval_status() -> dict[str, Any]:
    fts = get_fts5_status()
    graph = get_graph_status()
    hybrid_ready = bool(
        settings.hybrid_search_enabled
        and fts["available"]
        and fts["synchronized"]
    )
    scheme = resolve_scheme(
        settings.embedding_model,
        settings.embedding_prefix_scheme,
    )
    configured_mode = (
        "hybrid" if settings.hybrid_search_enabled else "semantic"
    )
    active_mode = "hybrid" if hybrid_ready else "semantic"
    if settings.graph_search_enabled:
        configured_mode += "+graph"
    if graph["active"]:
        active_mode += "+graph"
    return {
        "configured_mode": configured_mode,
        "active_mode": active_mode,
        "semantic_weight": settings.hybrid_semantic_weight,
        "lexical_weight": settings.hybrid_lexical_weight,
        "graph_weight": settings.hybrid_graph_weight,
        "rrf_k": settings.hybrid_rrf_k,
        "fts5": fts,
        "graph": graph,
        "embedding_prefix_scheme": scheme.id,
        "diversification": {
            "mmr_enabled": settings.mmr_enabled,
            "mmr_lambda": settings.mmr_lambda,
            "max_chunks_per_document": settings.max_chunks_per_document,
        },
        "vector_store": vector_store.stats(),
        "rerank": get_rerank_status(),
        "query_routing": {
            "configured": settings.query_routing_enabled,
            # Diagnóstico de la última consulta atendida por este proceso;
            # no es autoritativo bajo concurrencia (dos peticiones a la vez
            # pueden pisarse este campo), sólo orientativo para la UI.
            "last_policy": dict(last_routing_outcome) or None,
        },
        "posterior_abstention": {
            "configured": settings.posterior_abstention_enabled,
            "threshold": settings.posterior_abstention_threshold,
            "last_decision": dict(last_abstention_outcome) or None,
        },
    }


# ----------------------------------------------------------------------
# Consulta
# ----------------------------------------------------------------------
def embed_question(question: str) -> np.ndarray:
    scheme = resolve_scheme(
        settings.embedding_model,
        settings.embedding_prefix_scheme,
    )
    vector = create_query_embedding(f"{scheme.query}{question}")
    return np.asarray(vector, dtype=np.float32)


def _semantic_candidates(
    query_vector: np.ndarray,
    *,
    limit: int,
    min_similarity: float,
    status: str,
    vigencia: str,
    tags: list[str],
) -> list[dict[str, Any]]:
    return vector_store.search(
        query_vector,
        limit=limit,
        min_similarity=min_similarity,
        status=status or None,
        vigencia=vigencia or None,
        tags=tags or None,
    )


def apply_relative_cutoff(
    semantic: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Descarta candidatos muy por debajo del mejor resultado.

    Un umbral absoluto de coseno no se traslada bien entre modelos de
    embedding. El corte relativo mide la distancia al mejor candidato,
    que es una señal comparable con cualquier modelo.
    """
    ratio = settings.min_relative_score
    if ratio <= 0.0 or not semantic:
        return semantic
    best = float(semantic[0]["semantic_score"])
    if best <= 0.0:
        return semantic
    floor = best * ratio
    return [
        item
        for item in semantic
        if float(item["semantic_score"]) >= floor
    ] or semantic[:1]


def _lexical_candidates(
    lexical_text: str,
    *,
    limit: int,
    status: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not settings.hybrid_search_enabled:
        return [], []
    match_query, terms = build_fts_match_query(lexical_text)
    if not match_query:
        return [], terms
    try:
        rows = search_chunks_fts(
            match_query,
            limit=limit,
            status=status or None,
        )
    except sqlite3.Error:
        return [], terms
    return rows, terms


def _passes_filters(
    chunk: dict[str, Any],
    *,
    vigencia: str,
    tags: list[str],
) -> bool:
    metadata = chunk.get("metadata") or {}
    if vigencia:
        actual = document_vigencia(metadata)
        if vigencia == "no_caducada":
            if actual == "caducada":
                return False
        elif actual != vigencia:
            return False
    if tags:
        available = set(document_tags(metadata))
        if not set(tags).issubset(available):
            return False
    return True


def _fuse(
    semantic: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    graph: list[dict[str, Any]],
    hydrated: dict[int, dict[str, Any]],
    lexical_terms: list[str],
    policy: RetrievalPolicy,
) -> list[dict[str, Any]]:
    semantic_ranks = {
        int(item["chunk_id"]): rank
        for rank, item in enumerate(semantic, start=1)
    }
    lexical_ranks: dict[int, int] = {}
    lexical_payload: dict[int, dict[str, float]] = {}
    graph_ranks: dict[int, int] = {}
    graph_payload: dict[int, dict[str, Any]] = {}

    rank = 0
    minimum_matches = required_lexical_matches(len(lexical_terms))
    for row in lexical:
        chunk_id = int(row["chunk_id"])
        chunk = hydrated.get(chunk_id)
        if chunk is None:
            continue
        coverage, matched = lexical_term_coverage(chunk, lexical_terms)
        if matched < minimum_matches:
            continue
        rank += 1
        lexical_ranks[chunk_id] = rank
        lexical_payload[chunk_id] = {
            "lexical_score": max(0.0, -float(row["bm25_score"])),
            "lexical_coverage": coverage,
        }

    rank = 0
    for candidate in graph:
        chunk_id = int(candidate["chunk_id"])
        if chunk_id not in hydrated:
            continue
        rank += 1
        graph_ranks[chunk_id] = rank
        graph_payload[chunk_id] = candidate

    semantic_payload = {int(item["chunk_id"]): item for item in semantic}

    # Los pesos vienen de la política de la consulta, no de `settings`
    # directamente: con el enrutador activo, dos peticiones concurrentes
    # pueden fusionar con pesos distintos sin pisarse (ver query_routing.py).
    semantic_weight = policy.semantic_weight
    lexical_weight = (
        policy.lexical_weight if settings.hybrid_search_enabled else 0.0
    )
    graph_weight = policy.graph_weight if policy.use_graph else 0.0
    if semantic_weight + lexical_weight + graph_weight <= 0:
        semantic_weight = 1.0
    rrf_k = settings.hybrid_rrf_k
    maximum_rrf = (
        semantic_weight + lexical_weight + graph_weight
    ) / (rrf_k + 1)

    fused: list[dict[str, Any]] = []
    candidate_ids = (
        set(semantic_ranks) | set(lexical_ranks) | set(graph_ranks)
    )
    for chunk_id in candidate_ids:
        chunk = hydrated.get(chunk_id)
        if chunk is None:
            continue
        item = dict(chunk)

        semantic_rank = semantic_ranks.get(chunk_id)
        lexical_rank = lexical_ranks.get(chunk_id)
        graph_rank = graph_ranks.get(chunk_id)
        raw_rrf = 0.0
        if semantic_rank is not None:
            raw_rrf += semantic_weight / (rrf_k + semantic_rank)
        if lexical_rank is not None:
            raw_rrf += lexical_weight / (rrf_k + lexical_rank)
        if graph_rank is not None:
            raw_rrf += graph_weight / (rrf_k + graph_rank)

        semantic_score = (
            float(semantic_payload[chunk_id]["semantic_score"])
            if chunk_id in semantic_payload
            else None
        )
        item["row"] = (
            int(semantic_payload[chunk_id]["row"])
            if chunk_id in semantic_payload
            else (
                int(graph_payload[chunk_id]["row"])
                if chunk_id in graph_payload
                else vector_store.row_for_chunk(chunk_id)
            )
        )
        lexical_data = lexical_payload.get(chunk_id, {})
        lexical_coverage = lexical_data.get("lexical_coverage")
        graph_data = graph_payload.get(chunk_id, {})
        graph_score = graph_data.get("graph_score")

        item["semantic_score"] = semantic_score
        item["lexical_score"] = lexical_data.get("lexical_score")
        item["lexical_coverage"] = lexical_coverage
        if policy.use_graph:
            item["graph_score"] = graph_score
            item["graph_hop"] = graph_data.get("graph_hop")
            item["graph_chunk_similarity"] = graph_data.get(
                "graph_chunk_similarity"
            )

        components: list[tuple[str, float, float]] = []
        if semantic_score is not None:
            components.append(
                ("semántica", semantic_weight, semantic_score)
            )
        if lexical_coverage is not None:
            components.append(
                ("texto", lexical_weight, float(lexical_coverage))
            )
        if graph_score is not None:
            components.append(("grafo", graph_weight, float(graph_score)))

        weighted = sum(weight * value for _, weight, value in components)
        total_weight = sum(weight for _, weight, _ in components)
        score = (
            weighted / total_weight
            if total_weight > 0
            else max((value for _, _, value in components), default=0.0)
        )
        names = [name for name, _, _ in components]
        if names == ["semántica"]:
            reason = "búsqueda semántica"
        elif names == ["texto"]:
            reason = "búsqueda textual"
        elif names == ["grafo"]:
            reason = "búsqueda por grafo"
        else:
            reason = f"búsqueda híbrida ({' + '.join(names)})"

        item["score"] = max(0.0, min(1.0, score))
        item["fusion_score"] = (
            raw_rrf / maximum_rrf if maximum_rrf > 0 else 0.0
        )
        item["reason"] = reason
        fused.append(item)

    fused.sort(
        key=lambda entry: (entry["fusion_score"], entry["score"]),
        reverse=True,
    )
    return fused


def _cap_per_document(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    cap = settings.max_chunks_per_document
    selected: list[dict[str, Any]] = []
    per_document: dict[int, int] = {}
    overflow: list[dict[str, Any]] = []
    for candidate in candidates:
        document_id = candidate["document_id"]
        if per_document.get(document_id, 0) >= cap:
            overflow.append(candidate)
            continue
        per_document[document_id] = per_document.get(document_id, 0) + 1
        selected.append(candidate)
        if len(selected) >= top_k:
            return selected
    for candidate in overflow:
        if len(selected) >= top_k:
            break
        selected.append(candidate)
    return selected


def collapse_parent_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Conserva el mejor hijo por padre antes de reordenar y diversificar."""
    collapsed: list[dict[str, Any]] = []
    by_parent: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        parent_id = candidate.get("parent_id")
        if parent_id is None:
            item = dict(candidate)
            item["matched_chunk_ids"] = [int(candidate["chunk_id"])]
            collapsed.append(item)
            continue

        key = int(parent_id)
        existing = by_parent.get(key)
        if existing is None:
            item = dict(candidate)
            item["matched_chunk_ids"] = [int(candidate["chunk_id"])]
            by_parent[key] = item
            collapsed.append(item)
            continue
        existing["matched_chunk_ids"].append(int(candidate["chunk_id"]))
    return collapsed


def diversify(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """MMR con tope de fragmentos por documento.

    Evita que varios fragmentos casi idénticos, o una única nota muy
    larga, ocupen toda la ventana de contexto del redactor.
    """
    if not candidates or top_k <= 0:
        return []
    if not settings.mmr_enabled or len(candidates) <= 1:
        return candidates[:top_k]

    rows = [
        candidate["row"]
        for candidate in candidates
        if candidate.get("row") is not None
    ]
    if len(rows) != len(candidates):
        return _cap_per_document(candidates, top_k=top_k)

    vectors = vector_store.vectors_for_rows(rows)
    if vectors.shape[0] != len(candidates):
        return _cap_per_document(candidates, top_k=top_k)

    relevance = np.array(
        [float(candidate["fusion_score"]) for candidate in candidates],
        dtype=np.float32,
    )
    span = float(relevance.max() - relevance.min())
    if span > 0:
        relevance = (relevance - relevance.min()) / span
    else:
        relevance = np.ones_like(relevance)

    lambda_value = settings.mmr_lambda
    cap = settings.max_chunks_per_document

    selected: list[int] = []
    per_document: dict[int, int] = {}
    remaining = set(range(len(candidates)))
    similarity_to_selected = np.zeros(len(candidates), dtype=np.float32)

    while remaining and len(selected) < top_k:
        allowed = [
            index
            for index in remaining
            if per_document.get(candidates[index]["document_id"], 0) < cap
        ]
        pool = allowed or list(remaining)

        best_index = max(
            pool,
            key=lambda index: (
                lambda_value * float(relevance[index])
                - (1.0 - lambda_value) * float(similarity_to_selected[index])
            ),
        )
        selected.append(best_index)
        remaining.discard(best_index)
        document_id = candidates[best_index]["document_id"]
        per_document[document_id] = per_document.get(document_id, 0) + 1

        if remaining:
            similarities = vectors @ vectors[best_index]
            similarity_to_selected = np.maximum(
                similarity_to_selected,
                similarities,
            )

    return [candidates[index] for index in selected]


# ----------------------------------------------------------------------
# Expansión de enlaces [[...]]
# ----------------------------------------------------------------------
def choose_linked_chunk(
    candidates: list[dict[str, Any]],
    section: str,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    if section:
        wanted = normalize(section).lstrip("^")
        for candidate in candidates:
            heading = normalize(candidate["heading"])
            if wanted == heading or wanted in heading:
                return candidate
    return min(candidates, key=lambda item: item["chunk_index"])


def guarantee_graph_neighbors(
    selected: list[dict[str, Any]],
    graph: list[dict[str, Any]],
    hydrated: dict[int, dict[str, Any]],
    *,
    limit: int = MAX_LINKED_ADDITIONS,
) -> list[dict[str, Any]]:
    """Añade el mejor vecino a un salto que la fusión RRF dejó fuera.

    Con el grafo activo, `expand_wiki_links` se desactiva para no duplicar
    fragmentos. Pero un vecino a un salto puede perder la competencia de
    RRF frente a semántica y texto cuando `hybrid_graph_weight` es bajo:
    antes esa nota entraba garantizada vía `expand_links`, ahora podía
    desaparecer sin más. Esto restaura la misma garantía, limitada a salto 1
    y con el fragmento que el grafo ya eligió por similitud (no por
    `chunk_index`).
    """
    if not selected or not graph:
        return []

    selected_chunk_ids = {item["chunk_id"] for item in selected}
    selected_document_ids = {item["document_id"] for item in selected}
    additions: list[dict[str, Any]] = []

    for candidate in graph:
        if candidate.get("graph_hop") != 1:
            continue
        chunk_id = int(candidate["chunk_id"])
        document_id = candidate["document_id"]
        if (
            chunk_id in selected_chunk_ids
            or document_id in selected_document_ids
        ):
            continue
        chunk = hydrated.get(chunk_id)
        if chunk is None:
            continue

        addition = dict(chunk)
        similarity = float(candidate["graph_chunk_similarity"])
        addition["score"] = max(0.0, similarity * 0.85)
        addition["fusion_score"] = max(0.0, similarity * 0.85)
        addition["semantic_score"] = None
        addition["lexical_score"] = None
        addition["graph_score"] = float(candidate["graph_score"])
        addition["graph_hop"] = 1
        addition["graph_chunk_similarity"] = similarity
        addition["reason"] = "búsqueda por grafo (enlace directo)"
        additions.append(addition)
        selected_chunk_ids.add(chunk_id)
        selected_document_ids.add(document_id)

        if len(additions) >= limit:
            break
    return additions


def expand_wiki_links(
    selected: list[dict[str, Any]],
    *,
    status: str,
    vigencia: str = "",
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    tags = tags or []
    if not selected:
        return []

    alias_to_document = build_alias_index(document_directory())

    selected_chunk_ids = {item["chunk_id"] for item in selected}
    selected_document_ids = {item["document_id"] for item in selected}
    additions: list[dict[str, Any]] = []

    for source in selected[: min(3, len(selected))]:
        for link in source.get("links", []):
            document_id = resolve_document_target(
                alias_to_document,
                str(link.get("target", "")),
            )
            if not document_id or document_id in selected_document_ids:
                continue

            available = document_chunks(document_id)
            if status:
                available = [
                    chunk
                    for chunk in available
                    if metadata_status(chunk["metadata"]) == status
                ]
            if vigencia or tags:
                available = [
                    chunk
                    for chunk in available
                    if _passes_filters(chunk, vigencia=vigencia, tags=tags)
                ]
            candidate = choose_linked_chunk(
                available,
                str(link.get("section", "")),
            )
            if not candidate or candidate["chunk_id"] in selected_chunk_ids:
                continue

            addition = dict(candidate)
            addition["score"] = max(0.0, float(source["score"]) * 0.85)
            addition["fusion_score"] = max(
                0.0,
                float(source.get("fusion_score", 0.0)) * 0.85,
            )
            addition["semantic_score"] = None
            addition["lexical_score"] = None
            addition["reason"] = f"enlace desde «{source['title']}»"
            additions.append(addition)
            selected_chunk_ids.add(addition["chunk_id"])
            selected_document_ids.add(addition["document_id"])

            if len(additions) >= MAX_LINKED_ADDITIONS:
                return additions
    return additions


# ----------------------------------------------------------------------
# Punto de entrada
# ----------------------------------------------------------------------
def normalize_tag_filter(tags: list[str] | None) -> list[str]:
    from app.rag.obsidian_syntax import normalize_tag

    if not tags:
        return []
    cleaned = [normalize_tag(str(tag)) for tag in tags]
    return [tag for tag in dict.fromkeys(cleaned) if tag]


def _build_policy(
    question: str,
    semantic: list[dict[str, Any]],
    *,
    forced_policy: RetrievalPolicy | None,
    force_routing: bool,
) -> RetrievalPolicy:
    """Calcula la política de esta consulta sin tocar `settings`.

    `forced_policy` existe para la evaluación comparativa (objetivo 5):
    permite fijar "grafo siempre apagado", "grafo siempre a 1 salto" o una
    política oráculo derivada de `query_type`, sin pasar por el enrutador
    real. `force_routing` permite ejercitar el enrutador real aunque
    `RAG_QUERY_ROUTING` esté desactivado globalmente, para comparar su
    comportamiento sin cambiar la configuración activa del servidor. En
    producción ambos son su valor por defecto (`None`/`False`).
    """
    if forced_policy is not None:
        return forced_policy
    if not settings.query_routing_enabled and not force_routing:
        return RetrievalPolicy.from_settings()

    alias_index = build_alias_index(document_directory())
    edges = graph_store.edges()
    return route_query(
        question,
        semantic=semantic,
        alias_index=alias_index,
        edges=edges,
    )


def retrieve(
    question: str,
    *,
    top_k: int,
    min_similarity: float,
    status: str | None,
    expand_links: bool,
    lexical_question: str | None = None,
    vigencia: str | None = None,
    tags: list[str] | None = None,
    forced_policy: RetrievalPolicy | None = None,
    force_routing: bool = False,
) -> list[dict[str, Any]]:
    requested_status = status.strip().casefold() if status else ""
    requested_vigencia = vigencia.strip().casefold() if vigencia else ""
    requested_tags = normalize_tag_filter(tags)
    candidate_limit = max(20, top_k * settings.hybrid_candidate_multiplier)

    query_vector = embed_question(question)
    semantic = _semantic_candidates(
        query_vector,
        limit=candidate_limit,
        min_similarity=min_similarity,
        status=requested_status,
        vigencia=requested_vigencia,
        tags=requested_tags,
    )

    # Regla de abstención: sin evidencia semántica no se responde. FTS5 y
    # el grafo complementan y reordenan, pero no pueden anularla. Corre
    # antes de enrutar: el router nunca ve una consulta sin evidencia.
    if not semantic:
        return []

    semantic = apply_relative_cutoff(semantic)

    # La política es un valor local: dos consultas concurrentes con
    # políticas distintas no se pisan porque nada global se muta aquí.
    policy = _build_policy(
        question,
        semantic,
        forced_policy=forced_policy,
        force_routing=force_routing,
    )
    last_routing_outcome.clear()
    last_routing_outcome.update(policy.as_dict())

    lexical, lexical_terms = _lexical_candidates(
        lexical_question or question,
        limit=candidate_limit * 2,
        status=requested_status,
    )
    # Una consulta factual nunca invoca el grafo: no basta con que
    # `graph_candidates` se autolimite por dentro, la propia llamada se
    # omite para que "grafo desactivado para esta consulta" sea observable
    # y objeto de prueba, igual que la abstención previa.
    graph = (
        graph_candidates(
            semantic,
            query_vector,
            use_graph=policy.use_graph,
            seed_documents=policy.graph_seed_documents,
            max_hops=policy.graph_max_hops,
            decay=policy.graph_decay,
            backlink_weight=policy.graph_backlink_weight,
            max_candidates=policy.graph_max_candidates,
        )
        if policy.use_graph
        else []
    )

    chunk_ids = [int(item["chunk_id"]) for item in semantic]
    chunk_ids.extend(int(row["chunk_id"]) for row in lexical)
    chunk_ids.extend(int(item["chunk_id"]) for item in graph)
    hydrated = hydrate_chunks(chunk_ids)

    if requested_status or requested_vigencia or requested_tags:
        # FTS5 sólo conoce el estado y el grafo no almacena metadatos. Los
        # filtros se reaplican tras hidratar para que ninguna rama cuele
        # documentos que la semántica ya había descartado.
        hydrated = {
            chunk_id: chunk
            for chunk_id, chunk in hydrated.items()
            if (
                not requested_status
                or metadata_status(chunk.get("metadata") or {})
                == requested_status
            )
            and _passes_filters(
                chunk,
                vigencia=requested_vigencia,
                tags=requested_tags,
            )
        }

    fused = _fuse(semantic, lexical, graph, hydrated, lexical_terms, policy)
    fused = collapse_parent_candidates(fused)

    # El reordenador actúa sobre los candidatos fusionados y antes de la
    # diversificación, para que MMR y el tope por documento se apliquen
    # sobre el orden ya corregido.
    outcome = rerank(question, fused)
    fused = outcome.items
    last_rerank_outcome.clear()
    last_rerank_outcome.update(outcome.as_dict())

    selected = diversify(fused, top_k=top_k)

    for item in selected:
        item.pop("row", None)
        metadata = item.get("metadata") or {}
        item["vigencia"] = vigencia_details(metadata)
        item["tags"] = document_tags(metadata)

    # Abstención posterior: mira el conjunto final ya diversificado. Sigue
    # desactivada por defecto (ver app.rag.abstention); cuando se activa,
    # puede vaciar `selected` aunque la abstención previa ya haya pasado.
    abstention_decision = evaluate_posterior_abstention(selected, top_k=top_k)
    last_abstention_outcome.clear()
    last_abstention_outcome.update(abstention_decision.as_dict())
    if abstention_decision.should_abstain:
        return []

    if not expand_links or not selected:
        return selected

    if policy.use_graph:
        # El grafo ya sustituye la expansión por wikienlaces, pero por sí
        # solo puede perder un vecino directo en la competencia de RRF.
        # Se restaura esa garantía desde los propios candidatos del grafo,
        # sin volver a parsear enlaces ni duplicar fragmentos.
        additions = guarantee_graph_neighbors(selected, graph, hydrated)
    else:
        additions = expand_wiki_links(
            selected,
            status=requested_status,
            vigencia=requested_vigencia,
            tags=requested_tags,
        )
    for item in additions:
        metadata = item.get("metadata") or {}
        item["vigencia"] = vigencia_details(metadata)
        item["tags"] = document_tags(metadata)
    return selected + additions

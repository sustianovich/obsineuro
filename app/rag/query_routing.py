"""Enrutador determinista de consultas: decide la política de recuperación.

No hay llamada a LLM aquí. La ablación registrada en
`evaluations/reports/graph_comparison.md` mostró que activar el grafo con
un peso fijo para *toda* consulta perjudica a las preguntas puramente
factuales (MRR de 0,938 a 0,474 al subir el peso de 0 a 0,5 sobre las 16
preguntas PDPCM, todas factuales). La respuesta correcta no es un peso de
compromiso, sino decidir por consulta si el grafo debe participar.

El resultado de `route_query` es una `RetrievalPolicy`: un valor inmutable
que la recuperación pasa explícitamente a `_fuse` y `graph_candidates`. No
se muta `settings`, porque el servidor puede atender consultas
concurrentes con políticas distintas y una variable global compartida
produciría condiciones de carrera entre peticiones simultáneas.

Señales usadas, todas auditables (se registran en `reasons`):

  - expresiones relacionales configurables (`RAG_QUERY_ROUTER_RELATIONAL_PATTERNS`);
  - alias o nombres de nota conocidos mencionados en la propia pregunta;
  - conexión directa en el grafo entre los documentos semilla semánticos;
  - sustantivos estructurales configurables (nodo, proceso, riesgo...)
    mencionados más de una vez (`RAG_QUERY_ROUTER_STRUCTURAL_NOUNS`).

"Fuera de dominio" no se decide aquí: este router nunca ve una pregunta sin
evidencia semántica, porque `retrieve()` ya abstiene antes de enrutar. Lo
que sí puede pasar es evidencia débil pese a superar el umbral, y ese caso
cae en `unknown` con la política conservadora de `factual`.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Sequence

from app.config import settings


QueryType = str  # "factual" | "relational" | "hybrid" | "unknown" | "unrouted"


@dataclass(frozen=True)
class RetrievalPolicy:
    """Parámetros de una única consulta. Nunca se deriva mutando `settings`."""

    query_type: QueryType
    confidence: float
    use_graph: bool
    semantic_weight: float
    lexical_weight: float
    graph_weight: float
    graph_max_hops: int
    graph_decay: float
    graph_backlink_weight: float
    graph_seed_documents: int
    graph_max_candidates: int
    min_similarity: float
    min_relative_score: float
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "confidence": round(self.confidence, 3),
            "use_graph": self.use_graph,
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight,
            "graph_weight": self.graph_weight,
            "graph_max_hops": self.graph_max_hops,
            "graph_seed_documents": self.graph_seed_documents,
            "graph_max_candidates": self.graph_max_candidates,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_settings(
        cls,
        *,
        query_type: QueryType = "unrouted",
        confidence: float = 1.0,
        reasons: tuple[str, ...] = (),
    ) -> "RetrievalPolicy":
        """Política derivada únicamente de `settings`.

        Es el comportamiento previo al enrutador: cuando `RAG_QUERY_ROUTING`
        está desactivado, `retrieve()` usa exactamente esta política, así
        que el resultado es indistinguible del sistema sin enrutador.
        """
        return cls(
            query_type=query_type,
            confidence=confidence,
            use_graph=settings.graph_search_enabled,
            semantic_weight=settings.hybrid_semantic_weight,
            lexical_weight=(
                settings.hybrid_lexical_weight
                if settings.hybrid_search_enabled
                else 0.0
            ),
            graph_weight=(
                settings.hybrid_graph_weight
                if settings.graph_search_enabled
                else 0.0
            ),
            graph_max_hops=settings.graph_max_hops,
            graph_decay=settings.graph_decay,
            graph_backlink_weight=settings.graph_backlink_weight,
            graph_seed_documents=settings.graph_seed_documents,
            graph_max_candidates=settings.graph_max_candidates,
            min_similarity=settings.min_similarity,
            min_relative_score=settings.min_relative_score,
            reasons=reasons,
        )


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _matched_patterns(normalized_question: str, patterns: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        pattern for pattern in patterns if _normalize_text(pattern) in normalized_question
    )


def _matched_entities(
    normalized_question: str,
    alias_index: dict[str, Any],
    *,
    minimum_length: int = 4,
    limit: int = 3,
) -> tuple[str, ...]:
    matched: list[str] = []
    seen_identities: set[Any] = set()
    for alias, identity in alias_index.items():
        if len(alias) < minimum_length or identity in seen_identities:
            continue
        # `alias_index` normaliza con casefold pero conserva acentos
        # (app.rag.graph.normalize); aquí comparamos sin acentos para que
        # una pregunta escrita sin tildes siga reconociendo la nota.
        if _normalize_text(alias) in normalized_question:
            seen_identities.add(identity)
            matched.append(alias)
            if len(matched) >= limit:
                break
    return tuple(matched)


def _matched_structural_nouns(
    normalized_question: str,
    nouns: Sequence[str],
) -> tuple[str, ...]:
    return tuple(noun for noun in nouns if _normalize_text(noun) in normalized_question)


def _seed_documents_connected(
    seed_document_ids: Sequence[int],
    edges: Sequence[tuple[int, int]],
) -> bool:
    seeds = set(seed_document_ids)
    if len(seeds) < 2:
        return False
    for source, target in edges:
        if source in seeds and target in seeds and source != target:
            return True
    return False


def route_query(
    question: str,
    *,
    semantic: list[dict[str, Any]],
    alias_index: dict[str, Any] | None = None,
    edges: Sequence[tuple[int, int]] | None = None,
    force: bool = False,
) -> RetrievalPolicy:
    """Clasifica una consulta y devuelve la política a aplicar.

    `semantic` ya llegó filtrado por la abstención de `retrieve()`: nunca
    está vacío aquí. `alias_index` y `edges` son opcionales para que las
    pruebas puedan enrutar sin tocar SQLite; en producción `retrieve()` los
    calcula una sola vez por consulta. `force` sólo se usa en evaluaciones
    para medir el enrutador aunque esté desactivado globalmente.
    """
    if not settings.query_routing_enabled and not force:
        return RetrievalPolicy.from_settings(
            query_type="unrouted",
            confidence=1.0,
            reasons=("enrutador desactivado (RAG_QUERY_ROUTING=false)",),
        )

    alias_index = alias_index or {}
    edges = edges or ()
    normalized_question = _normalize_text(question)

    relational_hits = _matched_patterns(
        normalized_question, settings.query_router_relational_patterns
    )
    entity_hits = _matched_entities(normalized_question, alias_index)
    structural_hits = _matched_structural_nouns(
        normalized_question, settings.query_router_structural_nouns
    )
    seed_ids = [
        int(item["document_id"])
        for item in semantic[: settings.graph_seed_documents]
    ]
    connected = _seed_documents_connected(seed_ids, edges)

    signals = {
        "expresion_relacional": bool(relational_hits),
        "alias_conocidos": len(entity_hits) >= settings.query_router_min_entity_mentions,
        "semillas_conectadas": connected,
        "sustantivos_estructurales": len(structural_hits) >= 2,
    }
    relational_count = sum(signals.values())
    confidence = relational_count / len(signals)

    reasons: list[str] = []
    if relational_hits:
        reasons.append(f"expresión relacional: {', '.join(relational_hits)}")
    if signals["alias_conocidos"]:
        reasons.append(f"notas mencionadas por nombre: {', '.join(entity_hits)}")
    if connected:
        reasons.append("los documentos semilla semánticos ya están conectados en el grafo")
    if signals["sustantivos_estructurales"]:
        reasons.append(
            f"sustantivos estructurales: {', '.join(structural_hits)}"
        )
    if not any(signals.values()):
        reasons.append("ninguna señal relacional activada")

    best_semantic = float(semantic[0]["semantic_score"]) if semantic else 0.0
    weak_evidence = best_semantic < (
        settings.min_similarity + settings.query_router_weak_evidence_margin
    )

    baseline = RetrievalPolicy.from_settings()

    if weak_evidence and relational_count == 0:
        reasons.append(
            "evidencia semántica débil "
            f"({best_semantic:.3f} < umbral+margen "
            f"{settings.min_similarity + settings.query_router_weak_evidence_margin:.3f})"
        )
        return replace(
            baseline,
            query_type="unknown",
            confidence=confidence,
            use_graph=False,
            graph_weight=0.0,
            reasons=tuple(reasons),
        )

    if relational_count >= 2:
        reasons.append("2 o más señales relacionales -> grafo activo a 1 salto")
        return replace(
            baseline,
            query_type="relational",
            confidence=confidence,
            use_graph=True,
            graph_weight=settings.query_router_relational_graph_weight,
            graph_max_hops=1,
            reasons=tuple(reasons),
        )

    if relational_count == 1:
        reasons.append(
            "1 señal relacional -> híbrida: semántica y texto conservan peso, "
            "grafo se suma a 1 salto"
        )
        return replace(
            baseline,
            query_type="hybrid",
            confidence=confidence,
            use_graph=True,
            graph_weight=settings.query_router_hybrid_graph_weight,
            graph_max_hops=1,
            reasons=tuple(reasons),
        )

    reasons.append("sin señales relacionales -> factual: grafo desactivado")
    return replace(
        baseline,
        query_type="factual",
        confidence=confidence,
        use_graph=False,
        graph_weight=0.0,
        reasons=tuple(reasons),
    )


def policy_for_query_type(query_type: QueryType) -> RetrievalPolicy:
    """Política determinista para un `query_type` ya conocido.

    Sirve para la evaluación "oráculo" (objetivo 5): compara el techo que
    lograría un enrutador perfecto frente al enrutador real de
    `route_query`. Si ni siquiera esta política mejora las preguntas
    relacionales, no merece la pena invertir en el clasificador.
    """
    baseline = RetrievalPolicy.from_settings()
    if query_type == "relational":
        return replace(
            baseline,
            query_type="relational",
            confidence=1.0,
            use_graph=True,
            graph_weight=settings.query_router_relational_graph_weight,
            graph_max_hops=1,
            reasons=("oráculo: query_type=relational",),
        )
    if query_type == "hybrid":
        return replace(
            baseline,
            query_type="hybrid",
            confidence=1.0,
            use_graph=True,
            graph_weight=settings.query_router_hybrid_graph_weight,
            graph_max_hops=1,
            reasons=("oráculo: query_type=hybrid",),
        )
    resolved_type = query_type if query_type in ("factual", "out_of_domain") else "factual"
    return replace(
        baseline,
        query_type=resolved_type,
        confidence=1.0,
        use_graph=False,
        graph_weight=0.0,
        reasons=(f"oráculo: query_type={query_type}",),
    )

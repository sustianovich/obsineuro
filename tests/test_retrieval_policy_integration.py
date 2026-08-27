"""Integración entre el enrutador (app.rag.query_routing) y `retrieve()`.

Complementa tests/test_query_routing.py (que prueba `route_query` de forma
aislada) verificando que `retrieve()` de verdad conecta la política elegida
con `graph_candidates` y `_fuse`, sin tocar `settings`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.config import settings


def hydrated_chunk(chunk_id: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_index": 0,
        "heading": "Documento",
        "content": f"Contenido {chunk_id}",
        "embedding_dim": 2,
        "document_id": chunk_id,
        "path": f"{chunk_id}.md",
        "title": f"Nota {chunk_id}",
        "metadata": {},
        "links": [],
    }


def _base_monkeypatches(monkeypatch, retrieval):
    monkeypatch.setattr(settings, "query_routing_enabled", True)
    monkeypatch.setattr(settings, "hybrid_search_enabled", False)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "mmr_enabled", False)
    monkeypatch.setattr(settings, "min_similarity", 0.30)
    monkeypatch.setattr(
        retrieval, "embed_question", lambda _: np.array([1.0, 0.0], dtype=np.float32)
    )

    def fake_semantic(query_vector, *, limit, min_similarity, status, vigencia, tags):
        return [
            {
                "chunk_id": 1,
                "document_id": 1,
                "row": 0,
                "semantic_score": 0.9,
            }
        ]

    monkeypatch.setattr(retrieval, "_semantic_candidates", fake_semantic)
    monkeypatch.setattr(retrieval, "hydrate_chunks", lambda ids: {
        chunk_id: hydrated_chunk(chunk_id) for chunk_id in ids
    })
    monkeypatch.setattr(retrieval, "build_alias_index", lambda documents: {})
    monkeypatch.setattr(retrieval, "document_directory", lambda: [])

    class FakeGraphStore:
        def edges(self):
            return []

    monkeypatch.setattr(retrieval, "graph_store", FakeGraphStore())


# ----------------------------------------------------------------------
# 2. Una consulta factual no ejecuta graph_candidates.
# ----------------------------------------------------------------------
def test_consulta_factual_no_ejecuta_graph_candidates(monkeypatch):
    import app.rag.retrieval as retrieval

    _base_monkeypatches(monkeypatch, retrieval)

    def graph_must_not_run(*args, **kwargs):
        raise AssertionError("una consulta factual no debe ejecutar el grafo")

    monkeypatch.setattr(retrieval, "graph_candidates", graph_must_not_run)

    hits = retrieval.retrieve(
        "cuál es el plazo máximo de la fase de lectura",
        top_k=5,
        min_similarity=0.3,
        status=None,
        expand_links=True,
    )
    assert hits
    assert retrieval.last_routing_outcome["query_type"] == "factual"
    assert retrieval.last_routing_outcome["use_graph"] is False


# ----------------------------------------------------------------------
# 3. Una consulta relacional usa el grafo a un salto.
# ----------------------------------------------------------------------
def test_consulta_relacional_activa_grafo_a_un_salto(monkeypatch):
    import app.rag.retrieval as retrieval

    _base_monkeypatches(monkeypatch, retrieval)

    captured: dict = {}

    def fake_graph_candidates(semantic, query_vector, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(retrieval, "graph_candidates", fake_graph_candidates)

    hits = retrieval.retrieve(
        "¿cómo se relaciona el nodo con el proceso de auditoría?",
        top_k=5,
        min_similarity=0.3,
        status=None,
        expand_links=True,
    )
    assert hits
    assert captured["use_graph"] is True
    assert captured["max_hops"] == 1
    assert retrieval.last_routing_outcome["query_type"] == "relational"


# ----------------------------------------------------------------------
# 4. Una consulta híbrida conserva semántica, texto y grafo.
# ----------------------------------------------------------------------
def test_consulta_hibrida_conserva_las_tres_ramas(monkeypatch):
    import app.rag.retrieval as retrieval

    monkeypatch.setattr(settings, "query_routing_enabled", True)
    monkeypatch.setattr(settings, "hybrid_search_enabled", True)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "mmr_enabled", False)
    monkeypatch.setattr(settings, "min_similarity", 0.30)
    monkeypatch.setattr(
        retrieval, "embed_question", lambda _: np.array([1.0, 0.0], dtype=np.float32)
    )
    monkeypatch.setattr(
        retrieval,
        "_semantic_candidates",
        lambda *a, **k: [
            {"chunk_id": 1, "document_id": 1, "row": 0, "semantic_score": 0.9}
        ],
    )
    monkeypatch.setattr(
        retrieval,
        "_lexical_candidates",
        lambda *a, **k: (
            [{"chunk_id": 1, "bm25_score": -2.0}],
            ["nodo", "afecta"],
        ),
    )
    monkeypatch.setattr(
        retrieval,
        "graph_candidates",
        lambda *a, **k: [
            {
                "chunk_id": 1,
                "document_id": 1,
                "row": 0,
                "graph_score": 0.5,
                "graph_hop": 1,
                "graph_chunk_similarity": 0.8,
            }
        ],
    )
    monkeypatch.setattr(retrieval, "hydrate_chunks", lambda ids: {
        1: {
            **hydrated_chunk(1),
            "title": "nodo",
            "content": "el nodo afecta al resultado",
        }
    })
    monkeypatch.setattr(retrieval, "build_alias_index", lambda documents: {})
    monkeypatch.setattr(retrieval, "document_directory", lambda: [])

    class FakeGraphStore:
        def edges(self):
            return []

    monkeypatch.setattr(retrieval, "graph_store", FakeGraphStore())

    hits = retrieval.retrieve(
        "¿cómo afecta el nodo al resultado final?",
        top_k=5,
        min_similarity=0.3,
        status=None,
        expand_links=False,
    )
    assert len(hits) == 1
    hit = hits[0]
    assert hit["semantic_score"] is not None
    assert hit["lexical_score"] is not None
    assert hit["graph_score"] is not None
    assert retrieval.last_routing_outcome["query_type"] == "hybrid"


# ----------------------------------------------------------------------
# 14. El estado expone si el enrutador está configurado y la última política.
# ----------------------------------------------------------------------
def test_estado_expone_configuracion_del_router(monkeypatch):
    import app.rag.retrieval as retrieval

    monkeypatch.setattr(settings, "query_routing_enabled", True)
    status = retrieval.get_retrieval_status()
    assert status["query_routing"]["configured"] is True
    assert "posterior_abstention" in status
    assert status["posterior_abstention"]["configured"] == (
        settings.posterior_abstention_enabled
    )


def test_abstencion_previa_reemplaza_telemetria_posterior_obsoleta(monkeypatch):
    import app.rag.retrieval as retrieval

    _base_monkeypatches(monkeypatch, retrieval)
    monkeypatch.setattr(
        retrieval,
        "_semantic_candidates",
        lambda *args, **kwargs: [],
    )
    retrieval.last_abstention_outcome.clear()
    retrieval.last_abstention_outcome.update(
        {"stage": "post_fusion", "combined_score": 0.99}
    )

    hits = retrieval.retrieve(
        "pregunta sin evidencia semántica",
        top_k=5,
        min_similarity=0.3,
        status=None,
        expand_links=False,
    )

    assert hits == []
    assert retrieval.last_abstention_outcome["stage"] == "pre_retrieval"
    assert retrieval.last_abstention_outcome["should_abstain"] is True
    assert retrieval.last_abstention_outcome["combined_score"] is None


def test_consulta_contextual_rescata_una_pregunta_actual_sin_evidencia(
    monkeypatch,
):
    import app.rag.retrieval as retrieval

    monkeypatch.setattr(settings, "query_routing_enabled", False)
    monkeypatch.setattr(settings, "hybrid_search_enabled", False)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "mmr_enabled", False)
    embedded_questions: list[str] = []

    def fake_embed(question):
        embedded_questions.append(question)
        return (
            np.array([1.0, 0.0], dtype=np.float32)
            if question == "pregunta actual"
            else np.array([0.0, 1.0], dtype=np.float32)
        )

    def fake_semantic(query_vector, **kwargs):
        if float(query_vector[0]) == 1.0:
            return []
        return [
            {
                "chunk_id": 2,
                "document_id": 2,
                "row": 1,
                "semantic_score": 0.88,
            }
        ]

    monkeypatch.setattr(retrieval, "embed_question", fake_embed)
    monkeypatch.setattr(retrieval, "_semantic_candidates", fake_semantic)
    monkeypatch.setattr(
        retrieval,
        "hydrate_chunks",
        lambda ids: {chunk_id: hydrated_chunk(chunk_id) for chunk_id in ids},
    )

    hits = retrieval.retrieve(
        "pregunta actual",
        contextual_question="pregunta actual con el tema anterior",
        top_k=2,
        min_similarity=0.3,
        status=None,
        expand_links=False,
    )

    assert embedded_questions == [
        "pregunta actual",
        "pregunta actual con el tema anterior",
    ]
    assert [hit["chunk_id"] for hit in hits] == [2]
    assert hits[0]["semantic_sources"] == ["contextual"]


# ----------------------------------------------------------------------
# 6 y 7 (a nivel de retrieve completo): sin mutación global y sin
# interferencia entre políticas concurrentes.
# ----------------------------------------------------------------------
def test_retrieve_no_muta_settings_globalmente(monkeypatch):
    import app.rag.retrieval as retrieval

    _base_monkeypatches(monkeypatch, retrieval)
    monkeypatch.setattr(retrieval, "graph_candidates", lambda *a, **k: [])

    before = dict(vars(settings))
    retrieval.retrieve(
        "¿cómo se relaciona el nodo con el proceso?",
        top_k=5,
        min_similarity=0.3,
        status=None,
        expand_links=True,
    )
    after = dict(vars(settings))
    assert before == after


def test_dos_politicas_concurrentes_no_se_interfieren(monkeypatch):
    import app.rag.retrieval as retrieval

    _base_monkeypatches(monkeypatch, retrieval)
    monkeypatch.setattr(retrieval, "graph_candidates", lambda *a, **k: [])

    from app.rag.query_routing import RetrievalPolicy

    graph_on = RetrievalPolicy(
        query_type="forced_on",
        confidence=1.0,
        use_graph=True,
        semantic_weight=1.0,
        lexical_weight=0.0,
        graph_weight=0.9,
        graph_max_hops=1,
        graph_decay=0.5,
        graph_backlink_weight=0.7,
        graph_seed_documents=4,
        graph_max_candidates=20,
        min_similarity=0.3,
        min_relative_score=0.62,
        reasons=("forzada: grafo activo",),
    )
    graph_off = RetrievalPolicy(
        query_type="forced_off",
        confidence=1.0,
        use_graph=False,
        semantic_weight=1.0,
        lexical_weight=0.0,
        graph_weight=0.0,
        graph_max_hops=1,
        graph_decay=0.5,
        graph_backlink_weight=0.7,
        graph_seed_documents=4,
        graph_max_candidates=20,
        min_similarity=0.3,
        min_relative_score=0.62,
        reasons=("forzada: grafo apagado",),
    )

    # `graph_candidates` está monkeypatcheado para no tocar SQLite; lo que
    # se comprueba es que cada llamada a `retrieve()` recibe y aplica su
    # propia política sin pisar la de la otra, incluso ejecutándose a la vez.
    monkeypatch.setattr(
        retrieval,
        "graph_candidates",
        lambda *a, use_graph=None, **k: (
            [{"chunk_id": 1, "document_id": 1, "row": 0, "graph_score": 1.0,
              "graph_hop": 1, "graph_chunk_similarity": 0.9}]
            if use_graph
            else []
        ),
    )

    def call(policy):
        return retrieval.retrieve(
            "pregunta cualquiera",
            top_k=5,
            min_similarity=0.3,
            status=None,
            expand_links=False,
            forced_policy=policy,
        )

    policies = [graph_on, graph_off] * 6
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = [future.result() for future in [pool.submit(call, p) for p in policies]]

    for policy, hits in zip(policies, results):
        has_graph_score = any(hit.get("graph_score") is not None for hit in hits)
        assert has_graph_score == policy.use_graph

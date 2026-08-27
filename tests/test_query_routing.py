from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from app.config import settings
from app.rag.query_routing import RetrievalPolicy, policy_for_query_type, route_query


def semantic_seed(document_id: int = 1, score: float = 0.9) -> list[dict]:
    return [{"document_id": document_id, "semantic_score": score}]


# ----------------------------------------------------------------------
# 1. Router desactivado reproduce el comportamiento anterior.
# ----------------------------------------------------------------------
def test_router_desactivado_reproduce_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", False)
    policy = route_query(
        "cualquier pregunta",
        semantic=semantic_seed(),
        alias_index={},
        edges=(),
    )
    baseline = RetrievalPolicy.from_settings()
    assert policy.use_graph == baseline.use_graph
    assert policy.semantic_weight == baseline.semantic_weight
    assert policy.lexical_weight == baseline.lexical_weight
    assert policy.graph_weight == baseline.graph_weight
    assert policy.query_type == "unrouted"


def test_router_se_puede_forzar_para_evaluacion(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", False)
    monkeypatch.setattr(settings, "min_similarity", 0.30)

    policy = route_query(
        "¿cómo se relaciona el nodo con el proceso de auditoría?",
        semantic=semantic_seed(score=0.9),
        alias_index={},
        edges=(),
        force=True,
    )

    assert policy.query_type == "relational"
    assert policy.use_graph is True


# ----------------------------------------------------------------------
# 5. Consulta incierta cae en la política factual (conservadora).
# ----------------------------------------------------------------------
def test_evidencia_debil_sin_senales_cae_en_unknown_factual(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", True)
    monkeypatch.setattr(settings, "min_similarity", 0.30)
    monkeypatch.setattr(settings, "query_router_weak_evidence_margin", 0.05)
    policy = route_query(
        "pregunta ambigua sin ninguna señal reconocible",
        semantic=semantic_seed(score=0.32),
        alias_index={},
        edges=(),
    )
    assert policy.query_type == "unknown"
    assert policy.use_graph is False
    assert policy.graph_weight == 0.0


# ----------------------------------------------------------------------
# 3. Consulta relacional usa un salto (>=2 señales relacionales).
# ----------------------------------------------------------------------
def test_dos_senales_clasifica_como_relacional_a_un_salto(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", True)
    monkeypatch.setattr(settings, "min_similarity", 0.30)
    policy = route_query(
        "¿cómo se relaciona el nodo con el proceso de auditoría?",
        semantic=semantic_seed(score=0.9),
        alias_index={},
        edges=(),
    )
    assert policy.query_type == "relational"
    assert policy.use_graph is True
    assert policy.graph_max_hops == 1
    assert policy.graph_weight == settings.query_router_relational_graph_weight
    assert policy.confidence == pytest.approx(0.5)


# ----------------------------------------------------------------------
# 4. Consulta híbrida conserva las tres ramas (exactamente 1 señal).
# ----------------------------------------------------------------------
def test_una_senal_clasifica_como_hibrida_conserva_pesos(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", True)
    monkeypatch.setattr(settings, "min_similarity", 0.30)
    policy = route_query(
        "¿cómo afecta este cambio al resultado final?",
        semantic=semantic_seed(score=0.9),
        alias_index={},
        edges=(),
    )
    assert policy.query_type == "hybrid"
    assert policy.use_graph is True
    assert policy.graph_max_hops == 1
    assert policy.graph_weight == settings.query_router_hybrid_graph_weight
    assert policy.semantic_weight == settings.hybrid_semantic_weight
    assert policy.lexical_weight == (
        settings.hybrid_lexical_weight if settings.hybrid_search_enabled else 0.0
    )


def test_sin_ninguna_senal_clasifica_como_factual(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", True)
    monkeypatch.setattr(settings, "min_similarity", 0.30)
    policy = route_query(
        "cuál es el plazo máximo de la fase de lectura",
        semantic=semantic_seed(score=0.9),
        alias_index={},
        edges=(),
    )
    assert policy.query_type == "factual"
    assert policy.use_graph is False


def test_alias_mencionados_cuentan_como_senal(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", True)
    monkeypatch.setattr(settings, "min_similarity", 0.30)
    monkeypatch.setattr(settings, "query_router_min_entity_mentions", 2)
    alias_index = {"nodo 22": 42, "kpi 05": 43}
    policy = route_query(
        "¿qué documento conecta Nodo 22 con KPI 05?",
        semantic=semantic_seed(score=0.9),
        alias_index=alias_index,
        edges=(),
    )
    assert any("notas mencionadas" in reason for reason in policy.reasons)
    assert policy.query_type in {"relational", "hybrid"}


def test_semillas_conectadas_en_grafo_cuentan_como_senal(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", True)
    monkeypatch.setattr(settings, "min_similarity", 0.30)
    monkeypatch.setattr(settings, "graph_seed_documents", 4)
    semantic = [
        {"document_id": 1, "semantic_score": 0.9},
        {"document_id": 2, "semantic_score": 0.8},
    ]
    policy = route_query(
        "pregunta neutra sin ninguna expresión relacional evidente",
        semantic=semantic,
        alias_index={},
        edges=((1, 2),),
    )
    assert any("conectados" in reason for reason in policy.reasons)


# ----------------------------------------------------------------------
# 6. Ninguna llamada muta `settings` globalmente.
# ----------------------------------------------------------------------
def test_route_query_no_muta_settings(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", True)
    before = dict(vars(settings))
    route_query(
        "¿cómo se relaciona el nodo con el proceso?",
        semantic=semantic_seed(),
        alias_index={"nodo": 1},
        edges=((1, 2),),
    )
    after = dict(vars(settings))
    assert before == after


# ----------------------------------------------------------------------
# 7. Dos consultas con políticas distintas, en paralelo, no se interfieren.
# ----------------------------------------------------------------------
def test_route_query_concurrente_no_interfiere(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", True)
    monkeypatch.setattr(settings, "min_similarity", 0.30)

    def factual_call() -> str:
        return route_query(
            "cuál es el plazo máximo de la fase de lectura",
            semantic=semantic_seed(score=0.9),
            alias_index={},
            edges=(),
        ).query_type

    def relational_call() -> str:
        return route_query(
            "¿cómo se relaciona el nodo con el proceso de auditoría?",
            semantic=semantic_seed(score=0.9),
            alias_index={},
            edges=(),
        ).query_type

    calls = [factual_call, relational_call] * 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [future.result() for future in [pool.submit(call) for call in calls]]

    assert results == ["factual", "relational"] * 8


def test_policy_for_query_type_oraculo(monkeypatch):
    monkeypatch.setattr(settings, "query_routing_enabled", False)
    relational = policy_for_query_type("relational")
    assert relational.use_graph is True
    assert relational.graph_max_hops == 1
    factual = policy_for_query_type("factual")
    assert factual.use_graph is False
    out_of_domain = policy_for_query_type("out_of_domain")
    assert out_of_domain.use_graph is False

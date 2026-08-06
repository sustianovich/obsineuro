from __future__ import annotations

import pytest

import app.rag.evaluation as evaluation
from app.rag.evaluation import (
    EvaluationCase,
    _core_metrics,
    _router_confusion,
    _summarize,
    ndcg_at_k,
    precision_at_k,
)


def make_result(
    *,
    query_type: str,
    expect_abstention: bool,
    answered: bool,
    retrieval_pass: bool,
    document_recall: float | None = None,
    precision: float | None = None,
    predicted_query_type: str | None = None,
) -> dict:
    return {
        "id": f"{query_type}-{answered}-{retrieval_pass}",
        "expect_abstention": expect_abstention,
        "answered": answered,
        "retrieval_pass": retrieval_pass,
        "document_recall": document_recall,
        "precision_at_k": precision,
        "ndcg_at_k": None,
        "reciprocal_rank": 1.0 if retrieval_pass else 0.0,
        "term_coverage": None,
        "query_type": query_type,
        "predicted_query_type": predicted_query_type,
        "latency_seconds": 0.01,
    }


# ----------------------------------------------------------------------
# 12. Precision@k penaliza candidatos irrelevantes añadidos por el grafo.
# ----------------------------------------------------------------------
def test_precision_at_k_penaliza_documentos_de_mas():
    retrieved = ["a.md", "b.md", "c.md"]
    expected = ("a.md",)
    assert precision_at_k(retrieved, expected) == pytest.approx(1 / 3)


def test_precision_at_k_perfecto_sin_ruido():
    retrieved = ["a.md"]
    expected = ("a.md",)
    assert precision_at_k(retrieved, expected) == pytest.approx(1.0)


def test_retrieval_pass_no_detecta_ruido_pero_precision_si():
    """El caso que motivó el objetivo 5: recall completo, pero un vecino de
    grafo añadido de más no aparece en `forbidden_paths` y por tanto
    `retrieval_pass` (antes `retrieval_exact_hit_rate`) no lo penaliza."""
    retrieved = ["a.md", "extra-del-grafo.md"]
    expected = ("a.md",)
    recall = 1.0  # a.md está
    leaked = False  # extra-del-grafo.md no está en forbidden_paths
    retrieval_pass = recall == 1.0 and not leaked
    assert retrieval_pass is True
    assert precision_at_k(retrieved, expected) == pytest.approx(0.5)


def test_ndcg_binario_penaliza_orden():
    import math

    retrieved = ["b.md", "a.md"]
    expected = ("a.md",)
    dcg = 1.0 / math.log2(3)
    idcg = 1.0 / math.log2(2)
    assert ndcg_at_k(retrieved, expected) == pytest.approx(dcg / idcg)


def test_ndcg_orden_ideal_da_uno():
    retrieved = ["a.md", "b.md"]
    expected = ("a.md",)
    assert ndcg_at_k(retrieved, expected) == pytest.approx(1.0)


def test_ndcg_no_supera_uno_con_fragmentos_repetidos_del_mismo_documento():
    """RAG_MAX_CHUNKS_PER_DOCUMENT permite varios fragmentos de una misma
    nota; sin deduplicar por documento, contar cada fragmento como un
    acierto de relevancia distinto rompe la cota nDCG<=1."""
    retrieved = ["a.md", "a.md", "a.md", "b.md", "c.md"]
    expected = ("a.md",)
    result = ndcg_at_k(retrieved, expected)
    assert result is not None
    assert result <= 1.0 + 1e-9
    assert result == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 11. Datasets sin negativos muestran abstención como null/N/A.
# ----------------------------------------------------------------------
def test_sin_casos_de_abstencion_metricas_son_none():
    results = [
        make_result(
            query_type="factual",
            expect_abstention=False,
            answered=True,
            retrieval_pass=True,
            document_recall=1.0,
            precision=1.0,
        )
        for _ in range(3)
    ]
    metrics = _core_metrics(results)
    assert metrics["abstention_accuracy"] is None
    assert metrics["abstention_precision"] is None
    assert metrics["abstention_recall"] is None
    assert metrics["abstention_f1"] is None


def test_con_casos_de_abstencion_metricas_son_numericas():
    results = [
        make_result(
            query_type="factual",
            expect_abstention=False,
            answered=True,
            retrieval_pass=True,
            document_recall=1.0,
            precision=1.0,
        ),
        make_result(
            query_type="out_of_domain",
            expect_abstention=True,
            answered=False,
            retrieval_pass=True,
        ),
    ]
    metrics = _core_metrics(results)
    assert metrics["abstention_recall"] == pytest.approx(1.0)
    assert metrics["abstention_precision"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 13. La evaluación informa métricas separadas por tipo de consulta.
# ----------------------------------------------------------------------
def test_summarize_desglosa_por_query_type():
    results = [
        make_result(
            query_type="factual",
            expect_abstention=False,
            answered=True,
            retrieval_pass=True,
            document_recall=1.0,
            precision=1.0,
        ),
        make_result(
            query_type="relational",
            expect_abstention=False,
            answered=True,
            retrieval_pass=False,
            document_recall=0.5,
            precision=0.5,
        ),
    ]
    report = _summarize(results, generate_answers=False, strategy="configured")
    assert set(report["by_query_type"].keys()) == {"factual", "relational"}
    assert report["by_query_type"]["factual"]["retrieval_exact_hit_rate"] == pytest.approx(1.0)
    assert report["by_query_type"]["relational"]["retrieval_exact_hit_rate"] == pytest.approx(0.0)


def test_router_confusion_matrix_y_accuracy():
    results = [
        make_result(
            query_type="relational",
            expect_abstention=False,
            answered=True,
            retrieval_pass=True,
            document_recall=1.0,
            precision=1.0,
            predicted_query_type="relational",
        ),
        make_result(
            query_type="relational",
            expect_abstention=False,
            answered=True,
            retrieval_pass=True,
            document_recall=1.0,
            precision=1.0,
            predicted_query_type="factual",
        ),
        make_result(
            query_type="factual",
            expect_abstention=False,
            answered=True,
            retrieval_pass=True,
            document_recall=1.0,
            precision=1.0,
            predicted_query_type="factual",
        ),
    ]
    confusion = _router_confusion(results)
    assert confusion is not None
    assert confusion["accuracy"] == pytest.approx(2 / 3)
    assert confusion["matrix"]["relational"]["factual"] == 1
    assert confusion["matrix"]["relational"]["relational"] == 1
    assert confusion["matrix"]["factual"]["factual"] == 1


# ----------------------------------------------------------------------
# El informe registra qué modo de troceado lo generó: sin esto, dos
# informes con RAG_PARENT_CHILD_CHUNKING distinto (y una reconstrucción de
# por medio) serían indistinguibles y no se podrían comparar.
# ----------------------------------------------------------------------
def test_summarize_registra_troceado_plano(monkeypatch):
    monkeypatch.setattr(evaluation.settings, "parent_child_chunking_enabled", False)
    monkeypatch.setattr(evaluation.settings, "chunk_size", 1800)
    monkeypatch.setattr(evaluation.settings, "chunk_overlap", 250)

    report = _summarize([], generate_answers=False, strategy="configured")

    assert report["configuration"]["chunking"] == {
        "mode": "flat",
        "chunk_size": 1800,
        "chunk_overlap": 250,
    }


def test_summarize_registra_troceado_padre_hijo(monkeypatch):
    monkeypatch.setattr(evaluation.settings, "parent_child_chunking_enabled", True)
    monkeypatch.setattr(evaluation.settings, "parent_chunk_size", 6000)
    monkeypatch.setattr(evaluation.settings, "child_chunk_size", 700)
    monkeypatch.setattr(evaluation.settings, "child_chunk_overlap", 100)

    report = _summarize([], generate_answers=False, strategy="configured")

    assert report["configuration"]["chunking"] == {
        "mode": "parent_child",
        "parent_chunk_size": 6000,
        "child_chunk_size": 700,
        "child_chunk_overlap": 100,
    }


def test_router_confusion_none_sin_predicciones():
    results = [
        make_result(
            query_type="factual",
            expect_abstention=False,
            answered=True,
            retrieval_pass=True,
            document_recall=1.0,
            precision=1.0,
        )
    ]
    assert _router_confusion(results) is None


# ----------------------------------------------------------------------
# Evaluación de extremo a extremo con `retrieve()` sustituido (sin Ollama
# ni base de datos real).
# ----------------------------------------------------------------------
def test_evaluate_end_to_end_con_retrieve_falso(monkeypatch, tmp_path):
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        """
        {
          "cases": [
            {
              "id": "C1",
              "question": "pregunta factual",
              "expected_paths": ["a.md"],
              "query_type": "factual"
            },
            {
              "id": "C2",
              "question": "fuera de dominio",
              "expect_abstention": true,
              "query_type": "out_of_domain"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    cases = evaluation.load_dataset(dataset)

    def fake_retrieve(question, **kwargs):
        if "factual" in question:
            return [{"path": "a.md", "score": 0.9}]
        return []

    monkeypatch.setattr(evaluation, "retrieve", fake_retrieve)
    report = evaluation.evaluate(cases, generate_answers=False)
    assert report["metrics"]["cases"] == 2
    assert report["metrics"]["precision_at_k"] == pytest.approx(1.0)
    assert report["metrics"]["abstention_recall"] == pytest.approx(1.0)
    assert "factual" in report["by_query_type"]
    assert "out_of_domain" in report["by_query_type"]

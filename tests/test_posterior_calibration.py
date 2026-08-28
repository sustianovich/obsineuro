from __future__ import annotations

import pytest

from app.config import settings
from scripts.calibrate_posterior_threshold import (
    PosteriorObservation,
    calibrate,
    collect_observations,
    evaluate_threshold,
)


def observation(
    case_id: str,
    *,
    abstention: bool,
    score: float | None,
    fixed: bool = False,
    recall: float | None = None,
    reciprocal_rank: float | None = None,
) -> PosteriorObservation:
    return PosteriorObservation(
        case_id=case_id,
        expect_abstention=abstention,
        score=score,
        fixed_abstention=fixed,
        document_recall_when_answered=(
            None if abstention else (1.0 if recall is None else recall)
        ),
        reciprocal_rank_when_answered=(
            None
            if abstention
            else (1.0 if reciprocal_rank is None else reciprocal_rank)
        ),
        stage="pre_retrieval" if fixed else "post_fusion",
    )


def test_recomienda_mayor_umbral_que_preserva_recall_y_mrr():
    observations = [
        observation("P1", abstention=False, score=0.70),
        observation("P2", abstention=False, score=0.80),
        observation("N1", abstention=True, score=0.20),
        observation("N2", abstention=True, score=0.60),
    ]

    result = calibrate(observations)

    assert result["umbral_recomendado"] == pytest.approx(0.65)
    assert result["mejor"]["abstiene_debiendo_responder_FN"] == 0
    assert result["mejor"]["responde_debiendo_abstenerse_FP"] == 0
    assert result["mejor"]["document_recall"] == 1.0
    assert result["mejor"]["mrr"] == 1.0


def test_solape_prioriza_la_puerta_de_recall_sobre_el_coste():
    observations = [
        observation("P1", abstention=False, score=0.70),
        observation("P2", abstention=False, score=0.90),
        observation("N1", abstention=True, score=0.75),
    ]

    result = calibrate(observations, cost_fp=100.0, cost_fn=1.0)

    assert result["umbral_recomendado"] is not None
    assert result["mejor"]["abstiene_debiendo_responder_FN"] == 0
    assert result["mejor"]["responde_debiendo_abstenerse_FP"] == 1
    assert "solapadas" in result["veredicto"]


def test_no_inventa_umbral_sin_negativos_que_lleguen_al_posterior():
    observations = [
        observation("P1", abstention=False, score=0.70),
        observation("N1", abstention=True, score=None, fixed=True),
    ]

    result = calibrate(observations)

    assert result["umbral_recomendado"] is None
    assert "faltan casos" in result["veredicto"]


def test_respeta_minimo_de_muestras_por_clase():
    observations = [
        observation("P1", abstention=False, score=0.70),
        observation("N1", abstention=True, score=0.20),
    ]

    result = calibrate(observations, min_class_size=2)

    assert result["umbral_recomendado"] is None
    assert "al menos 2 por clase" in result["veredicto"]


def test_rechaza_todos_los_umbrales_si_la_recuperacion_base_ya_falla():
    observations = [
        observation("P1", abstention=False, score=0.70, recall=0.5),
        observation("N1", abstention=True, score=0.20),
    ]

    result = calibrate(observations, min_document_recall=1.0)

    assert result["umbral_recomendado"] is None
    assert "ningún umbral preserva" in result["veredicto"]


def test_abstencion_previa_es_fija_para_cualquier_umbral():
    observations = [
        observation("P1", abstention=False, score=0.80),
        observation("N1", abstention=True, score=None, fixed=True),
    ]

    metrics = evaluate_threshold(
        0.0,
        observations,
        cost_fp=3.0,
        cost_fn=1.0,
        min_document_recall=1.0,
        min_mrr=0.938,
    )

    assert metrics.true_negative == 1
    assert metrics.false_positive == 0


def test_recogida_fuerza_modo_observacion_y_restaura_settings(monkeypatch):
    from types import SimpleNamespace

    import app.rag.retrieval as retrieval

    original_enabled = settings.posterior_abstention_enabled
    original_threshold = settings.posterior_abstention_threshold

    def fake_retrieve(question, **kwargs):
        assert settings.posterior_abstention_enabled is True
        assert settings.posterior_abstention_threshold == 0.0
        retrieval.last_abstention_outcome.clear()
        retrieval.last_abstention_outcome.update(
            {
                "stage": "post_fusion",
                "combined_score": 0.72,
                "should_abstain": False,
            }
        )
        return [{"path": "documento.md"}]

    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    case = SimpleNamespace(
        id="P1",
        question="pregunta",
        expected_paths=("documento.md",),
        expect_abstention=False,
        status=None,
        expand_links=False,
        vigencia=None,
        tags=(),
    )

    observations = collect_observations([case])

    assert observations[0].score == pytest.approx(0.72)
    assert observations[0].document_recall_when_answered == 1.0
    assert settings.posterior_abstention_enabled is original_enabled
    assert settings.posterior_abstention_threshold == original_threshold

from __future__ import annotations

import pytest

from app.config import settings
from app.rag.abstention import evaluate_posterior_abstention


def selected_with(**overrides) -> list[dict]:
    base = {
        "semantic_score": 0.9,
        "lexical_score": 0.5,
        "lexical_coverage": 0.8,
        "graph_score": None,
        "fusion_score": 0.9,
        "rerank_score": None,
    }
    base.update(overrides)
    return [base]


def test_desactivada_por_defecto_nunca_abstiene(monkeypatch):
    monkeypatch.setattr(settings, "posterior_abstention_enabled", False)
    decision = evaluate_posterior_abstention([], top_k=5)
    assert decision.should_abstain is False
    assert decision.enabled is False


def test_activada_abstiene_si_no_queda_nada_seleccionado(monkeypatch):
    monkeypatch.setattr(settings, "posterior_abstention_enabled", True)
    decision = evaluate_posterior_abstention([], top_k=5)
    assert decision.should_abstain is True


def test_activada_con_señales_fuertes_no_abstiene(monkeypatch):
    monkeypatch.setattr(settings, "posterior_abstention_enabled", True)
    monkeypatch.setattr(settings, "posterior_abstention_threshold", 0.35)
    selected = [
        {
            "semantic_score": 0.9,
            "lexical_score": 0.6,
            "lexical_coverage": 0.9,
            "graph_score": None,
            "fusion_score": 0.95,
        },
        {
            "semantic_score": 0.7,
            "lexical_score": 0.4,
            "lexical_coverage": 0.6,
            "graph_score": None,
            "fusion_score": 0.5,
        },
    ]
    decision = evaluate_posterior_abstention(selected, top_k=2)
    assert decision.should_abstain is False
    assert decision.combined_score is not None
    assert decision.combined_score >= settings.posterior_abstention_threshold


def test_activada_con_señales_debiles_abstiene(monkeypatch):
    monkeypatch.setattr(settings, "posterior_abstention_enabled", True)
    monkeypatch.setattr(settings, "posterior_abstention_threshold", 0.6)
    selected = [
        {
            "semantic_score": 0.1,
            "lexical_score": None,
            "lexical_coverage": None,
            "graph_score": None,
            "fusion_score": 0.1,
        }
    ]
    decision = evaluate_posterior_abstention(selected, top_k=10)
    assert decision.should_abstain is True
    assert decision.reasons

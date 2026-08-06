"""Abstención posterior a la fusión, basada en señales de recuperación.

Distinta de la abstención previa (RAG_MIN_SIMILARITY, en `retrieval.retrieve`):
esa corta *antes* de ejecutar el grafo, cuando la mejor similitud semántica ya
es insuficiente. Esta capa corre *después* de fusionar, reordenar y
diversificar, y mira el conjunto final completo: puede detectar casos donde
cada rama por separado pasó el umbral pero el acuerdo entre ellas es bajo, el
margen sobre el segundo candidato es mínimo, o el conjunto quedó demasiado
corto para sostener una respuesta.

No existe todavía un conjunto dorado con suficientes etiquetas para ajustar
los pesos por separado (ver objetivo 4 del enrutador adaptativo), así que la
combinación es una media ponderada con pesos iniciales documentados, no una
fórmula validada. Por eso `RAG_POSTERIOR_ABSTENTION` está desactivada por
defecto: la infraestructura existe y es explicable (cada señal y su peso
aparece en `reasons`), pero activarla en producción exige antes calibrarla
con `scripts/calibrate_threshold.py` o equivalente sobre un dataset con
casos positivos y negativos reales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import settings


# Pesos iniciales, sin calibrar. Suman 1 para que `combined_score` quede en
# [0, 1] cuando todas las señales están disponibles; si alguna falta (p.ej.
# no hay reordenador activo) se renormaliza sobre las señales presentes.
SIGNAL_WEIGHTS: dict[str, float] = {
    "best_semantic_score": 0.30,
    "top_margin": 0.15,
    "lexical_coverage": 0.15,
    "branch_agreement": 0.15,
    "rerank_score": 0.15,
    "fragment_sufficiency": 0.10,
}


@dataclass(frozen=True)
class AbstentionSignals:
    best_semantic_score: float | None
    top_margin: float | None
    lexical_coverage: float | None
    branch_agreement: float | None
    rerank_score: float | None
    fragment_sufficiency: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "best_semantic_score": self.best_semantic_score,
            "top_margin": self.top_margin,
            "lexical_coverage": self.lexical_coverage,
            "branch_agreement": self.branch_agreement,
            "rerank_score": self.rerank_score,
            "fragment_sufficiency": self.fragment_sufficiency,
        }


@dataclass(frozen=True)
class AbstentionDecision:
    should_abstain: bool
    enabled: bool
    combined_score: float | None
    threshold: float
    signals: AbstentionSignals
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_abstain": self.should_abstain,
            "enabled": self.enabled,
            "combined_score": self.combined_score,
            "threshold": self.threshold,
            "signals": self.signals.as_dict(),
            "reasons": list(self.reasons),
        }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _branch_agreement(selected: list[dict[str, Any]]) -> float | None:
    if not selected:
        return None
    agreeing = 0
    for item in selected:
        present = sum(
            item.get(key) is not None
            for key in ("semantic_score", "lexical_score", "graph_score")
        )
        if present >= 2:
            agreeing += 1
    return agreeing / len(selected)


def compute_signals(
    selected: list[dict[str, Any]],
    *,
    top_k: int,
) -> AbstentionSignals:
    best_semantic = next(
        (
            float(item["semantic_score"])
            for item in selected
            if item.get("semantic_score") is not None
        ),
        None,
    )
    fusion_scores = [float(item.get("fusion_score", 0.0)) for item in selected]
    top_margin = (
        _clip01(fusion_scores[0] - fusion_scores[1])
        if len(fusion_scores) >= 2
        else None
    )
    lexical_coverages = [
        float(item["lexical_coverage"])
        for item in selected
        if item.get("lexical_coverage") is not None
    ]
    lexical_coverage = max(lexical_coverages) if lexical_coverages else None
    rerank_scores = [
        float(item["rerank_score"])
        for item in selected
        if item.get("rerank_score") is not None
    ]
    rerank_score = (
        _clip01(sum(rerank_scores) / len(rerank_scores))
        if rerank_scores
        else None
    )
    fragment_sufficiency = (
        _clip01(len(selected) / top_k) if top_k > 0 else None
    )
    return AbstentionSignals(
        best_semantic_score=(
            _clip01(best_semantic) if best_semantic is not None else None
        ),
        top_margin=top_margin,
        lexical_coverage=lexical_coverage,
        branch_agreement=_branch_agreement(selected),
        rerank_score=rerank_score,
        fragment_sufficiency=fragment_sufficiency,
    )


def evaluate_posterior_abstention(
    selected: list[dict[str, Any]],
    *,
    top_k: int,
) -> AbstentionDecision:
    """Decide si el conjunto final de fragmentos sostiene una respuesta.

    Devuelve `should_abstain=False` sin evaluar nada cuando
    `RAG_POSTERIOR_ABSTENTION` está desactivada, que es el valor por
    defecto: esta capa nunca cambia el comportamiento salvo que se active
    explícitamente.
    """
    if not settings.posterior_abstention_enabled:
        return AbstentionDecision(
            should_abstain=False,
            enabled=False,
            combined_score=None,
            threshold=settings.posterior_abstention_threshold,
            signals=AbstentionSignals(None, None, None, None, None, None),
            reasons=("desactivada (RAG_POSTERIOR_ABSTENTION=false)",),
        )

    if not selected:
        return AbstentionDecision(
            should_abstain=True,
            enabled=True,
            combined_score=0.0,
            threshold=settings.posterior_abstention_threshold,
            signals=AbstentionSignals(None, None, None, None, None, None),
            reasons=("no quedó ningún fragmento tras diversificar",),
        )

    signals = compute_signals(selected, top_k=top_k)
    present = [
        (name, value)
        for name, value in signals.as_dict().items()
        if value is not None
    ]
    if not present:
        return AbstentionDecision(
            should_abstain=False,
            enabled=True,
            combined_score=None,
            threshold=settings.posterior_abstention_threshold,
            signals=signals,
            reasons=("ninguna señal disponible; no se abstiene por omisión",),
        )

    total_weight = sum(SIGNAL_WEIGHTS[name] for name, _ in present)
    combined_score = sum(
        SIGNAL_WEIGHTS[name] * value for name, value in present
    ) / total_weight
    should_abstain = combined_score < settings.posterior_abstention_threshold

    reasons = [
        f"{name}={value:.3f} (peso {SIGNAL_WEIGHTS[name]:.2f})"
        for name, value in present
    ]
    reasons.append(
        f"combinado={combined_score:.3f} "
        f"{'<' if should_abstain else '>='} "
        f"umbral={settings.posterior_abstention_threshold:.3f}"
    )
    return AbstentionDecision(
        should_abstain=should_abstain,
        enabled=True,
        combined_score=combined_score,
        threshold=settings.posterior_abstention_threshold,
        signals=signals,
        reasons=tuple(reasons),
    )

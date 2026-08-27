"""Calibra el umbral de abstención posterior sobre la recuperación real.

A diferencia de ``scripts.calibrate_threshold``, que calibra el coseno previo
``RAG_MIN_SIMILARITY``, este módulo ejecuta fusión, reordenación y MMR una sola
vez por pregunta y recoge el ``combined_score`` de ``app.rag.abstention``.

Durante la recogida fuerza temporalmente el umbral posterior a 0.00. Así se
observan las señales sin rechazar conjuntos no vacíos y después se recorren
los umbrales en memoria. La configuración global se restaura siempre.

El umbral recomendado debe cumplir tres puertas duras antes de optimizar el
coste ponderado: cero abstenciones de preguntas respondibles, recall
documental mínimo y MRR mínimo. Si faltan positivos/negativos que hayan
superado la abstención semántica, no inventa una recomendación.

    python -m scripts.calibrate_posterior_threshold \
      --dataset evaluations/pdpcm_questions.json \
      --dataset evaluations/pdpcm_abstention_negatives.json --detalle
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class PosteriorObservation:
    case_id: str
    expect_abstention: bool
    score: float | None
    fixed_abstention: bool
    document_recall_when_answered: float | None
    reciprocal_rank_when_answered: float | None
    stage: str


@dataclass(frozen=True)
class PosteriorThresholdMetrics:
    threshold: float
    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int
    cost: float
    balanced_accuracy: float
    abstention_precision: float | None
    abstention_recall: float | None
    coverage: float
    document_recall: float | None
    mean_reciprocal_rank: float | None
    preserves_baseline: bool

    def as_dict(self) -> dict[str, Any]:
        def rounded(value: float | None) -> float | None:
            return round(value, 4) if value is not None else None

        return {
            "umbral": round(self.threshold, 4),
            "responde_correctamente": self.true_positive,
            "abstiene_debiendo_responder_FN": self.false_negative,
            "abstiene_correctamente": self.true_negative,
            "responde_debiendo_abstenerse_FP": self.false_positive,
            "coste": round(self.cost, 3),
            "balanced_accuracy": rounded(self.balanced_accuracy),
            "precision_abstencion": rounded(self.abstention_precision),
            "recall_abstencion": rounded(self.abstention_recall),
            "cobertura": rounded(self.coverage),
            "document_recall": rounded(self.document_recall),
            "mrr": rounded(self.mean_reciprocal_rank),
            "preserva_baseline": self.preserves_baseline,
        }


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def candidate_thresholds(
    observations: Iterable[PosteriorObservation],
) -> list[float]:
    """Umbrales donde puede cambiar la clasificación, limitados a [0, 1]."""
    values = sorted(
        {
            float(observation.score)
            for observation in observations
            if observation.score is not None
            and not observation.fixed_abstention
        }
    )
    if not values:
        return [0.0]
    midpoints = [
        (left + right) / 2.0
        for left, right in zip(values, values[1:])
    ]
    above_maximum = min(1.0, values[-1] + 1e-6)
    return sorted({0.0, *midpoints, above_maximum})


def evaluate_threshold(
    threshold: float,
    observations: list[PosteriorObservation],
    *,
    cost_fp: float,
    cost_fn: float,
    min_document_recall: float,
    min_mrr: float,
) -> PosteriorThresholdMetrics:
    true_positive = false_negative = true_negative = false_positive = 0
    answered = 0
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    for observation in observations:
        should_abstain = observation.fixed_abstention or (
            observation.score is not None
            and observation.score < threshold
        )
        if not should_abstain:
            answered += 1

        if observation.expect_abstention:
            if should_abstain:
                true_negative += 1
            else:
                false_positive += 1
            continue

        if should_abstain:
            false_negative += 1
            recalls.append(0.0)
            reciprocal_ranks.append(0.0)
        else:
            true_positive += 1
            recalls.append(
                float(observation.document_recall_when_answered or 0.0)
            )
            reciprocal_ranks.append(
                float(observation.reciprocal_rank_when_answered or 0.0)
            )

    sensitivity_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    sensitivity = (
        true_positive / sensitivity_denominator
        if sensitivity_denominator
        else None
    )
    specificity = (
        true_negative / specificity_denominator
        if specificity_denominator
        else None
    )
    balanced_parts = [
        value for value in (sensitivity, specificity) if value is not None
    ]
    balanced_accuracy = _average(balanced_parts) or 0.0

    abstention_denominator = true_negative + false_negative
    abstention_precision = (
        true_negative / abstention_denominator
        if abstention_denominator
        else None
    )
    document_recall = _average(recalls)
    mean_reciprocal_rank = _average(reciprocal_ranks)
    preserves_baseline = (
        false_negative == 0
        and document_recall is not None
        and document_recall >= min_document_recall
        and mean_reciprocal_rank is not None
        and mean_reciprocal_rank >= min_mrr
    )

    return PosteriorThresholdMetrics(
        threshold=threshold,
        true_positive=true_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        false_positive=false_positive,
        cost=cost_fp * false_positive + cost_fn * false_negative,
        balanced_accuracy=balanced_accuracy,
        abstention_precision=abstention_precision,
        abstention_recall=specificity,
        coverage=(answered / len(observations) if observations else 0.0),
        document_recall=document_recall,
        mean_reciprocal_rank=mean_reciprocal_rank,
        preserves_baseline=preserves_baseline,
    )


def calibrate(
    observations: list[PosteriorObservation],
    *,
    cost_fp: float = 3.0,
    cost_fn: float = 1.0,
    min_document_recall: float = 1.0,
    min_mrr: float = 0.938,
    min_class_size: int = 1,
) -> dict[str, Any]:
    answerable_scores = [
        float(item.score)
        for item in observations
        if not item.expect_abstention
        and item.score is not None
        and not item.fixed_abstention
    ]
    unanswerable_scores = [
        float(item.score)
        for item in observations
        if item.expect_abstention
        and item.score is not None
        and not item.fixed_abstention
    ]
    table = [
        evaluate_threshold(
            threshold,
            observations,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            min_document_recall=min_document_recall,
            min_mrr=min_mrr,
        )
        for threshold in candidate_thresholds(observations)
    ]

    if (
        len(answerable_scores) < min_class_size
        or len(unanswerable_scores) < min_class_size
    ):
        return {
            "umbral_recomendado": None,
            "margen": None,
            "veredicto": (
                "faltan casos respondibles y no respondibles que superen "
                "la abstención previa: se requieren al menos "
                f"{min_class_size} por clase y hay "
                f"{len(answerable_scores)}/{len(unanswerable_scores)}"
            ),
            "mejor": None,
            "tabla": [entry.as_dict() for entry in table],
        }

    safe = [entry for entry in table if entry.preserves_baseline]
    margin = min(answerable_scores) - max(unanswerable_scores)
    if not safe:
        return {
            "umbral_recomendado": None,
            "margen": round(margin, 4),
            "veredicto": (
                "ningún umbral preserva simultáneamente cero falsos "
                "negativos, recall documental y MRR"
            ),
            "mejor": None,
            "tabla": [entry.as_dict() for entry in table],
        }

    best = min(
        safe,
        key=lambda entry: (
            entry.cost,
            -entry.balanced_accuracy,
            -entry.threshold,
        ),
    )
    separation = (
        "separación amplia"
        if margin >= 0.08
        else (
            "separación estrecha"
            if margin > 0
            else "distribuciones solapadas"
        )
    )
    return {
        "umbral_recomendado": round(best.threshold, 4),
        "margen": round(margin, 4),
        "veredicto": separation,
        "mejor": best.as_dict(),
        "tabla": [entry.as_dict() for entry in table],
    }


def collect_observations(cases: list[Any]) -> list[PosteriorObservation]:
    """Ejecuta una pasada en modo observación y conserva sus scores."""
    from app.config import settings
    from app.rag.retrieval import last_abstention_outcome, normalize, retrieve

    previous_enabled = settings.posterior_abstention_enabled
    previous_threshold = settings.posterior_abstention_threshold
    settings.posterior_abstention_enabled = True
    settings.posterior_abstention_threshold = 0.0
    observations: list[PosteriorObservation] = []
    try:
        for case in cases:
            last_abstention_outcome.clear()
            hits = retrieve(
                case.question,
                top_k=settings.top_k,
                min_similarity=settings.min_similarity,
                status=case.status,
                expand_links=case.expand_links,
                vigencia=case.vigencia,
                tags=list(case.tags),
            )
            decision = dict(last_abstention_outcome)
            stage = str(decision.get("stage") or "unknown")
            raw_score = decision.get("combined_score")
            score = float(raw_score) if raw_score is not None else None
            fixed_abstention = stage == "pre_retrieval" or bool(
                decision.get("should_abstain")
            )

            retrieved_paths = [normalize(hit["path"]) for hit in hits]
            if case.expect_abstention:
                document_recall = reciprocal_rank = None
            else:
                matched = set(retrieved_paths).intersection(case.expected_paths)
                document_recall = len(matched) / len(case.expected_paths)
                ranks = [
                    retrieved_paths.index(path) + 1
                    for path in case.expected_paths
                    if path in retrieved_paths
                ]
                reciprocal_rank = 1.0 / min(ranks) if ranks else 0.0

            observations.append(
                PosteriorObservation(
                    case_id=case.id,
                    expect_abstention=case.expect_abstention,
                    score=score,
                    fixed_abstention=fixed_abstention,
                    document_recall_when_answered=document_recall,
                    reciprocal_rank_when_answered=reciprocal_rank,
                    stage=stage,
                )
            )
    finally:
        settings.posterior_abstention_enabled = previous_enabled
        settings.posterior_abstention_threshold = previous_threshold
    return observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        help=(
            "Dataset etiquetado; se puede repetir para combinar positivos "
            "y negativos revisados del mismo vault."
        ),
    )
    parser.add_argument("--cost-fp", type=float, default=3.0)
    parser.add_argument("--cost-fn", type=float, default=1.0)
    parser.add_argument("--min-document-recall", type=float, default=1.0)
    parser.add_argument("--min-mrr", type=float, default=0.938)
    parser.add_argument(
        "--min-class-size",
        type=int,
        default=10,
        help=(
            "Mínimo de casos de cada clase que deben alcanzar la etapa "
            "posterior (por defecto 10)."
        ),
    )
    parser.add_argument("--detalle", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    from app.config import settings
    from app.rag.evaluation import load_dataset
    from app.rag.vector_store import vector_store

    dataset_paths = [path.resolve() for path in args.dataset]
    cases = []
    seen_case_ids: set[str] = set()
    for dataset_path in dataset_paths:
        for case in load_dataset(dataset_path):
            if case.id in seen_case_ids:
                print(
                    f"ID de caso duplicado entre datasets: {case.id}",
                    file=sys.stderr,
                )
                return 2
            seen_case_ids.add(case.id)
            cases.append(case)
    vector_store.ensure_loaded()
    stats = vector_store.stats()
    if not stats["loaded_chunks"]:
        print(
            "El índice está vacío; indexa el vault antes de calibrar.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Vault: {settings.vault_path} · {stats['loaded_chunks']} fragmentos · "
        f"{len(cases)} casos"
    )
    observations = collect_observations(cases)
    result = calibrate(
        observations,
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn,
        min_document_recall=args.min_document_recall,
        min_mrr=args.min_mrr,
        min_class_size=max(1, args.min_class_size),
    )

    if args.detalle:
        for observation in observations:
            score = (
                "N/A"
                if observation.score is None
                else f"{observation.score:.4f}"
            )
            label = "abstener" if observation.expect_abstention else "responder"
            print(
                f"  {observation.case_id:<42} {label:<10} "
                f"score={score:<7} etapa={observation.stage}"
            )
        print()

    print(f"Umbral posterior recomendado : {result['umbral_recomendado']}")
    print(f"Margen                      : {result['margen']}")
    print(f"Veredicto                   : {result['veredicto']}")
    if result["mejor"]:
        best = result["mejor"]
        print(
            "Recall / MRR               : "
            f"{best['document_recall']} / {best['mrr']}"
        )
        print(
            "Precisión / recall abstención: "
            f"{best['precision_abstencion']} / {best['recall_abstencion']}"
        )

    payload = {
        "datasets": [str(path) for path in dataset_paths],
        "vault": str(settings.vault_path),
        "configuration": {
            "top_k": settings.top_k,
            "min_similarity": settings.min_similarity,
            "cost_fp": args.cost_fp,
            "cost_fn": args.cost_fn,
            "min_document_recall": args.min_document_recall,
            "min_mrr": args.min_mrr,
            "min_class_size": max(1, args.min_class_size),
        },
        "observations": [observation.__dict__ for observation in observations],
        "calibration": result,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Informe: {args.output}")
    return 0 if result["umbral_recomendado"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())

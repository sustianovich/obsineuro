"""Calibra `RAG_MIN_SIMILARITY` para un vault y un modelo concretos.

La escala absoluta del coseno depende del modelo de embedding. Un 0,30
que abstiene bien con `nomic-embed-text` puede dejar pasar todo con otro
modelo, o callar demasiado. No hay un valor universal, así que en lugar
de adivinarlo se mide.

El script necesita preguntas etiquetadas: unas que el vault **sí** puede
responder y otras que **no**. Con ellas calcula la mejor puntuación de
cada una y recorre un grid de umbrales candidatos, no sólo el punto medio.

La versión anterior minimizaba el número bruto de errores, lo que favorece
a la clase mayoritaria cuando el dataset está desequilibrado (p.ej. muchas
más preguntas respondibles que de abstención): un umbral que ignora todas
las de abstención puede "ganar" en errores totales aunque sea inútil en
producción. Esta versión pondera dos tipos de error por separado y deja el
coste configurable:

  - falso positivo: responder cuando debía abstenerse (`--cost-fp`);
  - falso negativo: abstenerse ante una pregunta respondible (`--cost-fn`).

Por defecto `--cost-fp` > `--cost-fn`: una respuesta incorrecta pesa más
que una abstención de más, pero ambos costes son ajustables porque esa
prioridad depende del dominio.

    python -m scripts.calibrate_threshold
    python -m scripts.calibrate_threshold --dataset evaluations/questions.json
    python -m scripts.calibrate_threshold --cost-fp 5 --cost-fn 1 --detalle
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def best_score(matrix: np.ndarray, vector: np.ndarray) -> float:
    if not matrix.size:
        return 0.0
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return 0.0
    return float(np.max(matrix @ (vector / norm)))


def load_questions(dataset: Path) -> tuple[list[str], list[str]]:
    """Separa el conjunto dorado en respondibles y no respondibles."""
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    answerable: list[str] = []
    unanswerable: list[str] = []
    for case in raw.get("cases", []):
        question = str(case.get("question", "")).strip()
        if not question:
            continue
        if case.get("expect_abstention"):
            unanswerable.append(question)
        else:
            answerable.append(question)
    return answerable, unanswerable


@dataclass(frozen=True)
class ThresholdMetrics:
    """Matriz de confusión de un umbral candidato.

    "Positivo" = el sistema responde (score >= umbral). Los nombres siguen
    la semántica del objetivo, no la de la clase mayoritaria:
    `false_positive` es "respondió debiendo abstenerse" y `false_negative`
    es "abstuvo debiendo responder", tal cual se pidió calibrar.
    """

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
    accuracy_given_answered: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "umbral": round(self.threshold, 4),
            "responde_correctamente": self.true_positive,
            "abstiene_debiendo_responder_FN": self.false_negative,
            "abstiene_correctamente": self.true_negative,
            "responde_debiendo_abstenerse_FP": self.false_positive,
            "coste": round(self.cost, 3),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "precision_abstencion": (
                round(self.abstention_precision, 4)
                if self.abstention_precision is not None
                else None
            ),
            "recall_abstencion": (
                round(self.abstention_recall, 4)
                if self.abstention_recall is not None
                else None
            ),
            "cobertura": round(self.coverage, 4),
            "exactitud_si_responde": (
                round(self.accuracy_given_answered, 4)
                if self.accuracy_given_answered is not None
                else None
            ),
        }


def evaluate_threshold(
    threshold: float,
    answerable_scores: list[float],
    unanswerable_scores: list[float],
    *,
    cost_fp: float,
    cost_fn: float,
) -> ThresholdMetrics:
    true_positive = sum(1 for score in answerable_scores if score >= threshold)
    false_negative = len(answerable_scores) - true_positive
    true_negative = sum(1 for score in unanswerable_scores if score < threshold)
    false_positive = len(unanswerable_scores) - true_negative

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else None
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else None
    )
    present = [value for value in (sensitivity, specificity) if value is not None]
    balanced_accuracy = sum(present) / len(present) if present else 0.0

    abstention_precision = (
        true_negative / (true_negative + false_negative)
        if (true_negative + false_negative) > 0
        else None
    )
    abstention_recall = specificity

    total = len(answerable_scores) + len(unanswerable_scores)
    answered = true_positive + false_positive
    coverage = answered / total if total > 0 else 0.0
    accuracy_given_answered = (
        true_positive / answered if answered > 0 else None
    )

    cost = cost_fp * false_positive + cost_fn * false_negative

    return ThresholdMetrics(
        threshold=threshold,
        true_positive=true_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        false_positive=false_positive,
        cost=cost,
        balanced_accuracy=balanced_accuracy,
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
        coverage=coverage,
        accuracy_given_answered=accuracy_given_answered,
    )


def candidate_thresholds(
    answerable_scores: list[float],
    unanswerable_scores: list[float],
) -> list[float]:
    """Puntos medios entre valores consecutivos: son los únicos umbrales
    donde la clasificación puede cambiar, así que basta evaluarlos a ellos."""
    values = sorted(set(answerable_scores + unanswerable_scores))
    if not values:
        return [0.0]
    midpoints = [
        (left + right) / 2 for left, right in zip(values, values[1:])
    ]
    return [values[0] - 1e-6, *midpoints, values[-1] + 1e-6]


def calibrate(
    answerable_scores: list[float],
    unanswerable_scores: list[float],
    *,
    cost_fp: float,
    cost_fn: float,
) -> dict[str, object]:
    """Recorre el grid de umbrales y recomienda el de menor coste ponderado.

    No asume que exista un umbral perfecto: si el margen entre el mínimo
    respondible y el máximo no respondible es negativo o nulo, lo informa
    explícitamente en `veredicto` en vez de fingir una separación limpia.
    """
    if not answerable_scores or not unanswerable_scores:
        return {
            "umbral": 0.0,
            "coste": 0.0,
            "margen": 0.0,
            "veredicto": "faltan preguntas etiquetadas de ambos tipos",
            "tabla": [],
        }

    table = [
        evaluate_threshold(
            threshold,
            answerable_scores,
            unanswerable_scores,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )
        for threshold in candidate_thresholds(answerable_scores, unanswerable_scores)
    ]

    best = min(
        table,
        key=lambda entry: (entry.cost, -entry.balanced_accuracy),
    )

    lowest_answerable = min(answerable_scores)
    highest_unanswerable = max(unanswerable_scores)
    margin = lowest_answerable - highest_unanswerable

    if margin <= 0:
        veredicto = (
            "distribuciones solapadas: ningún umbral acierta en todo "
            f"(mejor coste ponderado: {best.cost:.2f} con "
            f"cost_fp={cost_fp}, cost_fn={cost_fn})"
        )
    elif margin >= 0.08:
        veredicto = "separación amplia"
    else:
        veredicto = "separación estrecha: el umbral será frágil"

    return {
        "umbral": round(best.threshold, 3),
        "coste": round(best.cost, 3),
        "margen": round(margin, 3),
        "veredicto": veredicto,
        "mejor": best.as_dict(),
        "tabla": [entry.as_dict() for entry in table],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evaluations" / "questions.json",
    )
    parser.add_argument(
        "--detalle",
        action="store_true",
        help="Muestra la puntuación de cada pregunta y la tabla completa.",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=3.0,
        help="Coste de responder cuando debía abstenerse (por defecto 3.0).",
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=1.0,
        help="Coste de abstenerse ante una pregunta respondible (por defecto 1.0).",
    )
    args = parser.parse_args()

    from app.config import settings
    from app.rag.retrieval import embed_question
    from app.rag.vector_store import vector_store

    vector_store.ensure_loaded()
    stats = vector_store.stats()
    if not stats["loaded_chunks"]:
        print(
            "El índice está vacío. Ejecuta la indexación antes de "
            "calibrar.",
            file=sys.stderr,
        )
        return 2

    answerable, unanswerable = load_questions(args.dataset)
    print(
        f"Modelo: {settings.embedding_model} · "
        f"{stats['loaded_chunks']} fragmentos · "
        f"{len(answerable)} respondibles / {len(unanswerable)} no · "
        f"cost_fp={args.cost_fp} cost_fn={args.cost_fn}"
    )
    print()

    matrix = vector_store.vectors_for_rows(range(stats["loaded_chunks"]))

    answerable_scores: list[float] = []
    unanswerable_scores: list[float] = []

    for question in answerable:
        score = best_score(matrix, embed_question(question))
        answerable_scores.append(score)
        if args.detalle:
            print(f"  respondible   {score:.3f}  {question[:56]}")

    for question in unanswerable:
        score = best_score(matrix, embed_question(question))
        unanswerable_scores.append(score)
        if args.detalle:
            print(f"  no respondible {score:.3f}  {question[:56]}")

    if args.detalle:
        print()

    def summary(label: str, scores: list[float]) -> None:
        if not scores:
            return
        array = np.array(scores)
        print(
            f"  {label:<16} min={array.min():.3f}  "
            f"mediana={np.median(array):.3f}  max={array.max():.3f}"
        )

    summary("respondibles", answerable_scores)
    summary("no respondibles", unanswerable_scores)
    print()

    result = calibrate(
        answerable_scores,
        unanswerable_scores,
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn,
    )
    print(f"  RAG_MIN_SIMILARITY recomendado : {result['umbral']}")
    print(f"  coste ponderado en el óptimo   : {result['coste']}")
    print(f"  margen de separación           : {result['margen']}")
    print(f"  veredicto                      : {result['veredicto']}")
    if "mejor" in result:
        mejor = result["mejor"]
        print(
            f"  balanced accuracy              : {mejor['balanced_accuracy']}"
        )
        print(
            f"  precision / recall abstención  : "
            f"{mejor['precision_abstencion']} / {mejor['recall_abstencion']}"
        )
        print(f"  cobertura en el óptimo         : {mejor['cobertura']}")
    print()
    print(
        "  Valor actual en configuración  : "
        f"{settings.min_similarity}"
    )

    if args.detalle and result.get("tabla"):
        print()
        print("  cobertura vs. exactitud por umbral candidato:")
        for row in result["tabla"]:
            print(
                f"    umbral={row['umbral']:<8} coste={row['coste']:<6} "
                f"cobertura={row['cobertura']:<6} "
                f"exactitud_si_responde={row['exactitud_si_responde']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

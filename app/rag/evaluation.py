from __future__ import annotations

import argparse
import json
import math
import sys
import time
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.rag.agents import run_rag_agent_pipeline
from app.rag.query_routing import RetrievalPolicy, policy_for_query_type
from app.rag.retrieval import last_routing_outcome, normalize, retrieve


QUERY_TYPES = ("factual", "relational", "hybrid", "out_of_domain")
STRATEGIES = ("configured", "graph_off", "graph_on_1hop", "oracle", "router_real")


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    expected_paths: tuple[str, ...]
    expect_abstention: bool
    expected_terms: tuple[tuple[str, ...], ...]
    query_type: str
    status: str | None = None
    expand_links: bool = True
    vigencia: str | None = None
    tags: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    requires_semantic: bool = False


def load_dataset(path: Path) -> list[EvaluationCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("El dataset debe contener una lista no vacía 'cases'.")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"El caso {position} no es un objeto.")
        case_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        if not case_id or not question:
            raise ValueError(f"El caso {position} necesita id y question.")
        if case_id in seen_ids:
            raise ValueError(f"ID de caso duplicado: {case_id}")
        seen_ids.add(case_id)

        expected_paths = tuple(
            normalize(str(value))
            for value in item.get("expected_paths", [])
            if str(value).strip()
        )
        expect_abstention = bool(item.get("expect_abstention", False))
        if expect_abstention and expected_paths:
            raise ValueError(
                f"{case_id}: una abstención no puede tener expected_paths."
            )
        if not expect_abstention and not expected_paths:
            raise ValueError(
                f"{case_id}: indica expected_paths o expect_abstention=true."
            )

        term_groups: list[tuple[str, ...]] = []
        for group in item.get("expected_terms", []):
            values = [group] if isinstance(group, str) else group
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"{case_id}: cada grupo de expected_terms debe tener texto."
                )
            term_groups.append(tuple(str(value) for value in values))

        # Sin etiqueta explícita, se infiere el tipo más conservador posible
        # para no romper datasets existentes que no la incluyen todavía.
        query_type = str(
            item.get(
                "query_type",
                "out_of_domain" if expect_abstention else "factual",
            )
        ).strip()
        if query_type not in QUERY_TYPES:
            raise ValueError(
                f"{case_id}: query_type debe ser uno de {QUERY_TYPES}, "
                f"recibido {query_type!r}."
            )

        cases.append(
            EvaluationCase(
                id=case_id,
                question=question,
                expected_paths=expected_paths,
                expect_abstention=expect_abstention,
                expected_terms=tuple(term_groups),
                query_type=query_type,
                status=item.get("status"),
                expand_links=bool(item.get("expand_links", True)),
                vigencia=item.get("vigencia"),
                tags=tuple(
                    str(value) for value in item.get("tags", [])
                ),
                forbidden_paths=tuple(
                    normalize(str(value))
                    for value in item.get("forbidden_paths", [])
                    if str(value).strip()
                ),
                requires_semantic=bool(item.get("requires_semantic", False)),
            )
        )
    return cases


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def term_coverage(
    answer: str,
    expected_terms: tuple[tuple[str, ...], ...],
) -> float | None:
    if not expected_terms:
        return None
    normalized_answer = normalize_text(answer)
    matched = sum(
        any(normalize_text(term) in normalized_answer for term in group)
        for group in expected_terms
    )
    return matched / len(expected_terms)


def precision_at_k(retrieved_paths: list[str], expected_paths: tuple[str, ...]) -> float | None:
    """Penaliza cualquier documento de más, no sólo los `forbidden_paths`.

    `retrieval_exact_hit_rate` (antes `retrieval_pass`) sólo exige recall
    completo y ausencia de rutas explícitamente prohibidas; no penaliza un
    vecino de grafo añadido que no venía a cuento. Precision@k sí lo hace.
    """
    if not retrieved_paths:
        return None
    matched = sum(1 for path in retrieved_paths if path in expected_paths)
    return matched / len(retrieved_paths)


def ndcg_at_k(retrieved_paths: list[str], expected_paths: tuple[str, ...]) -> float | None:
    """nDCG con relevancia binaria a nivel de documento: 1 si la ruta está
    en expected_paths, contada una sola vez aunque varios fragmentos de la
    misma nota aparezcan en la lista (`RAG_MAX_CHUNKS_PER_DOCUMENT` permite
    varios por documento). Sin deduplicar, un documento relevante repetido
    tres veces sumaría DCG tres veces mientras que IDCG sólo lo cuenta una,
    dando un nDCG mayor que 1.
    """
    if not retrieved_paths or not expected_paths:
        return None
    seen_paths: set[str] = set()
    deduplicated = []
    for path in retrieved_paths:
        if path not in seen_paths:
            seen_paths.add(path)
            deduplicated.append(path)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, path in enumerate(deduplicated, start=1)
        if path in expected_paths
    )
    ideal_hits = min(len(expected_paths), len(deduplicated))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else None


def _forced_policy_for_strategy(
    strategy: str,
    case: EvaluationCase,
) -> tuple[RetrievalPolicy | None, bool]:
    """Devuelve (forced_policy, force_routing) para `retrieve()`.

    "configured" reproduce exactamente el comportamiento previo (respeta
    `settings` tal cual, sin forzar nada): es el valor por defecto y no
    cambia ninguna evaluación ya existente que no pase `--strategy`.
    """
    if strategy == "configured":
        return None, False
    if strategy == "graph_off":
        baseline = RetrievalPolicy.from_settings()
        return (
            replace(
                baseline,
                query_type="forced_graph_off",
                use_graph=False,
                graph_weight=0.0,
                reasons=("estrategia de evaluación: grafo siempre apagado",),
            ),
            False,
        )
    if strategy == "graph_on_1hop":
        baseline = RetrievalPolicy.from_settings()
        return (
            replace(
                baseline,
                query_type="forced_graph_on_1hop",
                use_graph=True,
                graph_max_hops=1,
                graph_weight=(
                    settings.hybrid_graph_weight
                    or settings.query_router_relational_graph_weight
                ),
                reasons=(
                    "estrategia de evaluación: grafo siempre activo a 1 salto",
                ),
            ),
            False,
        )
    if strategy == "oracle":
        return policy_for_query_type(case.query_type), False
    if strategy == "router_real":
        return None, True
    raise ValueError(f"Estrategia de evaluación desconocida: {strategy}")


def evaluate(
    cases: list[EvaluationCase],
    *,
    generate_answers: bool = False,
    strategy: str = "configured",
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy debe ser uno de {STRATEGIES}")

    results: list[dict[str, Any]] = []

    for case in cases:
        forced_policy, force_routing = _forced_policy_for_strategy(strategy, case)

        started = time.perf_counter()
        hits = retrieve(
            case.question,
            top_k=settings.top_k,
            min_similarity=settings.min_similarity,
            status=case.status,
            expand_links=case.expand_links,
            vigencia=case.vigencia,
            tags=list(case.tags),
            forced_policy=forced_policy,
            force_routing=force_routing,
        )
        latency_seconds = time.perf_counter() - started
        # `retrieve()` no expone la política que aplicó; se lee el
        # diagnóstico del propio módulo justo después de la llamada. La
        # evaluación es secuencial, así que no hay condición de carrera
        # aquí (a diferencia de servir peticiones concurrentes reales).
        predicted_query_type = last_routing_outcome.get("query_type")

        retrieved_paths = [normalize(hit["path"]) for hit in hits]
        scores = [round(float(hit["score"]), 4) for hit in hits]

        leaked = sorted(
            set(retrieved_paths).intersection(case.forbidden_paths)
        )

        if case.expect_abstention:
            retrieval_pass = not hits
            recall = None
            reciprocal_rank = None
            precision = None
            ndcg = None
        else:
            matched = set(retrieved_paths).intersection(case.expected_paths)
            recall = len(matched) / len(case.expected_paths)
            ranks = [
                retrieved_paths.index(path) + 1
                for path in case.expected_paths
                if path in retrieved_paths
            ]
            reciprocal_rank = 1 / min(ranks) if ranks else 0.0
            retrieval_pass = recall == 1.0 and not leaked
            precision = precision_at_k(retrieved_paths, case.expected_paths)
            ndcg = ndcg_at_k(retrieved_paths, case.expected_paths)

        agent_metrics = None
        if generate_answers:
            agent_result = run_rag_agent_pipeline(
                case.question,
                hits,
                conversation_memory="",
                project_memory="",
                verification_enabled=True,
                verifier_context_tokens=(
                    settings.default_verifier_context_tokens
                ),
                writer_context_tokens=(
                    settings.default_writer_context_tokens
                ),
            )
            answer = agent_result.answer
            agent_metrics = agent_result.metrics
        else:
            answer = None
        coverage = (
            term_coverage(answer or "", case.expected_terms)
            if generate_answers and not case.expect_abstention
            else None
        )
        answer_pass = (
            coverage == 1.0
            if coverage is not None
            else (not hits if generate_answers and case.expect_abstention else None)
        )

        results.append(
            {
                "id": case.id,
                "question": case.question,
                "query_type": case.query_type,
                "predicted_query_type": predicted_query_type,
                "expected_paths": list(case.expected_paths),
                "expect_abstention": case.expect_abstention,
                "retrieved_paths": retrieved_paths,
                "scores": scores,
                "retrieval_pass": retrieval_pass,
                "leaked_paths": leaked,
                "document_recall": recall,
                "precision_at_k": precision,
                "ndcg_at_k": ndcg,
                "reciprocal_rank": reciprocal_rank,
                "answer": answer,
                "agent_metrics": agent_metrics,
                "term_coverage": coverage,
                "answer_pass": answer_pass,
                "latency_seconds": round(latency_seconds, 4),
                "answered": bool(hits),
            }
        )

    return _summarize(results, generate_answers=generate_answers, strategy=strategy)


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _core_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Métricas núcleo (recall/precision/MRR/nDCG/ruido) sobre un subconjunto.

    Reutilizada tanto para el informe global como para el desglose por
    `query_type`.
    """
    answerable = [result for result in results if not result["expect_abstention"]]
    unanswerable = [result for result in results if result["expect_abstention"]]

    true_positive = sum(
        1 for result in unanswerable if result["retrieval_pass"]
    )
    false_negative = len(unanswerable) - true_positive
    false_positive = sum(
        1 for result in answerable if not result["answered"]
    )

    abstention_precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else None
    )
    abstention_recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else None
    )
    abstention_f1 = (
        2 * abstention_precision * abstention_recall
        / (abstention_precision + abstention_recall)
        if abstention_precision is not None
        and abstention_recall is not None
        and (abstention_precision + abstention_recall) > 0
        else None
    )

    answered_answerable = [result for result in answerable if result["answered"]]

    return {
        "cases": len(results),
        "answerable_cases": len(answerable),
        "abstention_cases": len(unanswerable),
        "retrieval_exact_hit_rate": _average(
            [float(result["retrieval_pass"]) for result in answerable]
        ),
        "document_recall": _average(
            [float(result["document_recall"]) for result in answerable]
        ),
        "precision_at_k": _average(
            [
                float(result["precision_at_k"])
                for result in answerable
                if result["precision_at_k"] is not None
            ]
        ),
        "ndcg_at_k": _average(
            [
                float(result["ndcg_at_k"])
                for result in answerable
                if result["ndcg_at_k"] is not None
            ]
        ),
        "mean_reciprocal_rank": _average(
            [float(result["reciprocal_rank"]) for result in answerable]
        ),
        # Se conserva por compatibilidad: es el recall de la abstención
        # (TP / (TP+FN)), no una accuracy binaria completa. `null` cuando
        # el dataset no trae ningún caso de abstención, en vez de un 0%
        # que sugeriría "falla siempre".
        "abstention_accuracy": abstention_recall,
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "abstention_f1": abstention_f1,
        "coverage": _average(
            [float(result["answered"]) for result in results]
        ),
        "accuracy_given_answered": _average(
            [float(result["retrieval_pass"]) for result in answered_answerable]
        ),
        "mean_latency_seconds": _average(
            [float(result["latency_seconds"]) for result in results]
        ),
    }


def _router_confusion(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Sólo tiene sentido cuando el router realmente se ejecutó para cada
    caso (predicted_query_type no es None) y el dataset trae `query_type`
    de referencia."""
    labeled = [
        result
        for result in results
        if result.get("predicted_query_type") is not None
    ]
    if not labeled:
        return None
    matrix: dict[str, dict[str, int]] = {}
    correct = 0
    for result in labeled:
        expected = result["query_type"]
        predicted = result["predicted_query_type"]
        matrix.setdefault(expected, {}).setdefault(predicted, 0)
        matrix[expected][predicted] += 1
        if expected == predicted:
            correct += 1
    return {
        "accuracy": correct / len(labeled),
        "labeled_cases": len(labeled),
        "matrix": matrix,
    }


def _summarize(
    results: list[dict[str, Any]],
    *,
    generate_answers: bool,
    strategy: str,
) -> dict[str, Any]:
    answerable = [result for result in results if not result["expect_abstention"]]
    evaluated_answers = [
        result for result in answerable if result["term_coverage"] is not None
    ]

    metrics = _core_metrics(results)
    metrics["answer_term_coverage"] = (
        _average([float(result["term_coverage"]) for result in evaluated_answers])
        if evaluated_answers
        else None
    )

    by_query_type: dict[str, Any] = {}
    for query_type in {result["query_type"] for result in results}:
        subset = [result for result in results if result["query_type"] == query_type]
        by_query_type[query_type] = _core_metrics(subset)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "configuration": {
            "embedding_model": settings.embedding_model,
            "chat_model": settings.chat_model if generate_answers else None,
            "top_k": settings.top_k,
            "min_similarity": settings.min_similarity,
            "retrieval_mode": (
                (
                    "hybrid+graph"
                    if settings.graph_search_enabled
                    else "hybrid"
                )
                if settings.hybrid_search_enabled
                else (
                    "semantic+graph"
                    if settings.graph_search_enabled
                    else "semantic"
                )
            ),
            "hybrid_rrf_k": settings.hybrid_rrf_k,
            "hybrid_semantic_weight": (
                settings.hybrid_semantic_weight
            ),
            "hybrid_lexical_weight": settings.hybrid_lexical_weight,
            "graph_search_enabled": settings.graph_search_enabled,
            "hybrid_graph_weight": settings.hybrid_graph_weight,
            "graph_max_hops": settings.graph_max_hops,
            "graph_decay": settings.graph_decay,
            "graph_backlink_weight": settings.graph_backlink_weight,
            "graph_seed_documents": settings.graph_seed_documents,
            "graph_max_candidates": settings.graph_max_candidates,
            "query_routing_enabled": settings.query_routing_enabled,
            "answers_generated": generate_answers,
            "verification_enabled": generate_answers,
            # Registrado explícitamente porque el modo de troceado se fija al
            # indexar, no en cada consulta: sin esto, dos informes generados
            # con `RAG_PARENT_CHILD_CHUNKING` distinto (y una reconstrucción
            # de por medio) son indistinguibles entre sí, y no se puede
            # comparar su calidad como ya se hace con las estrategias de
            # grafo en compare_strategies().
            "chunking": (
                {
                    "mode": "parent_child",
                    "parent_chunk_size": settings.parent_chunk_size,
                    "child_chunk_size": settings.child_chunk_size,
                    "child_chunk_overlap": settings.child_chunk_overlap,
                }
                if settings.parent_child_chunking_enabled
                else {
                    "mode": "flat",
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                }
            ),
        },
        "metrics": metrics,
        "by_query_type": by_query_type,
        "router_confusion": _router_confusion(results),
        "results": results,
    }


def compare_strategies(
    cases: list[EvaluationCase],
    *,
    strategies: tuple[str, ...] = (
        "graph_off",
        "graph_on_1hop",
        "oracle",
        "router_real",
    ),
) -> dict[str, Any]:
    """Compara estrategias sobre el mismo dataset (objetivo 5).

    Si ni "oracle" (enrutador perfecto) mejora document_recall/MRR de las
    preguntas relacionales frente a "graph_off", el clasificador real no
    tiene techo que alcanzar y no vale la pena activarlo.
    """
    reports = {strategy: evaluate(cases, strategy=strategy) for strategy in strategies}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategies": {
            strategy: {
                "metrics": report["metrics"],
                "by_query_type": report["by_query_type"],
                "router_confusion": report["router_confusion"],
            }
            for strategy, report in reports.items()
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]

    def percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.1%}"

    chunking = report.get("configuration", {}).get("chunking", {})
    if chunking.get("mode") == "parent_child":
        chunking_line = (
            f"padre-hijo (padres {chunking.get('parent_chunk_size')}, "
            f"hijos {chunking.get('child_chunk_size')}"
            f"/{chunking.get('child_chunk_overlap')})"
        )
    else:
        chunking_line = (
            f"plano ({chunking.get('chunk_size')}"
            f"/{chunking.get('chunk_overlap')})"
        )

    lines = [
        "# Evaluación automática del RAG",
        "",
        f"- Estrategia: {report.get('strategy', 'configured')}",
        f"- Troceado: {chunking_line}",
        f"- Casos: {metrics['cases']}",
        f"- Acierto de recuperación: "
        f"{percent(metrics['retrieval_exact_hit_rate'])}",
        f"- Recall documental: {percent(metrics['document_recall'])}",
        f"- Precision@k: {percent(metrics['precision_at_k'])}",
        f"- nDCG@k: {percent(metrics['ndcg_at_k'])}",
        f"- MRR: {metrics['mean_reciprocal_rank']:.3f}"
        if metrics["mean_reciprocal_rank"] is not None
        else "- MRR: N/A",
        f"- Cobertura del sistema (no abstuvo): {percent(metrics['coverage'])}",
        f"- Precisión de abstención: {percent(metrics['abstention_precision'])}",
        f"- Recall de abstención: {percent(metrics['abstention_recall'])}",
        f"- Cobertura de términos en respuestas: "
        f"{percent(metrics['answer_term_coverage'])}",
        "",
        "| Caso | Tipo | Recuperación | Recall | Precision@k | Primer acierto |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        recall = (
            "—"
            if result["document_recall"] is None
            else percent(result["document_recall"])
        )
        precision = (
            "—"
            if result["precision_at_k"] is None
            else percent(result["precision_at_k"])
        )
        rank = (
            "—"
            if result["reciprocal_rank"] is None
            else f"{result['reciprocal_rank']:.3f}"
        )
        passed = "[OK]" if result["retrieval_pass"] else "[FAIL]"
        lines.append(
            f"| {result['id']} | {result['query_type']} | {passed} | "
            f"{recall} | {precision} | {rank} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evalúa la recuperación y, opcionalmente, las respuestas."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=settings.base_dir / "evaluations" / "questions.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.base_dir / "evaluations" / "reports",
    )
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Evalúa también respuestas; consume llamadas al modelo de chat.",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="configured",
        help=(
            "'configured' respeta settings tal cual (comportamiento previo). "
            "Las demás fuerzan una política sin tocar settings."
        ),
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help=(
            "Ignora --strategy y compara graph_off, graph_on_1hop, oracle "
            "y router_real sobre el mismo dataset."
        ),
    )
    parser.add_argument("--min-hit-rate", type=float, default=0.80)
    parser.add_argument("--min-abstention-accuracy", type=float, default=0.80)
    parser.add_argument("--min-answer-coverage", type=float, default=0.80)
    args = parser.parse_args(argv)

    try:
        cases = load_dataset(args.dataset.resolve())
    except Exception as exc:
        print(f"Error de evaluación: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_strategies:
        comparison = compare_strategies(cases)
        comparison_path = args.output_dir / "strategy_comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for strategy, payload in comparison["strategies"].items():
            metrics = payload["metrics"]
            print(
                f"{strategy:<14} recall={metrics['document_recall']} "
                f"precision@k={metrics['precision_at_k']} "
                f"mrr={metrics['mean_reciprocal_rank']}"
            )
        print(f"Informe: {comparison_path}")
        return 0

    report = evaluate(
        cases, generate_answers=args.generate_answers, strategy=args.strategy
    )
    json_path = args.output_dir / "latest.json"
    markdown_path = args.output_dir / "latest.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"Informes: {json_path} | {markdown_path}")
    metrics = report["metrics"]
    passed = (
        metrics["retrieval_exact_hit_rate"] is not None
        and metrics["retrieval_exact_hit_rate"] >= args.min_hit_rate
        and (
            metrics["abstention_accuracy"] is None
            or metrics["abstention_accuracy"] >= args.min_abstention_accuracy
        )
        and (
            not args.generate_answers
            or (
                metrics["answer_term_coverage"] is not None
                and metrics["answer_term_coverage"]
                >= args.min_answer_coverage
            )
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Ejecuta el conjunto dorado en integración continua, sin GPU.

Levanta el Ollama simulado, reconstruye el vault y el índice en una base
temporal y lanza la evaluación. No mide calidad semántica real —para eso
hace falta el modelo de verdad— sino que **detecta regresiones**: si un
filtro deja de aplicarse, si la expansión de enlaces se rompe o si la
abstención empieza a fallar, aquí salta.

    python -m scripts.run_eval_ci [--umbral-recuperacion 0.9]

Devuelve 0 si se cumplen los umbrales y 1 si no, para que CI falle.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAKE_PORT = 11599


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umbral-recuperacion", type=float, default=1.0)
    parser.add_argument("--umbral-abstencion", type=float, default=1.0)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evaluations" / "ci_questions.json",
    )
    parser.add_argument(
        "--informe",
        type=Path,
        default=ROOT / "evaluations" / "reports" / "ci.json",
    )
    args = parser.parse_args()

    from tests.fake_ollama import start

    start(FAKE_PORT)
    time.sleep(0.3)

    temporary = Path(tempfile.mkdtemp(prefix="rag-ci-"))

    from app.config import settings

    settings.ollama_base_url = f"http://127.0.0.1:{FAKE_PORT}"
    settings.database_path = temporary / "index.sqlite3"
    settings.vault_path = ROOT / "vault_demo"
    # La escala absoluta del coseno depende del modelo de embedding: el
    # 0,30 del .env está calibrado para nomic-embed-text y no significa
    # lo mismo en el doble de pruebas. La discriminación real la aporta
    # el umbral relativo, que sí es comparable entre modelos.
    settings.min_similarity = 0.21  # calibrado con scripts.calibrate_threshold
    import os
    if os.getenv("CI_RERANK") == "1":
        settings.rerank_enabled = True
        settings.rerank_backend = "llm"

    from app.rag import ollama_client

    ollama_client.get_ollama_client.cache_clear()

    from scripts.build_demo_vault import main as build_vault

    build_vault()

    from app.rag.indexer import index_vault

    result = index_vault()
    if result["errors"]:
        print("Errores de indexación:", file=sys.stderr)
        for error in result["errors"]:
            print(f"  {error}", file=sys.stderr)
        return 2
    print(
        f"Índice: {result['indexed']} documentos, "
        f"{result['chunks_created']} fragmentos"
    )

    from app.rag.evaluation import evaluate, load_dataset

    todos = load_dataset(args.dataset)
    # El doble de pruebas puntúa por solapamiento de palabras: los casos
    # que dependen de sinonimia real quedan fuera de CI y se reservan
    # para la evaluación con el modelo de verdad. No se rebaja el
    # listón: se declara qué puede medir cada entorno.
    cases = [case for case in todos if not case.requires_semantic]
    aplazados = [case.id for case in todos if case.requires_semantic]
    report = evaluate(cases, generate_answers=False)
    metrics = report["metrics"]

    print()
    print(f"{'caso':<34}{'ok':>4}{'recall':>9}{'fugas':>7}")
    print("-" * 54)
    failures: list[str] = []
    for item in report["results"]:
        passed = item["retrieval_pass"]
        recall = item["document_recall"]
        leaked = len(item.get("leaked_paths") or [])
        print(
            f"{item['id']:<34}"
            f"{'sí' if passed else 'NO':>4}"
            f"{'—' if recall is None else f'{recall:.0%}':>9}"
            f"{leaked:>7}"
        )
        if not passed:
            failures.append(item["id"])

    print("-" * 54)
    hit_rate = metrics["retrieval_exact_hit_rate"]
    abstention = metrics["abstention_accuracy"]
    print(f"acierto de recuperación : {hit_rate:.1%}")
    print(f"precisión de abstención : {abstention:.1%}")
    print(f"MRR                     : {metrics['mean_reciprocal_rank']:.3f}")

    args.informe.parent.mkdir(parents=True, exist_ok=True)
    args.informe.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if aplazados:
        print()
        print(
            f"{len(aplazados)} casos requieren semántica real y no se "
            f"evalúan aquí: {', '.join(aplazados)}"
        )
        print(
            "  Ejecuta 'python -m app.rag.evaluation' con Ollama real "
            "para medirlos."
        )
    print(f"informe: {args.informe}")

    passed = (
        hit_rate >= args.umbral_recuperacion
        and abstention >= args.umbral_abstencion
    )
    if not passed:
        print()
        print(f"FALLAN {len(failures)} casos: {', '.join(failures)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

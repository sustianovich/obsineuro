"""Compara la búsqueda semántica antigua con la nueva.

Uso:
    python -m scripts.bench_retrieval [n_fragmentos] [n_consultas]

Genera un índice sintético en una base temporal, no toca los datos
reales, y mide el mismo trabajo con los dos enfoques:

  antiguo : load_all_chunks() + bucle Python con cosine_similarity()
  nuevo   : vector_store con matriz normalizada y producto matricial
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

TEMP_DIR = Path(tempfile.mkdtemp(prefix="rag-bench-"))
settings.database_path = TEMP_DIR / "bench.sqlite3"

from app.db import get_connection, init_db  # noqa: E402
from app.rag.vector_store import vector_store  # noqa: E402


DIMENSION = 768


def build_corpus(chunk_count: int) -> None:
    init_db()
    generator = np.random.default_rng(7)
    now = "2026-01-01T00:00:00+00:00"
    documents = max(1, chunk_count // 8)

    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO documents(
                path, title, sha256, mtime, metadata_json,
                links_json, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f"notas/documento-{index}.md",
                    f"Documento {index}",
                    f"{index:064x}",
                    0.0,
                    json.dumps(
                        {"estado": "aprobado" if index % 2 else "borrador"}
                    ),
                    "[]",
                    now,
                )
                for index in range(documents)
            ],
        )
        rows = []
        text = "Contenido sintético de prueba. " * 40
        for index in range(chunk_count):
            vector = generator.standard_normal(DIMENSION).astype(np.float32)
            rows.append(
                (
                    index % documents + 1,
                    index,
                    f"Sección {index}",
                    text,
                    vector.tobytes(),
                    DIMENSION,
                )
            )
        connection.executemany(
            """
            INSERT INTO chunks(
                document_id, chunk_index, heading, content,
                embedding, embedding_dim
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def legacy_search(query: np.ndarray, top_k: int) -> list[int]:
    """Reproduce el camino antiguo: cargar todo y recorrer en Python."""
    with get_connection() as connection:
        raw = connection.execute(
            """
            SELECT c.id, c.embedding, d.metadata_json, d.title, c.content
            FROM chunks c JOIN documents d ON d.id = c.document_id
            """
        ).fetchall()

    scored = []
    for row in raw:
        vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        json.loads(row["metadata_json"] or "{}")
        denominator = float(
            np.linalg.norm(query) * np.linalg.norm(vector)
        )
        score = (
            float(np.dot(query, vector) / denominator)
            if denominator
            else 0.0
        )
        scored.append((score, int(row["id"])))
    scored.sort(reverse=True)
    return [chunk_id for _, chunk_id in scored[:top_k]]


def new_search(query: np.ndarray, top_k: int) -> list[int]:
    return [
        item["chunk_id"]
        for item in vector_store.search(
            query,
            limit=top_k,
            min_similarity=-1.0,
        )
    ]


def main() -> None:
    chunk_count = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    query_count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    top_k = 6

    print(f"Construyendo corpus sintético: {chunk_count} fragmentos…")
    build_corpus(chunk_count)

    generator = np.random.default_rng(99)
    queries = [
        generator.standard_normal(DIMENSION).astype(np.float32)
        for _ in range(query_count)
    ]

    start = time.perf_counter()
    legacy_results = [legacy_search(query, top_k) for query in queries]
    legacy_seconds = time.perf_counter() - start

    vector_store.invalidate()
    vector_store.ensure_loaded()  # carga en frío, fuera de la medición

    start = time.perf_counter()
    new_results = [new_search(query, top_k) for query in queries]
    new_seconds = time.perf_counter() - start

    agreement = sum(
        len(set(a) & set(b)) / top_k
        for a, b in zip(legacy_results, new_results)
    ) / query_count

    print()
    print(f"{'enfoque':<12}{'total (s)':>12}{'por consulta (ms)':>20}")
    print("-" * 44)
    print(
        f"{'antiguo':<12}{legacy_seconds:>12.3f}"
        f"{legacy_seconds / query_count * 1000:>20.1f}"
    )
    print(
        f"{'nuevo':<12}{new_seconds:>12.3f}"
        f"{new_seconds / query_count * 1000:>20.1f}"
    )
    print("-" * 44)
    print(f"aceleración: {legacy_seconds / max(new_seconds, 1e-9):.1f}×")
    print(f"coincidencia de resultados top-{top_k}: {agreement:.1%}")
    print(f"memoria de la matriz: {vector_store.stats()['memory_mb']} MB")


if __name__ == "__main__":
    main()

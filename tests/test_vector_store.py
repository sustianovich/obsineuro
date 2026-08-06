from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.config import settings


@pytest.fixture()
def temporary_index(tmp_path: Path):
    settings.database_path = tmp_path / "index.sqlite3"
    from app.db import get_connection, init_db
    from app.rag.vector_store import vector_store

    vector_store.invalidate()
    init_db()

    vectors = {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
        3: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        4: np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    estados = {1: "aprobado", 2: "aprobado", 3: "borrador", 4: "aprobado"}

    with get_connection() as connection:
        for document_id in (1, 2):
            connection.execute(
                """
                INSERT INTO documents(
                    path, title, sha256, mtime, metadata_json,
                    links_json, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"nota-{document_id}.md",
                    f"Nota {document_id}",
                    f"{document_id:064x}",
                    0.0,
                    json.dumps(
                        {
                            "estado": (
                                "aprobado" if document_id == 1 else "borrador"
                            ),
                            "_tags": (
                                ["normativa", "protocolo"]
                                if document_id == 1
                                else ["normativa"]
                            ),
                        }
                    ),
                    "[]",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
        for chunk_id, vector in vectors.items():
            connection.execute(
                """
                INSERT INTO chunks(
                    id, document_id, chunk_index, heading, content,
                    embedding, embedding_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    1 if estados[chunk_id] == "aprobado" else 2,
                    chunk_id,
                    f"Sección {chunk_id}",
                    f"Contenido {chunk_id}",
                    vector.tobytes(),
                    3,
                ),
            )
        connection.commit()

    vector_store.invalidate()
    yield vector_store
    vector_store.invalidate()


def test_ordena_por_similitud_coseno(temporary_index):
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = temporary_index.search(
        query, limit=3, min_similarity=-1.0
    )
    assert [item["chunk_id"] for item in results][:2] == [1, 2]
    assert results[0]["semantic_score"] == pytest.approx(1.0, abs=1e-5)


def test_respeta_el_umbral_minimo(temporary_index):
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = temporary_index.search(query, limit=10, min_similarity=0.5)
    assert {item["chunk_id"] for item in results} == {1, 2}


def test_filtra_por_estado_documental(temporary_index):
    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    results = temporary_index.search(
        query, limit=10, min_similarity=-1.0, status="borrador"
    )
    assert {item["chunk_id"] for item in results} == {3}


def test_rechaza_dimension_incompatible(temporary_index):
    query = np.ones(999, dtype=np.float32)
    assert temporary_index.search(
        query, limit=5, min_similarity=-1.0
    ) == []


def test_vector_nulo_no_rompe(temporary_index):
    query = np.zeros(3, dtype=np.float32)
    assert temporary_index.search(
        query, limit=5, min_similarity=-1.0
    ) == []


def test_recarga_tras_reindexar(temporary_index):
    from app.db import get_connection

    # invalidate() sólo marca la caché como caducada: la matriz anterior
    # sigue disponible hasta que ensure_loaded() la sustituye.
    temporary_index.ensure_loaded()
    assert temporary_index.stats()["loaded_chunks"] == 4

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chunks(
                id, document_id, chunk_index, heading, content,
                embedding, embedding_dim
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                99,
                1,
                99,
                "Nueva",
                "Nuevo contenido",
                np.array([0.5, 0.5, 0.5], dtype=np.float32).tobytes(),
                3,
            ),
        )
        connection.commit()

    # La firma cambia sola: no hace falta invalidar a mano.
    temporary_index.ensure_loaded()
    assert temporary_index.stats()["loaded_chunks"] == 5


def test_stats_cuenta_documentos_unicos_por_etiqueta(temporary_index):
    counts = {
        item["tag"]: item["documents"]
        for item in temporary_index.stats()["tag_counts"]
    }
    assert counts == {"normativa": 2, "protocolo": 1}


def test_hidrata_solo_los_solicitados(temporary_index):
    from app.rag.vector_store import hydrate_chunks

    hydrated = hydrate_chunks([1, 3])
    assert set(hydrated) == {1, 3}
    assert hydrated[1]["content"] == "Contenido 1"
    assert hydrated[1]["metadata"]["estado"] == "aprobado"

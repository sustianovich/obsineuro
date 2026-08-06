from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from app.config import settings
from app.rag.agents import _compact_hit_content
from app.rag.indexer import (
    build_index_configuration,
    changed_configuration_fields,
)
from app.main import _build_sources
from app.rag.markdown import parse_markdown_file
from app.rag.retrieval import collapse_parent_candidates


def test_parser_crea_padres_y_hijos_relacionados(tmp_path: Path):
    note = tmp_path / "nota.md"
    note.write_text(
        "# Tema\n\n" + " ".join(f"termino-{index}" for index in range(120)),
        encoding="utf-8",
    )

    document = parse_markdown_file(
        note,
        chunk_size=1800,
        chunk_overlap=250,
        parent_child_enabled=True,
        parent_chunk_size=220,
        child_chunk_size=90,
        child_chunk_overlap=20,
    )

    assert len(document.parents) > 1
    assert len(document.chunks) > len(document.parents)
    parents = {parent.index: parent.content for parent in document.parents}
    for child in document.chunks:
        assert child.parent_index in parents
        assert child.content in parents[child.parent_index]
        assert len(child.content) <= 90


def test_parser_plano_conserva_el_comportamiento_anterior(tmp_path: Path):
    note = tmp_path / "nota.md"
    note.write_text("# Tema\n\nContenido breve.", encoding="utf-8")

    document = parse_markdown_file(
        note,
        chunk_size=1800,
        chunk_overlap=250,
    )

    assert document.parents == []
    assert document.chunks[0].parent_index is None
    assert document.chunks[0].content == "Contenido breve."


def test_persistencia_expande_el_hijo_al_padre(tmp_path: Path):
    previous_database = settings.database_path
    settings.database_path = tmp_path / "parent-child.sqlite3"
    try:
        from app.db import init_db, replace_document
        from app.rag.vector_store import hydrate_chunks, vector_store

        init_db()
        replace_document(
            path="nota.md",
            title="Nota",
            sha256="a" * 64,
            mtime=0.0,
            metadata={},
            links=[],
            parents=[
                {
                    "parent_index": 0,
                    "heading": "Tema",
                    "content": "Contexto anterior. Dato preciso. Contexto posterior.",
                }
            ],
            chunks=[
                {
                    "chunk_index": 0,
                    "parent_index": 0,
                    "heading": "Tema",
                    "content": "Dato preciso.",
                    "embedding": np.array([1.0, 0.0], dtype=np.float32),
                }
            ],
        )

        with sqlite3.connect(settings.database_path) as connection:
            chunk_id = int(connection.execute("SELECT id FROM chunks").fetchone()[0])
        hit = hydrate_chunks([chunk_id])[chunk_id]

        assert hit["content"].startswith("Contexto anterior")
        assert hit["matched_content"] == "Dato preciso."
        assert hit["parent_id"] is not None
        assert hit["context_expanded"] is True
    finally:
        settings.database_path = previous_database
        vector_store.invalidate()


def test_reconstruccion_atomica_conserva_las_relaciones(tmp_path: Path):
    previous_database = settings.database_path
    settings.database_path = tmp_path / "staged-parent-child.sqlite3"
    try:
        from app.db import (
            commit_staged_index,
            get_connection,
            init_db,
            resolve_staged_document_links,
            stage_document,
        )

        init_db()
        stage_document(
            run_id="run-parent",
            path="nota.md",
            title="Nota",
            sha256="b" * 64,
            mtime=0.0,
            metadata={},
            links=[],
            parents=[
                {
                    "parent_index": 0,
                    "heading": "Tema",
                    "content": "Padre completo",
                }
            ],
            chunks=[
                {
                    "chunk_index": 0,
                    "parent_index": 0,
                    "heading": "Tema",
                    "content": "hijo",
                    "embedding": np.array([0.0, 1.0], dtype=np.float32),
                }
            ],
        )
        assert resolve_staged_document_links("run-parent") == 0
        commit_staged_index(
            run_id="run-parent",
            fingerprint_key="index_fingerprint",
            fingerprint_value="parent-child",
            expected_documents=1,
            expected_chunks=1,
            expected_links=0,
            expected_parents=1,
        )

        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT parent.content, child.parent_id
                FROM chunks AS child
                JOIN chunk_parents AS parent ON parent.id = child.parent_id
                """
            ).fetchone()
        assert row is not None
        assert row["content"] == "Padre completo"
        assert row["parent_id"] is not None
    finally:
        settings.database_path = previous_database


def test_migracion_conserva_un_indice_plano_existente(tmp_path: Path):
    previous_database = settings.database_path
    settings.database_path = tmp_path / "legacy.sqlite3"
    vector = np.array([1.0, 0.0], dtype=np.float32)
    with sqlite3.connect(settings.database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                mtime REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                links_json TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                heading TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                UNIQUE(document_id, chunk_index)
            );
            CREATE TABLE staged_documents (
                run_id TEXT NOT NULL,
                path TEXT NOT NULL,
                title TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                mtime REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                links_json TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                PRIMARY KEY(run_id, path)
            );
            CREATE TABLE staged_chunks (
                run_id TEXT NOT NULL,
                document_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                heading TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                PRIMARY KEY(run_id, document_path, chunk_index)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO documents(
                path, title, sha256, mtime, metadata_json,
                links_json, indexed_at
            ) VALUES ('legacy.md', 'Legacy', ?, 0, '{}', '[]', '2026-01-01')
            """,
            ("c" * 64,),
        )
        connection.execute(
            """
            INSERT INTO chunks(
                document_id, chunk_index, heading, content,
                embedding, embedding_dim
            ) VALUES (1, 0, 'Tema', 'Contenido antiguo', ?, 2)
            """,
            (vector.tobytes(),),
        )

    try:
        from app.db import get_connection, init_db
        from app.rag.vector_store import hydrate_chunks, vector_store

        init_db()
        with get_connection() as connection:
            chunk_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(chunks)")
            }
            staged_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(staged_chunks)"
                )
            }
            chunk_id = int(
                connection.execute("SELECT id FROM chunks").fetchone()["id"]
            )

        hit = hydrate_chunks([chunk_id])[chunk_id]
        assert "parent_id" in chunk_columns
        assert "parent_index" in staged_columns
        assert hit["content"] == "Contenido antiguo"
        assert hit["context_expanded"] is False
    finally:
        settings.database_path = previous_database
        vector_store.invalidate()


def test_colapsa_varios_hijos_del_mismo_padre():
    candidates = [
        {"chunk_id": 11, "parent_id": 7, "fusion_score": 0.9},
        {"chunk_id": 12, "parent_id": 7, "fusion_score": 0.8},
        {"chunk_id": 13, "parent_id": 8, "fusion_score": 0.7},
        {"chunk_id": 14, "parent_id": None, "fusion_score": 0.6},
    ]

    collapsed = collapse_parent_candidates(candidates)

    assert [item["chunk_id"] for item in collapsed] == [11, 13, 14]
    assert collapsed[0]["matched_chunk_ids"] == [11, 12]


def test_recorte_del_padre_mantiene_la_coincidencia():
    hit = {
        "content": "A" * 500 + "DATO-OBJETIVO" + "Z" * 500,
        "matched_content": "DATO-OBJETIVO",
    }

    compacted = _compact_hit_content(hit, 120)

    assert len(compacted) <= 120
    assert "DATO-OBJETIVO" in compacted


def test_build_sources_expone_matched_chunk_ids():
    """`collapse_parent_candidates` los calcula; la API no debe descartarlos.

    Cuántos hijos coincidieron para un mismo padre es una señal de fuerza de
    la evidencia (un padre con 3 hijos coincidentes es más sólido que uno
    con 1), igual de auditable que el resto de razones de recuperación.
    """
    hits = [
        {
            "title": "Nota",
            "path": "nota.md",
            "heading": "Tema",
            "score": 0.9,
            "reason": "recuperado",
            "content": "Contexto padre completo.",
            "matched_content": "Dato preciso.",
            "context_expanded": True,
            "matched_chunk_ids": [11, 12, 13],
        },
        {
            "title": "Nota plana",
            "path": "plana.md",
            "heading": "Tema",
            "score": 0.5,
            "reason": "recuperado",
            "content": "Fragmento sin padre.",
        },
    ]

    sources = _build_sources(hits)

    assert sources[0].matched_chunk_ids == [11, 12, 13]
    assert sources[1].matched_chunk_ids == []


def test_huella_plana_sigue_igual_y_el_modo_es_reversible(tmp_path: Path):
    base = build_index_configuration(
        vault_path=tmp_path,
        embedding_provider="ollama",
        embedding_model="modelo",
        embedding_dimension=3,
        chunk_size=1800,
        chunk_overlap=250,
        embedding_prefix_scheme="none",
        chunker_version="structured-1",
    )
    enabled = build_index_configuration(
        vault_path=tmp_path,
        embedding_provider="ollama",
        embedding_model="modelo",
        embedding_dimension=3,
        chunk_size=1800,
        chunk_overlap=250,
        embedding_prefix_scheme="none",
        chunker_version="structured-1",
        parent_child_chunking_enabled=True,
        parent_chunk_size=6000,
        child_chunk_size=700,
        child_chunk_overlap=100,
    )

    assert "chunking_mode" not in base
    assert enabled["chunking_mode"] == "parent_child-1"
    assert changed_configuration_fields(json.dumps(enabled), base) == [
        "chunking_mode"
    ]

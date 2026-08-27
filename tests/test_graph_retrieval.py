from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.rag.graph import (
    graph_candidates,
    graph_store,
    propagate_document_scores,
)
from app.rag.markdown import extract_wiki_links


def prepared_chunk(
    index: int = 0,
    *,
    heading: str = "Documento",
    vector: tuple[float, ...] = (1.0, 0.0),
) -> dict:
    return {
        "chunk_index": index,
        "heading": heading,
        "content": f"Contenido {index}",
        "embedding": np.asarray(vector, dtype=np.float32),
    }


@pytest.fixture()
def graph_db(tmp_path: Path):
    from app.db import init_db
    from app.rag.vector_store import vector_store

    previous = settings.database_path
    settings.database_path = tmp_path / "graph.sqlite3"
    vector_store.invalidate()
    graph_store.invalidate()
    init_db()
    try:
        yield
    finally:
        settings.database_path = previous
        vector_store.invalidate()
        graph_store.invalidate()


def replace_note(
    path: str,
    title: str,
    *,
    links: list[dict] | None = None,
    chunks: list[dict] | None = None,
    metadata: dict | None = None,
) -> None:
    from app.db import replace_document

    replace_document(
        path=path,
        title=title,
        sha256=(path.encode().hex() + "0" * 64)[:64],
        mtime=0.0,
        metadata=metadata or {},
        links=links or [],
        chunks=chunks or [prepared_chunk()],
    )


def test_extrae_seccion_alias_y_enlace_roto_sin_perder_el_destino():
    links = extract_wiki_links(
        "[[carpeta/Destino#Sección útil|texto visible]] "
        "![[Nota inexistente|vista]]"
    )
    assert links == [
        {
            "target": "carpeta/Destino",
            "section": "Sección útil",
            "alias": "texto visible",
            "embedded": False,
        },
        {
            "target": "Nota inexistente",
            "section": "",
            "alias": "vista",
            "embedded": True,
        },
    ]


def test_extrae_alias_con_separador_escapado_en_tabla_markdown():
    links = extract_wiki_links(
        "| Área | [[Hospitalizacion\\|hospitalización]] |"
    )

    assert links == [
        {
            "target": "Hospitalizacion",
            "section": "",
            "alias": "hospitalización",
            "embedded": False,
        }
    ]


def test_ignora_wikilinks_documentados_dentro_de_codigo_markdown():
    links = extract_wiki_links(
        "Usa `[[Ejemplo]]` como sintaxis.\n"
        "```markdown\n[[Tambien ejemplo]]\n```\n"
        "El enlace real es [[Destino]]."
    )

    assert links == [
        {
            "target": "Destino",
            "section": "",
            "alias": "",
            "embedded": False,
        }
    ]


def test_alta_sustitucion_y_baja_recalculan_aliases_y_backlinks(graph_db):
    from app.db import delete_documents_not_in, get_connection

    link = {
        "target": "Alias Original",
        "section": "Detalle",
        "alias": "ver",
        "embedded": False,
    }
    replace_note("origen.md", "Origen", links=[link])
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM document_links"
        ).fetchone()
    assert row["target_document_id"] is None
    assert row["section"] == "Detalle"

    replace_note("carpeta/destino.md", "Alias Original")
    with get_connection() as connection:
        row = connection.execute(
            "SELECT target_document_id FROM document_links"
        ).fetchone()
    assert row["target_document_id"] is not None

    replace_note("carpeta/destino.md", "Alias Cambiado")
    with get_connection() as connection:
        row = connection.execute(
            "SELECT target_document_id FROM document_links"
        ).fetchone()
    assert row["target_document_id"] is None

    replace_note(
        "origen.md",
        "Origen",
        links=[{**link, "target": "Alias Cambiado"}],
    )
    with get_connection() as connection:
        row = connection.execute(
            "SELECT target_document_id FROM document_links"
        ).fetchone()
    assert row["target_document_id"] is not None

    assert delete_documents_not_in({"origen.md"}) == 1
    with get_connection() as connection:
        row = connection.execute(
            "SELECT target_document_id FROM document_links"
        ).fetchone()
    assert row["target_document_id"] is None


def test_edicion_de_contenido_no_reconstruye_el_grafo_completo(
    graph_db, monkeypatch
):
    """Una edición de contenido (misma ruta, mismo título) no debe pasar
    por la reconstrucción completa del grafo, sólo por el refresco de las
    aristas propias de la nota tocada."""
    import app.db as db

    link = {
        "target": "Destino",
        "section": "",
        "alias": "",
        "embedded": False,
    }
    replace_note("origen.md", "Origen", links=[link])
    replace_note("destino.md", "Destino")

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "una edición de contenido no debe reconstruir el grafo entero"
        )

    monkeypatch.setattr(db, "_rebuild_document_links", fail_if_called)

    replace_note(
        "origen.md",
        "Origen",
        links=[link],
        chunks=[prepared_chunk(0, heading="Cambiado")],
    )

    with db.get_connection() as connection:
        origen_id = connection.execute(
            "SELECT id FROM documents WHERE path = 'origen.md'"
        ).fetchone()["id"]
        edge = connection.execute(
            """
            SELECT source_document_id, target_document_id, target_raw
            FROM document_links
            """
        ).fetchone()
    assert edge["source_document_id"] == origen_id
    assert edge["target_document_id"] is not None
    assert edge["target_raw"] == "Destino"


def test_renombrar_una_nota_reresuelve_lo_que_apuntaba_a_ella(
    graph_db,
):
    """El id de una nota cambia en cada sustitución (AUTOINCREMENT). Al
    cambiar su título, las aristas que otras notas tenían apuntando a ella
    deben sobrevivir apuntando al id nuevo, o romperse si el alias ya no
    coincide — nunca desaparecer por el borrado en cascada."""
    from app.db import get_connection

    replace_note(
        "origen.md",
        "Origen",
        links=[
            {
                "target": "Alias Original",
                "section": "Detalle",
                "alias": "",
                "embedded": False,
            }
        ],
    )
    replace_note("carpeta/destino.md", "Alias Original")
    with get_connection() as connection:
        edge = connection.execute(
            "SELECT target_document_id FROM document_links"
        ).fetchone()
    assert edge["target_document_id"] is not None

    # Editar el contenido de la nota destino sin cambiar el título no debe
    # afectar a quien la enlaza.
    replace_note(
        "carpeta/destino.md",
        "Alias Original",
        chunks=[prepared_chunk(0, heading="Otro contenido")],
    )
    with get_connection() as connection:
        edge = connection.execute(
            "SELECT target_document_id FROM document_links"
        ).fetchone()
    assert edge["target_document_id"] is not None

    # Renombrarla sí debe reresolver la arista entrante: se rompe porque el
    # alias antiguo ya no existe en el vault.
    replace_note("carpeta/destino.md", "Otro Nombre")
    with get_connection() as connection:
        edge = connection.execute(
            "SELECT target_document_id, target_raw FROM document_links"
        ).fetchone()
    assert edge["target_document_id"] is None
    assert edge["target_raw"] == "Alias Original"


def test_migracion_materializa_links_json_sin_tocar_huella(graph_db):
    from app.db import (
        GRAPH_VERSION_META_KEY,
        get_connection,
        get_meta,
        init_db,
    )

    replace_note(
        "origen.md",
        "Origen",
        links=[
            {
                "target": "Destino",
                "section": "",
                "alias": "",
                "embedded": False,
            },
            {
                "target": "Roto",
                "section": "",
                "alias": "",
                "embedded": False,
            },
        ],
    )
    replace_note("destino.md", "Destino")
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO app_meta(key, value) VALUES ('index_fingerprint', 'igual')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )
        connection.execute("DELETE FROM document_links")
        connection.execute(
            "DELETE FROM app_meta WHERE key = ?",
            (GRAPH_VERSION_META_KEY,),
        )
        connection.commit()

    init_db()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT target_raw, target_document_id
            FROM document_links
            ORDER BY target_raw
            """
        ).fetchall()
    assert [row["target_raw"] for row in rows] == ["Destino", "Roto"]
    assert rows[0]["target_document_id"] is not None
    assert rows[1]["target_document_id"] is None
    assert get_meta("index_fingerprint") == "igual"


def test_publicacion_atomica_conserva_el_grafo_si_el_staging_es_incompleto(
    graph_db,
):
    from app.db import (
        commit_staged_index,
        get_connection,
        resolve_staged_document_links,
        stage_document,
    )

    replace_note(
        "anterior.md",
        "Anterior",
        links=[
            {
                "target": "Destino",
                "section": "",
                "alias": "",
                "embedded": False,
            }
        ],
    )
    replace_note("destino.md", "Destino")
    with get_connection() as connection:
        before = connection.execute(
            """
            SELECT source_document_id, target_document_id, target_raw
            FROM document_links
            """
        ).fetchall()

    stage_document(
        run_id="fallido",
        path="nuevo.md",
        title="Nuevo",
        sha256="1" * 64,
        mtime=0.0,
        metadata={},
        links=[],
        chunks=[prepared_chunk()],
    )
    assert resolve_staged_document_links("fallido") == 0
    with pytest.raises(ValueError, match="incompleta"):
        commit_staged_index(
            run_id="fallido",
            fingerprint_key="index_fingerprint",
            fingerprint_value="nuevo",
            expected_documents=1,
            expected_chunks=1,
            expected_links=1,
        )

    with get_connection() as connection:
        after = connection.execute(
            """
            SELECT source_document_id, target_document_id, target_raw
            FROM document_links
            """
        ).fetchall()
        active_paths = {
            row["path"]
            for row in connection.execute("SELECT path FROM documents")
        }
    assert [tuple(row) for row in after] == [tuple(row) for row in before]
    assert active_paths == {"anterior.md", "destino.md"}


def test_publicacion_atomica_activa_documentos_fragmentos_y_grafo_juntos(
    graph_db,
):
    from app.db import (
        commit_staged_index,
        get_connection,
        resolve_staged_document_links,
        stage_document,
    )

    stage_document(
        run_id="correcto",
        path="origen.md",
        title="Origen",
        sha256="1" * 64,
        mtime=0.0,
        metadata={},
        links=[
            {
                "target": "Destino",
                "section": "Apartado",
                "alias": "",
                "embedded": False,
            }
        ],
        chunks=[prepared_chunk()],
    )
    stage_document(
        run_id="correcto",
        path="destino.md",
        title="Destino",
        sha256="2" * 64,
        mtime=0.0,
        metadata={},
        links=[],
        chunks=[prepared_chunk()],
    )
    assert resolve_staged_document_links("correcto") == 1
    commit_staged_index(
        run_id="correcto",
        fingerprint_key="index_fingerprint",
        fingerprint_value="correcto",
        expected_documents=2,
        expected_chunks=2,
        expected_links=1,
    )

    with get_connection() as connection:
        edge = connection.execute(
            """
            SELECT source.path AS source, target.path AS target, link.section
            FROM document_links AS link
            JOIN documents AS source ON source.id = link.source_document_id
            JOIN documents AS target ON target.id = link.target_document_id
            """
        ).fetchone()
        chunk_count = connection.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        fts_count = connection.execute(
            "SELECT COUNT(*) FROM chunks_fts"
        ).fetchone()[0]
    assert dict(edge) == {
        "source": "origen.md",
        "target": "destino.md",
        "section": "Apartado",
    }
    assert chunk_count == fts_count == 2


def test_graph_store_no_repite_la_consulta_completa_si_no_hay_cambios(
    graph_db, monkeypatch
):
    """El grafo debe cachear igual que `vector_store` cachea la matriz: una
    segunda consulta sin cambios no debe releer `document_links`."""
    import app.rag.graph as graph_module

    replace_note(
        "origen.md",
        "Origen",
        links=[
            {
                "target": "Destino",
                "section": "",
                "alias": "",
                "embedded": False,
            }
        ],
    )
    replace_note("destino.md", "Destino")
    graph_module.graph_store.invalidate()

    calls = {"count": 0}
    original_load = graph_module._load_edges

    def counting_load(connection):
        calls["count"] += 1
        return original_load(connection)

    monkeypatch.setattr(graph_module, "_load_edges", counting_load)

    first = graph_module.graph_store.edges()
    second = graph_module.graph_store.edges()

    assert calls["count"] == 1
    assert first == second
    assert len(first) == 1


def test_graph_store_detecta_cambios_sin_invalidar_a_mano(graph_db):
    """Altas, bajas y resoluciones de enlaces rotos deben ser visibles en
    la siguiente consulta aunque nadie llame a `invalidate()` a mano."""
    from app.rag.graph import graph_store

    replace_note("origen.md", "Origen")
    assert graph_store.edges() == []

    replace_note(
        "origen.md",
        "Origen",
        links=[
            {
                "target": "Destino",
                "section": "",
                "alias": "",
                "embedded": False,
            }
        ],
    )
    assert graph_store.edges() == []  # el destino todavía no existe

    replace_note("destino.md", "Destino")
    resolved = graph_store.edges()
    assert len(resolved) == 1


def test_propaga_un_y_dos_saltos_con_decaimiento_y_backlinks():
    scores = propagate_document_scores(
        {1: 1.0},
        [(1, 2), (2, 3), (4, 1)],
        max_hops=2,
        decay=0.5,
        backlink_weight=0.7,
    )
    assert scores[2] == pytest.approx(0.5)
    assert scores[3] == pytest.approx(0.25)
    assert scores[4] == pytest.approx(0.35)
    assert 1 not in scores


def test_elige_el_fragmento_mas_similar_y_no_el_indice_cero(
    graph_db,
    monkeypatch,
):
    from app.db import get_connection
    from app.rag.vector_store import vector_store

    replace_note(
        "semilla.md",
        "Semilla",
        links=[
            {
                "target": "Alcanzada",
                "section": "",
                "alias": "",
                "embedded": False,
            }
        ],
        chunks=[prepared_chunk(vector=(0.8, 0.2))],
    )
    replace_note(
        "alcanzada.md",
        "Alcanzada",
        chunks=[
            prepared_chunk(0, vector=(0.0, 1.0)),
            prepared_chunk(1, vector=(1.0, 0.0)),
        ],
    )
    vector_store.invalidate()
    vector_store.ensure_loaded()
    with get_connection() as connection:
        seed = connection.execute(
            """
            SELECT d.id AS document_id, c.id AS chunk_id
            FROM documents d JOIN chunks c ON c.document_id = d.id
            WHERE d.path = 'semilla.md'
            """
        ).fetchone()
        wanted = connection.execute(
            """
            SELECT c.id AS chunk_id
            FROM documents d JOIN chunks c ON c.document_id = d.id
            WHERE d.path = 'alcanzada.md' AND c.chunk_index = 1
            """
        ).fetchone()["chunk_id"]

    monkeypatch.setattr(settings, "graph_search_enabled", True)
    monkeypatch.setattr(settings, "graph_seed_documents", 1)
    monkeypatch.setattr(settings, "graph_max_hops", 1)
    monkeypatch.setattr(settings, "graph_decay", 0.5)
    monkeypatch.setattr(settings, "graph_backlink_weight", 0.7)
    monkeypatch.setattr(settings, "graph_max_candidates", 20)
    candidates = graph_candidates(
        [
            {
                "document_id": int(seed["document_id"]),
                "chunk_id": int(seed["chunk_id"]),
                "row": vector_store.row_for_chunk(int(seed["chunk_id"])),
                "semantic_score": 0.9,
            }
        ],
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert candidates[0]["chunk_id"] == int(wanted)
    assert candidates[0]["graph_chunk_similarity"] == pytest.approx(1.0)


def hydrated_chunk(
    chunk_id: int,
    *,
    status: str = "vigente",
    vigencia: str = "vigente",
    tags: list[str] | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_index": 0,
        "heading": "Documento",
        "content": f"Contenido {chunk_id}",
        "embedding_dim": 2,
        "document_id": chunk_id,
        "path": f"{chunk_id}.md",
        "title": f"Nota {chunk_id}",
        "metadata": {
            "estado": status,
            "_vigencia": {"estado_temporal": vigencia},
            "_tags": tags or [],
        },
        "links": [],
    }


def test_filtros_descartan_candidatos_del_grafo_tras_hidratarlos(
    monkeypatch,
):
    import app.rag.retrieval as retrieval

    candidates = {
        1: hydrated_chunk(1, tags=["ok"]),
        2: hydrated_chunk(2, status="borrador", tags=["ok"]),
        3: hydrated_chunk(3, vigencia="caducada", tags=["ok"]),
        4: hydrated_chunk(4, tags=["otra"]),
        5: hydrated_chunk(5, tags=["ok"]),
    }
    monkeypatch.setattr(settings, "graph_search_enabled", True)
    monkeypatch.setattr(settings, "hybrid_search_enabled", False)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "mmr_enabled", False)
    monkeypatch.setattr(
        retrieval,
        "embed_question",
        lambda _: np.array([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        retrieval,
        "_semantic_candidates",
        lambda *args, **kwargs: [
            {
                "chunk_id": 1,
                "document_id": 1,
                "row": 0,
                "semantic_score": 0.9,
            }
        ],
    )
    monkeypatch.setattr(
        retrieval,
        "graph_candidates",
        lambda *args, **kwargs: [
            {
                "chunk_id": chunk_id,
                "document_id": chunk_id,
                "row": chunk_id,
                "graph_score": 0.5,
                "graph_hop": 1,
                "graph_chunk_similarity": 0.8,
            }
            for chunk_id in (2, 3, 4, 5)
        ],
    )
    monkeypatch.setattr(
        retrieval,
        "hydrate_chunks",
        lambda _: candidates,
    )

    hits = retrieval.retrieve(
        "pregunta",
        top_k=10,
        min_similarity=0.3,
        status="vigente",
        expand_links=True,
        vigencia="vigente",
        tags=["ok"],
    )
    assert {hit["chunk_id"] for hit in hits} == {1, 5}
    assert next(hit for hit in hits if hit["chunk_id"] == 5)["reason"] == (
        "búsqueda por grafo"
    )


def test_guarantee_graph_neighbors_solo_anade_salto_1_no_repetido():
    from app.rag.retrieval import guarantee_graph_neighbors

    selected = [{"chunk_id": 1, "document_id": 1}]
    hydrated = {
        2: hydrated_chunk(2),
        3: hydrated_chunk(3),
    }
    graph = [
        {
            "chunk_id": 2,
            "document_id": 2,
            "graph_score": 0.4,
            "graph_hop": 1,
            "graph_chunk_similarity": 0.6,
        },
        {
            # Descartado: dos saltos, no es la garantía de vecino directo.
            "chunk_id": 3,
            "document_id": 3,
            "graph_score": 0.3,
            "graph_hop": 2,
            "graph_chunk_similarity": 0.9,
        },
        {
            # Descartado: el documento ya está en la selección.
            "chunk_id": 1,
            "document_id": 1,
            "graph_score": 0.2,
            "graph_hop": 1,
            "graph_chunk_similarity": 0.5,
        },
    ]
    additions = guarantee_graph_neighbors(selected, graph, hydrated)
    assert [item["chunk_id"] for item in additions] == [2]
    assert additions[0]["reason"] == "búsqueda por grafo (enlace directo)"
    assert additions[0]["score"] == pytest.approx(0.6 * 0.85)


def test_guarantee_graph_neighbors_respeta_el_limite():
    from app.rag.retrieval import guarantee_graph_neighbors

    selected = [{"chunk_id": 1, "document_id": 1}]
    hydrated = {chunk_id: hydrated_chunk(chunk_id) for chunk_id in range(2, 6)}
    graph = [
        {
            "chunk_id": chunk_id,
            "document_id": chunk_id,
            "graph_score": 1.0 / chunk_id,
            "graph_hop": 1,
            "graph_chunk_similarity": 0.5,
        }
        for chunk_id in range(2, 6)
    ]
    additions = guarantee_graph_neighbors(selected, graph, hydrated, limit=2)
    assert len(additions) == 2


def test_grafo_garantiza_vecino_a_un_salto_que_pierde_el_top_k(monkeypatch):
    """Reproduce la regresión del informe: sin garantía, un vecino directo
    puede perder la competencia de RRF y desaparecer, cuando con
    `expand_links` (grafo apagado) siempre se añadía tras el `top_k`."""
    import app.rag.retrieval as retrieval

    candidates = {
        1: hydrated_chunk(1),
        2: hydrated_chunk(2),
    }
    monkeypatch.setattr(settings, "graph_search_enabled", True)
    monkeypatch.setattr(settings, "hybrid_search_enabled", False)
    monkeypatch.setattr(settings, "hybrid_semantic_weight", 1.0)
    monkeypatch.setattr(settings, "hybrid_graph_weight", 0.5)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "mmr_enabled", False)
    monkeypatch.setattr(
        retrieval,
        "embed_question",
        lambda _: np.array([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        retrieval,
        "_semantic_candidates",
        lambda *args, **kwargs: [
            {
                "chunk_id": 1,
                "document_id": 1,
                "row": 0,
                "semantic_score": 0.9,
            }
        ],
    )
    monkeypatch.setattr(
        retrieval,
        "graph_candidates",
        lambda *args, **kwargs: [
            {
                "chunk_id": 2,
                "document_id": 2,
                "row": 1,
                "graph_score": 0.4,
                "graph_hop": 1,
                "graph_chunk_similarity": 0.6,
            }
        ],
    )
    monkeypatch.setattr(retrieval, "hydrate_chunks", lambda _: candidates)

    # top_k=1 deja un único hueco: la semántica lo gana con peso 1.0 frente
    # al 0.5 del grafo, así que el vecino nunca entra por fusión pura.
    hits = retrieval.retrieve(
        "pregunta",
        top_k=1,
        min_similarity=0.3,
        status=None,
        expand_links=True,
    )
    assert [hit["chunk_id"] for hit in hits] == [1, 2]
    assert hits[1]["reason"] == "búsqueda por grafo (enlace directo)"


def test_abstencion_sucede_antes_de_consultar_el_grafo(monkeypatch):
    import app.rag.retrieval as retrieval

    monkeypatch.setattr(settings, "graph_search_enabled", True)
    monkeypatch.setattr(
        retrieval,
        "embed_question",
        lambda _: np.array([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        retrieval,
        "_semantic_candidates",
        lambda *args, **kwargs: [],
    )

    def graph_must_not_run(*args, **kwargs):
        raise AssertionError("el grafo no debe ejecutarse sin semillas")

    monkeypatch.setattr(retrieval, "graph_candidates", graph_must_not_run)
    assert retrieval.retrieve(
        "fuera de dominio",
        top_k=5,
        min_similarity=0.3,
        status=None,
        expand_links=True,
    ) == []


def test_estado_del_grafo_cuenta_resueltas_rotas_y_huerfanas(graph_db):
    from app.rag.graph import get_graph_status

    replace_note(
        "origen.md",
        "Origen",
        links=[
            {
                "target": "Destino",
                "section": "",
                "alias": "",
                "embedded": False,
            },
            {
                "target": "Rota",
                "section": "",
                "alias": "",
                "embedded": False,
            },
        ],
    )
    replace_note("destino.md", "Destino")
    replace_note("huerfana.md", "Huérfana")
    status = get_graph_status()
    assert status["total_edges"] == 2
    assert status["resolved_edges"] == 1
    assert status["broken_edges"] == 1
    assert status["orphan_documents"] == 1
    assert status["ambiguous_aliases"] == 0


def test_estado_del_grafo_cuenta_alias_ambiguos(graph_db):
    from app.rag.graph import get_graph_status

    replace_note("carpetaA/Index.md", "Índice")
    replace_note("carpetaB/Index.md", "Índice")
    replace_note("unica.md", "Nota única")

    status = get_graph_status()
    # "index" (nombre de archivo) y "índice" (título) coinciden entre las
    # dos notas: ambas cuentan como alias ambiguos.
    assert status["ambiguous_aliases"] == 2

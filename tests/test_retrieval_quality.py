from __future__ import annotations

import pytest

from app.config import settings
from app.rag.embedding_tasks import PREFIX_SCHEMES, resolve_scheme
from app.rag.ollama_client import _ThinkFilter
from app.rag.retrieval import (
    _cap_per_document,
    apply_relative_cutoff,
    build_fts_match_query,
    lexical_term_coverage,
    required_lexical_matches,
)


# ----------------------------------------------------------------------
# Prefijos de tarea
# ----------------------------------------------------------------------
def test_detecta_nomic_automaticamente():
    scheme = resolve_scheme("nomic-embed-text", "auto")
    assert scheme.id == "nomic"
    assert scheme.query == "search_query: "
    assert scheme.document == "search_document: "


def test_detecta_e5_y_bge():
    assert resolve_scheme("multilingual-e5-large", "auto").id == "e5"
    assert resolve_scheme("bge-m3:latest", "auto").id == "bge"


def test_modelo_desconocido_sin_prefijos():
    scheme = resolve_scheme("mxbai-embed-otro", "auto")
    assert scheme.id == "none"
    assert scheme.query == ""


def test_configuracion_explicita_gana_a_la_deteccion():
    assert resolve_scheme("nomic-embed-text", "none").id == "none"


def test_esquema_invalido_falla_pronto():
    with pytest.raises(ValueError, match="RAG_EMBEDDING_PREFIX_SCHEME"):
        resolve_scheme("nomic-embed-text", "inventado")


def test_todos_los_esquemas_declaran_ambos_prefijos():
    for scheme in PREFIX_SCHEMES.values():
        assert isinstance(scheme.document, str)
        assert isinstance(scheme.query, str)


# ----------------------------------------------------------------------
# Corte relativo
# ----------------------------------------------------------------------
def test_corte_relativo_descarta_lo_lejano():
    original = settings.min_relative_score
    settings.min_relative_score = 0.6
    try:
        candidates = [
            {"semantic_score": 0.80},
            {"semantic_score": 0.55},
            {"semantic_score": 0.20},
        ]
        kept = apply_relative_cutoff(candidates)
        assert [item["semantic_score"] for item in kept] == [0.80, 0.55]
    finally:
        settings.min_relative_score = original


def test_corte_relativo_desactivado_no_toca_nada():
    original = settings.min_relative_score
    settings.min_relative_score = 0.0
    try:
        candidates = [{"semantic_score": 0.9}, {"semantic_score": 0.1}]
        assert apply_relative_cutoff(candidates) == candidates
    finally:
        settings.min_relative_score = original


def test_corte_relativo_conserva_siempre_el_mejor():
    original = settings.min_relative_score
    settings.min_relative_score = 0.99
    try:
        kept = apply_relative_cutoff([{"semantic_score": 0.5}])
        assert len(kept) == 1
    finally:
        settings.min_relative_score = original


# ----------------------------------------------------------------------
# 9. El corte relativo nunca se contabiliza como abstención: para
# cualquier umbral y cualquier dispersión de puntuaciones, siempre queda
# al menos el mejor candidato. La abstención real sólo la decide
# RAG_MIN_SIMILARITY, antes de llegar aquí.
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "scores, ratio",
    [
        ([0.9, 0.1], 0.0),
        ([0.9, 0.1], 0.5),
        ([0.9, 0.1], 0.99),
        ([0.31, 0.30, 0.05], 0.62),
        ([0.05], 0.99),
        ([0.0, 0.0, 0.0], 0.62),
    ],
)
def test_corte_relativo_nunca_vacia_el_resultado(scores, ratio):
    original = settings.min_relative_score
    settings.min_relative_score = ratio
    try:
        candidates = [{"semantic_score": score} for score in scores]
        kept = apply_relative_cutoff(candidates)
        assert len(kept) >= 1
        assert kept[0]["semantic_score"] == pytest.approx(scores[0])
    finally:
        settings.min_relative_score = original


# ----------------------------------------------------------------------
# Tope por documento
# ----------------------------------------------------------------------
def test_tope_por_documento_evita_monopolio():
    original = settings.max_chunks_per_document
    settings.max_chunks_per_document = 2
    try:
        candidates = [
            {"document_id": 1, "chunk_id": index} for index in range(5)
        ] + [{"document_id": 2, "chunk_id": 99}]
        selected = _cap_per_document(candidates, top_k=3)
        documents = [item["document_id"] for item in selected]
        assert documents.count(1) == 2
        assert 2 in documents
    finally:
        settings.max_chunks_per_document = original


def test_tope_cede_si_no_hay_alternativa():
    original = settings.max_chunks_per_document
    settings.max_chunks_per_document = 1
    try:
        candidates = [
            {"document_id": 1, "chunk_id": index} for index in range(4)
        ]
        # Con un solo documento disponible, el tope no puede dejar
        # huecos vacíos en el contexto.
        assert len(_cap_per_document(candidates, top_k=3)) == 3
    finally:
        settings.max_chunks_per_document = original


# ----------------------------------------------------------------------
# Capa léxica
# ----------------------------------------------------------------------
def test_consulta_fts_ignora_palabras_vacias():
    query, terms = build_fts_match_query("¿Cuáles son las fases del alta?")
    assert "cuales" not in terms
    assert "fases" in terms and "alta" in terms
    assert query.startswith('"')


def test_cobertura_lexica_ignora_acentos():
    chunk = {
        "title": "Política de validación",
        "heading": "Criterios",
        "content": "La validacion exige dos comprobaciones.",
        "path": "ref/politica.md",
    }
    coverage, matched = lexical_term_coverage(
        chunk, ["validación", "comprobaciones"]
    )
    assert matched == 2
    assert coverage == pytest.approx(1.0)


def test_umbral_de_coincidencias_escala_con_la_consulta():
    assert required_lexical_matches(1) == 1
    assert required_lexical_matches(5) == 2
    assert required_lexical_matches(30) == 6


# ----------------------------------------------------------------------
# Filtro de razonamiento en streaming
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "pieces, expected",
    [
        (["Respuesta directa."], "Respuesta directa."),
        (["<think>oculto</think>Visible"], "Visible"),
        (["Ho", "la <th", "ink>x</thi", "nk> mundo"], "Hola  mundo"),
        (['<think type="a">x</think>B'], "B"),
        (["Texto con < menor que"], "Texto con < menor que"),
    ],
)
def test_filtro_de_razonamiento(pieces, expected):
    filtro = _ThinkFilter()
    result = "".join(filtro.feed(piece) for piece in pieces)
    assert result + filtro.flush() == expected


def test_razonamiento_sin_cerrar_no_se_filtra():
    filtro = _ThinkFilter()
    visible = filtro.feed("Antes<think>nunca cierra")
    assert visible == "Antes"
    assert filtro.flush() == ""

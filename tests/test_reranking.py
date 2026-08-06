from __future__ import annotations

import math

import numpy as np
import pytest

from app.config import settings
from app.rag import reranking
from app.rag.reranking import (
    OnnxReranker,
    RerankUnavailable,
    _parse_scores,
    get_rerank_status,
    rerank,
)


@pytest.fixture()
def candidatos():
    return [
        {
            "chunk_id": index,
            "title": f"Documento {index}",
            "heading": f"Seccion {index}",
            "content": f"Contenido del fragmento {index}." * 5,
            "fusion_score": 1.0 - index * 0.1,
            "reason": "búsqueda híbrida",
        }
        for index in range(4)
    ]


@pytest.fixture()
def rerank_activo():
    previo = (
        settings.rerank_enabled,
        settings.rerank_backend,
        settings.rerank_weight,
    )
    settings.rerank_enabled = True
    settings.rerank_backend = "llm"
    settings.rerank_weight = 1.0
    yield
    (
        settings.rerank_enabled,
        settings.rerank_backend,
        settings.rerank_weight,
    ) = previo


def test_desactivado_devuelve_el_orden_original(candidatos):
    previo = settings.rerank_enabled
    settings.rerank_enabled = False
    try:
        resultado = rerank("pregunta", candidatos)
        assert resultado.status == "disabled"
        assert resultado.items == candidatos
    finally:
        settings.rerank_enabled = previo


def test_reordena_segun_las_puntuaciones(
    candidatos, rerank_activo, monkeypatch
):
    # El último candidato es en realidad el más relevante.
    monkeypatch.setitem(
        reranking.BACKENDS,
        "llm",
        lambda question, passages: [0.1, 0.2, 0.3, 0.9],
    )
    resultado = rerank("pregunta", candidatos)
    assert resultado.status == "completed"
    assert resultado.scored == 4
    assert resultado.items[0]["title"] == "Documento 3"
    assert resultado.moved > 0


def test_backend_onnx_reordena_por_lotes(
    candidatos, rerank_activo, monkeypatch
):
    class FakeOnnxReranker:
        def __init__(self):
            self.calls: list[list[str]] = []

        def score(self, question, passages, *, max_tokens):
            assert question == "pregunta"
            assert max_tokens == settings.rerank_onnx_max_tokens
            self.calls.append(passages)
            return [0.1, 0.2, 0.3, 0.9][: len(passages)]

    fake = FakeOnnxReranker()
    previous = settings.rerank_backend
    settings.rerank_backend = "onnx"
    monkeypatch.setattr(reranking, "_load_onnx_reranker", lambda _: fake)
    try:
        resultado = rerank("pregunta", candidatos)
    finally:
        settings.rerank_backend = previous

    assert resultado.status == "completed"
    assert resultado.backend == "onnx"
    assert resultado.items[0]["title"] == "Documento 3"
    assert sum(len(batch) for batch in fake.calls) == len(candidatos)


def test_onnx_reranker_tokeniza_pares_y_aplica_sigmoide():
    class FakeTokenizer:
        def __call__(self, questions, passages, **kwargs):
            assert questions == ["pregunta", "pregunta"]
            assert passages == ["uno", "dos"]
            assert kwargs["return_tensors"] == "np"
            return {
                "input_ids": np.array([[1, 2], [3, 4]]),
                "attention_mask": np.array([[1, 1], [1, 1]]),
            }

    class FakeSession:
        def run(self, _, inputs):
            assert set(inputs) == {"input_ids", "attention_mask"}
            return [np.array([[0.0], [math.log(3)]])]

    model = OnnxReranker(
        tokenizer=FakeTokenizer(),
        session=FakeSession(),
        input_names=frozenset({"input_ids", "attention_mask"}),
    )

    assert model.score(
        "pregunta",
        ["uno", "dos"],
        max_tokens=512,
    ) == pytest.approx([0.5, 0.75])


def test_estado_onnx_indica_si_falta_el_modelo(tmp_path):
    previous = (
        settings.rerank_enabled,
        settings.rerank_backend,
        settings.rerank_onnx_model_dir,
    )
    settings.rerank_enabled = True
    settings.rerank_backend = "onnx"
    settings.rerank_onnx_model_dir = tmp_path
    try:
        status = get_rerank_status()
        assert status["model_available"] is False
        model_file = tmp_path / "onnx" / "model_quantized.onnx"
        model_file.parent.mkdir()
        model_file.touch()
        assert get_rerank_status()["model_available"] is True
    finally:
        (
            settings.rerank_enabled,
            settings.rerank_backend,
            settings.rerank_onnx_model_dir,
        ) = previous


def test_error_del_backend_conserva_el_orden(
    candidatos, rerank_activo, monkeypatch
):
    def _falla(question, passages):
        raise RerankUnavailable("modelo no instalado")

    monkeypatch.setitem(reranking.BACKENDS, "llm", _falla)
    resultado = rerank("pregunta", candidatos)
    assert resultado.status == "error"
    assert resultado.items == candidatos
    assert resultado.error == "El reordenador no pudo completarse."


def test_numero_de_puntuaciones_incorrecto_no_rompe(
    candidatos, rerank_activo, monkeypatch
):
    monkeypatch.setitem(
        reranking.BACKENDS,
        "llm",
        lambda question, passages: [0.5],
    )
    resultado = rerank("pregunta", candidatos)
    assert resultado.status == "error"
    assert resultado.items == candidatos


def test_backend_desconocido_se_informa(candidatos, rerank_activo):
    settings.rerank_backend = "inventado"
    resultado = rerank("pregunta", candidatos)
    assert resultado.status == "error"
    assert "inventado" in resultado.error


def test_respeta_el_presupuesto_de_candidatos(
    candidatos, rerank_activo, monkeypatch
):
    previo = settings.rerank_candidates
    settings.rerank_candidates = 2
    try:
        vistos: list[int] = []

        def _spy(question, passages):
            vistos.append(len(passages))
            return [0.9, 0.1]

        monkeypatch.setitem(reranking.BACKENDS, "llm", _spy)
        resultado = rerank("pregunta", candidatos)
        assert vistos == [2]
        # Los no puntuados conservan su posición al final.
        assert len(resultado.items) == 4
        assert resultado.items[-1]["title"] == "Documento 3"
    finally:
        settings.rerank_candidates = previo


# ----------------------------------------------------------------------
# Lectura de las puntuaciones del modelo
# ----------------------------------------------------------------------
def test_lee_json_limpio():
    assert _parse_scores('{"1": 10, "2": 5}', 2) == [1.0, 0.5]


def test_lee_json_envuelto_en_texto():
    crudo = 'Aqui tienes:\n```json\n{"1": 8, "2": 0}\n```\nListo.'
    assert _parse_scores(crudo, 2) == [0.8, 0.0]


def test_puntuacion_ausente_se_trata_como_cero():
    assert _parse_scores('{"1": 7}', 3) == [0.7, 0.0, 0.0]


def test_acota_valores_fuera_de_escala():
    assert _parse_scores('{"1": 99, "2": -4}', 2) == [1.0, 0.0]


def test_respuesta_sin_json_falla_de_forma_controlada():
    with pytest.raises(RerankUnavailable):
        _parse_scores("No puedo puntuar esto.", 2)

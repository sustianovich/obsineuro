from __future__ import annotations

import pytest

from app.config import settings
from app.rag.agents import (
    _verifier_report_body,
    parse_verifier_sufficiency,
    run_rag_agent_pipeline,
)
from app.rag.ollama_client import GenerationResult


def fake_generation(content: str) -> GenerationResult:
    return GenerationResult(
        content=content, prompt_tokens=10, completion_tokens=5, estimated=False
    )


SOME_HITS = [
    {
        "chunk_id": 1,
        "document_id": 1,
        "title": "Nota",
        "path": "nota.md",
        "heading": "Sección",
        "content": "Contenido de prueba.",
        "score": 0.9,
        "reason": "búsqueda semántica",
        "metadata": {},
    }
]


@pytest.fixture(autouse=True)
def sin_metadatos_ollama(monkeypatch):
    """Aísla la resolución de contexto de si hay un Ollama real corriendo.

    Sin esto, `resolve_model_context` intentaría un `api/show` real en
    cada prueba; con esto, se comporta como si Ollama no aportara
    metadatos, dejando sólo el perfil conocido (determinista).
    """
    from app.rag import ollama_client

    monkeypatch.setattr(ollama_client, "get_model_show", lambda model: None)


@pytest.fixture()
def modelo_restaurable():
    previo = settings.chat_model
    yield
    settings.chat_model = previo


# ----------------------------------------------------------------------
# Parseo de la línea SUFICIENCIA
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "report, expected",
    [
        ("SUFICIENCIA: suficiente\n\nresto", "suficiente"),
        ("suficiencia: PARCIAL\n\nresto", "parcial"),
        ("  SUFICIENCIA:   insuficiente  \nresto", "insuficiente"),
        ("\n\nSUFICIENCIA: suficiente\nresto", "suficiente"),
    ],
)
def test_parse_verifier_sufficiency_formatos_validos(report, expected):
    assert parse_verifier_sufficiency(report) == expected


@pytest.mark.parametrize(
    "report",
    [
        "El informe no sigue el formato esperado.",
        "Suficiencia total: sí",
        "",
        "SUFICIENCIA suficiente",
    ],
)
def test_parse_verifier_sufficiency_formato_invalido_da_none(report):
    assert parse_verifier_sufficiency(report) is None


def test_verifier_report_body_quita_la_linea_de_encabezado():
    report = "SUFICIENCIA: parcial\n\nEl resto del informe."
    assert _verifier_report_body(report) == "El resto del informe."


def test_verifier_report_body_sin_encabezado_devuelve_todo():
    report = "Informe sin encabezado reconocible."
    assert _verifier_report_body(report) == report


# ----------------------------------------------------------------------
# 6. Integración: abstención dirigida por el verificador (fallback seguro).
# ----------------------------------------------------------------------
def test_verificador_insuficiente_abstiene_cuando_esta_habilitado(monkeypatch):
    import app.rag.agents as agents

    monkeypatch.setattr(settings, "verifier_abstain_on_insufficient", True)

    calls = []

    def fake(instructions, prompt, **kwargs):
        calls.append(instructions)
        if instructions == agents.VERIFIER_INSTRUCTIONS:
            return fake_generation("SUFICIENCIA: insuficiente\n\nFalta evidencia.")
        raise AssertionError("el redactor no debía ejecutarse")

    monkeypatch.setattr(agents, "generate_rag_response_with_metrics", fake)

    result = agents.run_rag_agent_pipeline(
        "pregunta",
        SOME_HITS,
        conversation_memory="",
        project_memory="",
        verification_enabled=True,
        verifier_context_tokens=8192,
        writer_context_tokens=16384,
    )
    assert result.metrics["writer"]["status"] == "skipped_insufficient_evidence"
    assert result.metrics["verifier"]["sufficiency"] == "insuficiente"
    assert calls == [agents.VERIFIER_INSTRUCTIONS]
    assert "insuficiente" in (result.warning or "")


def test_verificador_insuficiente_no_abstiene_por_defecto(monkeypatch):
    import app.rag.agents as agents

    monkeypatch.setattr(settings, "verifier_abstain_on_insufficient", False)

    def fake(instructions, prompt, **kwargs):
        if instructions == agents.VERIFIER_INSTRUCTIONS:
            return fake_generation("SUFICIENCIA: insuficiente\n\nFalta evidencia.")
        return fake_generation("Respuesta final del redactor.")

    monkeypatch.setattr(agents, "generate_rag_response_with_metrics", fake)

    result = agents.run_rag_agent_pipeline(
        "pregunta",
        SOME_HITS,
        conversation_memory="",
        project_memory="",
        verification_enabled=True,
        verifier_context_tokens=8192,
        writer_context_tokens=16384,
    )
    assert result.metrics["writer"]["status"] != "skipped_insufficient_evidence"
    assert result.answer == "Respuesta final del redactor."


def test_formato_invalido_del_verificador_no_bloquea_al_redactor(monkeypatch):
    """Fallback seguro: un informe sin `SUFICIENCIA: ...` no cuenta como
    insuficiente, sólo como "no se pudo determinar"."""
    import app.rag.agents as agents

    monkeypatch.setattr(settings, "verifier_abstain_on_insufficient", True)

    def fake(instructions, prompt, **kwargs):
        if instructions == agents.VERIFIER_INSTRUCTIONS:
            return fake_generation("Informe sin el formato esperado.")
        return fake_generation("Respuesta final del redactor.")

    monkeypatch.setattr(agents, "generate_rag_response_with_metrics", fake)

    result = agents.run_rag_agent_pipeline(
        "pregunta",
        SOME_HITS,
        conversation_memory="",
        project_memory="",
        verification_enabled=True,
        verifier_context_tokens=8192,
        writer_context_tokens=16384,
    )
    assert result.metrics["verifier"]["sufficiency"] is None
    assert result.metrics["verifier"]["sufficiency_parse_error"] is True
    assert result.metrics["writer"]["status"] != "skipped_insufficient_evidence"
    assert result.answer == "Respuesta final del redactor."


# ----------------------------------------------------------------------
# Telemetría de contexto por agente
# ----------------------------------------------------------------------
def _fake_pipeline_generation(monkeypatch):
    import app.rag.agents as agents

    def fake(instructions, prompt, **kwargs):
        if instructions == agents.VERIFIER_INSTRUCTIONS:
            return fake_generation("SUFICIENCIA: suficiente\n\nInforme.")
        return fake_generation("Respuesta final.")

    monkeypatch.setattr(agents, "generate_rag_response_with_metrics", fake)


def test_metricas_por_rol_incluyen_los_campos_de_contexto(monkeypatch):
    import app.rag.agents as agents

    _fake_pipeline_generation(monkeypatch)

    result = agents.run_rag_agent_pipeline(
        "pregunta",
        SOME_HITS,
        conversation_memory="",
        project_memory="",
        verification_enabled=True,
        verifier_context_tokens=8192,
        writer_context_tokens=16384,
    )

    assert "model_context" in result.metrics
    for role in ("verifier", "writer"):
        role_metrics = result.metrics[role]
        assert role_metrics["model"] == settings.chat_model
        assert role_metrics["effective_context_window"] > 0
        assert role_metrics["context_source"] in (
            "profile",
            "ollama",
            "profile+ollama",
            "fallback",
        )
        assert isinstance(role_metrics["context_verified"], bool)
        assert role_metrics["remaining_tokens"] == max(
            role_metrics["effective_context_window"]
            - role_metrics["total_tokens"],
            0,
        )
        # Campo histórico: se mantiene, y siempre es la ventana efectiva.
        assert (
            role_metrics["context_window_tokens"]
            == role_metrics["effective_context_window"]
        )
    assert result.metrics["verifier"]["configured_context_window"] == 8192
    assert result.metrics["writer"]["configured_context_window"] == 16384


def test_ventana_efectiva_se_recorta_a_la_capacidad_real_del_modelo(
    monkeypatch, modelo_restaurable
):
    import app.rag.agents as agents

    settings.chat_model = "llama3:latest"
    _fake_pipeline_generation(monkeypatch)

    result = agents.run_rag_agent_pipeline(
        "pregunta",
        SOME_HITS,
        conversation_memory="",
        project_memory="",
        verification_enabled=True,
        verifier_context_tokens=8192,
        writer_context_tokens=32768,
    )

    writer_metrics = result.metrics["writer"]
    assert writer_metrics["model_context_window"] == 8192
    assert writer_metrics["configured_context_window"] == 32768
    assert writer_metrics["effective_context_window"] == 8192


def test_completion_tokens_conserva_el_razonamiento_oculto_de_gpt_oss(
    monkeypatch, modelo_restaurable
):
    """Ollama cuenta el razonamiento oculto en `eval_count`; no se resta
    aunque `<think>` no llegue nunca al texto visible."""
    import app.rag.agents as agents
    from app.rag.ollama_client import GenerationResult

    settings.chat_model = "gpt-oss:latest"

    def fake(instructions, prompt, **kwargs):
        if instructions == agents.VERIFIER_INSTRUCTIONS:
            return fake_generation("SUFICIENCIA: suficiente\n\nInforme.")
        return GenerationResult(
            content="Respuesta final visible.",
            prompt_tokens=50,
            completion_tokens=900,
            estimated=False,
        )

    monkeypatch.setattr(agents, "generate_rag_response_with_metrics", fake)

    result = agents.run_rag_agent_pipeline(
        "pregunta",
        SOME_HITS,
        conversation_memory="",
        project_memory="",
        verification_enabled=True,
        verifier_context_tokens=8192,
        writer_context_tokens=16384,
    )

    assert result.metrics["writer"]["completion_tokens"] == 900
    assert result.metrics["writer"]["total_tokens"] == 950


def test_contadores_ausentes_quedan_marcados_como_estimados(monkeypatch):
    import app.rag.agents as agents

    def fake(instructions, prompt, **kwargs):
        if instructions == agents.VERIFIER_INSTRUCTIONS:
            return fake_generation("SUFICIENCIA: suficiente\n\nInforme.")
        return GenerationResult(
            content="Respuesta final.",
            prompt_tokens=10,
            completion_tokens=5,
            estimated=True,
        )

    monkeypatch.setattr(agents, "generate_rag_response_with_metrics", fake)

    result = agents.run_rag_agent_pipeline(
        "pregunta",
        SOME_HITS,
        conversation_memory="",
        project_memory="",
        verification_enabled=True,
        verifier_context_tokens=8192,
        writer_context_tokens=16384,
    )

    assert result.metrics["writer"]["estimated"] is True


def test_paridad_entre_via_sincrona_y_streaming(monkeypatch):
    import app.rag.agents as agents
    from app.rag.ollama_client import StreamChunk

    _fake_pipeline_generation(monkeypatch)

    def fake_stream(instructions, prompt, **kwargs):
        answer = "Respuesta final."
        yield StreamChunk(text=answer, done=False)
        yield StreamChunk(
            text="", done=True, metrics=fake_generation(answer)
        )

    monkeypatch.setattr(agents, "stream_rag_response", fake_stream)

    kwargs = dict(
        conversation_memory="",
        project_memory="",
        verification_enabled=True,
        verifier_context_tokens=8192,
        writer_context_tokens=16384,
    )
    sync_result = agents.run_rag_agent_pipeline(
        "pregunta", SOME_HITS, **kwargs
    )
    stream_events = list(
        agents.stream_rag_agent_pipeline("pregunta", SOME_HITS, **kwargs)
    )
    final_event = next(
        event for event in stream_events if event["type"] == "final"
    )

    for role in ("verifier", "writer"):
        assert sync_result.metrics[role] == final_event["metrics"][role]

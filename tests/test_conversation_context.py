from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.db import get_conversation, init_db, save_conversation_turn
from app.rag import ollama_client
from app.rag.memory import (
    build_conversation_context_overview,
    build_memory_aware_retrieval_query,
)


LOCAL_HEADERS = {
    "host": "127.0.0.1:8000",
    "origin": "http://127.0.0.1:8000",
    "sec-fetch-site": "same-origin",
}


# ----------------------------------------------------------------------
# La versión contextual se construye siempre que haya memoria. `main` la
# entrega como una segunda consulta semántica, sin sustituir a la pregunta
# actual ni contaminar su rama léxica.
# ----------------------------------------------------------------------
def test_memoria_construye_una_consulta_contextual_separada():
    question = "Exploraciones habituales\nEndocopia digestia"
    memory = "El turno anterior trataba sobre farmacia hospitalaria."

    contextual = build_memory_aware_retrieval_query(question, memory)

    assert contextual != question
    assert contextual.count(question) == 2
    assert memory in contextual


def test_sin_memoria_no_se_crea_una_segunda_consulta():
    question = "¿Cuáles son sus contraindicaciones?"
    assert build_memory_aware_retrieval_query(question, "") == question


def test_consulta_contextual_conserva_la_memoria_mas_reciente():
    old_context = (
        "TURNOS RECIENTES\n"
        "USUARIO\nPregunta antigua sobre farmacia.\n"
        "ASISTENTE\n" + ("farmacia " * 500) + "\n\n"
    )
    recent_context = (
        "USUARIO\nPregunta sobre una colonoscopia.\n"
        "ASISTENTE\nNo se encontró documentación suficientemente relacionada.\n\n"
        "USUARIO\n¿Y la preparación?\n"
        "ASISTENTE\nDebe seguir las instrucciones documentadas."
    )

    contextual = build_memory_aware_retrieval_query(
        "¿Y cuáles son sus riesgos?",
        old_context + recent_context,
    )

    assert "Pregunta sobre una colonoscopia" in contextual
    assert "No se encontró documentación" not in contextual
    assert "Debe seguir las instrucciones documentadas" in contextual
    assert "Pregunta antigua sobre farmacia" not in contextual
    assert len(contextual) < len(old_context + recent_context)


def test_prepare_query_envia_pregunta_actual_y_contextual_por_separado(
    monkeypatch,
):
    captured: dict = {}

    monkeypatch.setattr(main, "_resolve_project", lambda payload: "default")
    monkeypatch.setattr(main, "get_memory_context", lambda cid: "Tema previo")
    monkeypatch.setattr(main, "get_project_memory_context", lambda pid: "")
    monkeypatch.setattr(
        main,
        "get_project_agent_settings",
        lambda pid: main._default_agent_settings(),
    )

    def fake_retrieve(question, **kwargs):
        captured["question"] = question
        captured.update(kwargs)
        return []

    monkeypatch.setattr(main, "retrieve", fake_retrieve)
    payload = main.QueryRequest(
        question="¿Y sus riesgos?",
        conversation_id="conversation-id",
    )

    main._prepare_query(payload)

    assert captured["question"] == payload.question
    assert captured["lexical_question"] == payload.question
    assert captured["contextual_question"] != payload.question
    assert "Tema previo" in captured["contextual_question"]


@pytest.fixture()
def temporary_database(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "database_path", tmp_path / "context.sqlite3")
    # Ollama accesible o no en la máquina de pruebas no debe cambiar el
    # resultado: se fija a "sin metadatos" para que sólo cuente el perfil.
    monkeypatch.setattr(ollama_client, "get_model_show", lambda model: None)
    init_db()
    yield


@pytest.fixture()
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "database_path", tmp_path / "rag.sqlite3")
    monkeypatch.setattr(main, "_warm_start", lambda: None)
    monkeypatch.setattr(ollama_client, "get_model_show", lambda model: None)
    with TestClient(main.app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


# ----------------------------------------------------------------------
# Persistencia histórica: normalización al leer, sin migraciones ni
# reescritura de datos antiguos.
# ----------------------------------------------------------------------
def test_turno_legado_sin_campos_de_contexto_se_marca_incompleto(
    temporary_database,
):
    conversation_id, _ = save_conversation_turn(
        conversation_id=None,
        project_id=None,
        question="Pregunta",
        answer="Respuesta",
        sources=[],
        chat_model="modelo-antiguo",
        agent_metrics={
            "verifier": {"status": "completed", "usage_percent": 12.0},
            "writer": {"status": "completed", "usage_percent": 20.0},
        },
    )

    conversation = get_conversation(conversation_id)
    turn = conversation["turns"][0]
    assert turn["chat_model"] == "modelo-antiguo"
    assert turn["agent_metrics"]["verifier"]["incomplete_context_data"] is True
    assert turn["agent_metrics"]["writer"]["incomplete_context_data"] is True


def test_turno_completo_no_se_marca_incompleto(temporary_database):
    def role_metrics(percent: float) -> dict:
        return {
            "status": "completed",
            "model": "qwen3.5:0.8b",
            "model_context_window": 32768,
            "configured_context_window": 8192,
            "effective_context_window": 8192,
            "context_source": "profile",
            "usage_percent": percent,
        }

    conversation_id, _ = save_conversation_turn(
        conversation_id=None,
        project_id=None,
        question="Pregunta",
        answer="Respuesta",
        sources=[],
        chat_model="qwen3.5:0.8b",
        agent_metrics={
            "verifier": role_metrics(12.0),
            "writer": role_metrics(20.0),
        },
    )

    conversation = get_conversation(conversation_id)
    turn = conversation["turns"][0]
    assert turn["agent_metrics"]["verifier"]["incomplete_context_data"] is False
    assert turn["agent_metrics"]["writer"]["incomplete_context_data"] is False


def test_turno_sin_metricas_de_agentes_no_rompe_la_lectura(
    temporary_database,
):
    conversation_id, _ = save_conversation_turn(
        conversation_id=None,
        project_id=None,
        question="Pregunta",
        answer="Respuesta",
        sources=[],
        chat_model="modelo-antiguo",
    )

    conversation = get_conversation(conversation_id)
    assert conversation["turns"][0]["agent_metrics"] == {}


def test_cambiar_de_modelo_no_altera_metricas_historicas(
    temporary_database, monkeypatch
):
    def role_metrics() -> dict:
        return {
            "status": "completed",
            "model": "llama3:latest",
            "model_context_window": 8192,
            "configured_context_window": 8192,
            "effective_context_window": 8192,
            "context_source": "profile",
            "usage_percent": 40.0,
        }

    conversation_id, _ = save_conversation_turn(
        conversation_id=None,
        project_id=None,
        question="Pregunta",
        answer="Respuesta",
        sources=[],
        chat_model="llama3:latest",
        agent_metrics={
            "verifier": role_metrics(),
            "writer": role_metrics(),
        },
    )

    monkeypatch.setattr(settings, "chat_model", "gpt-oss:latest")

    conversation = get_conversation(conversation_id)
    turn = conversation["turns"][0]
    assert turn["chat_model"] == "llama3:latest"
    assert turn["agent_metrics"]["writer"]["model"] == "llama3:latest"
    assert turn["agent_metrics"]["writer"]["model_context_window"] == 8192


# ----------------------------------------------------------------------
# build_conversation_context_overview(): historial almacenado vs activo
# ----------------------------------------------------------------------
def test_overview_conversacion_inexistente_da_none(temporary_database):
    assert build_conversation_context_overview("no-existe") is None


def test_overview_diferencia_historial_almacenado_de_activo(
    temporary_database, monkeypatch
):
    monkeypatch.setattr(settings, "memory_recent_turns", 2)
    conversation_id = None
    for index in range(5):
        conversation_id, _ = save_conversation_turn(
            conversation_id=conversation_id,
            project_id=None,
            question=" ".join([f"pregunta{index}"] * 40),
            answer=" ".join([f"respuesta{index}"] * 40),
            sources=[],
            chat_model=settings.chat_model,
            agent_metrics={
                "verifier": {"status": "completed", "usage_percent": 10.0},
                "writer": {"status": "completed", "usage_percent": 15.0},
            },
        )

    overview = build_conversation_context_overview(conversation_id)
    assert overview["history"]["total_turns"] == 5
    assert overview["history"]["recent_turns_included"] == 2
    assert (
        overview["history"]["stored_estimated_tokens"]
        > overview["history"]["recent_estimated_tokens"]
    )
    assert overview["next_inference_estimate"]["estimated"] is True
    assert overview["next_inference_estimate"]["effective_context_window"] > 0


def test_overview_no_suma_porcentajes_de_distintos_turnos(
    temporary_database,
):
    conversation_id = None
    for percent in (10.0, 90.0):
        conversation_id, _ = save_conversation_turn(
            conversation_id=conversation_id,
            project_id=None,
            question="Pregunta",
            answer="Respuesta",
            sources=[],
            chat_model=settings.chat_model,
            agent_metrics={
                "verifier": {
                    "status": "completed",
                    "usage_percent": percent,
                },
                "writer": {
                    "status": "completed",
                    "usage_percent": percent,
                },
            },
        )

    overview = build_conversation_context_overview(conversation_id)
    # Sólo el último turno cuenta: ni se suma ni se promedia con el
    # primero (10%). Cada llamada a Ollama tiene su propia ventana.
    assert overview["last_inference"]["maximum_usage_percent"] == 90.0


# ----------------------------------------------------------------------
# Endpoint HTTP
# ----------------------------------------------------------------------
def test_endpoint_context_de_conversacion(client: TestClient):
    conversation_id, _ = save_conversation_turn(
        conversation_id=None,
        project_id=None,
        question="Pregunta",
        answer="Respuesta",
        sources=[],
        chat_model=settings.chat_model,
        agent_metrics={
            "verifier": {"status": "completed", "usage_percent": 10.0},
            "writer": {"status": "completed", "usage_percent": 20.0},
        },
    )

    response = client.get(
        f"/api/conversations/{conversation_id}/context",
        headers=LOCAL_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == conversation_id
    assert "current_model" in payload
    assert payload["history"]["total_turns"] == 1
    assert "warnings" in payload


def test_endpoint_context_404_si_no_existe(client: TestClient):
    response = client.get(
        "/api/conversations/no-existe/context",
        headers=LOCAL_HEADERS,
    )
    assert response.status_code == 404


def test_endpoint_status_expone_model_context(client: TestClient):
    response = client.get("/api/status", headers=LOCAL_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["capabilities"]["context_inspector"] is True
    assert payload["model_context"]["model"] == settings.chat_model
    assert payload["model_context"]["model_context_window"] > 0

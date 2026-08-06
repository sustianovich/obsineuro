from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings


LOCAL_HEADERS = {
    "host": "127.0.0.1:8000",
    "origin": "http://127.0.0.1:8000",
    "sec-fetch-site": "same-origin",
}


@pytest.fixture()
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "database_path", tmp_path / "rag.sqlite3")
    monkeypatch.setattr(main, "_warm_start", lambda: None)
    with TestClient(main.app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def test_rejects_non_loopback_host(client: TestClient):
    response = client.get("/health", headers={"host": "example.test"})

    assert response.status_code == 400
    assert response.json() == {"detail": "Cabecera Host no permitida."}


def test_rejects_cross_origin_mutation(client: TestClient):
    response = client.post(
        "/api/query",
        json={},
        headers={
            "host": "127.0.0.1:8000",
            "origin": "https://example.test",
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Solicitud no permitida."}


def test_rejects_cross_site_fetch_metadata(client: TestClient):
    response = client.post(
        "/api/query",
        json={},
        headers={
            **LOCAL_HEADERS,
            "sec-fetch-site": "cross-site",
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Solicitud no permitida."}


def test_allows_same_origin_mutation_to_reach_validation(client: TestClient):
    response = client.post(
        "/api/query",
        json={},
        headers=LOCAL_HEADERS,
    )

    assert response.status_code == 422


def test_http_exception_does_not_expose_internal_error(
    client: TestClient,
    monkeypatch,
):
    def fail_to_configure(_: str) -> dict:
        raise RuntimeError("TOKEN_SUPER_SECRETO")

    monkeypatch.setattr(main, "configure_chat_model_profile", fail_to_configure)
    response = client.post(
        "/api/config/chat-profile",
        json={"profile": "ligero"},
        headers=LOCAL_HEADERS,
    )

    assert response.status_code == 400
    assert response.json() == {"detail": main.PUBLIC_OPERATION_ERROR}
    assert "TOKEN_SUPER_SECRETO" not in response.text


def test_stream_error_does_not_expose_internal_error(
    client: TestClient,
    monkeypatch,
):
    def fail_to_prepare(_: object) -> dict:
        raise RuntimeError("TOKEN_SUPER_SECRETO")

    monkeypatch.setattr(main, "_prepare_query", fail_to_prepare)
    response = client.post(
        "/api/query/stream",
        json={"question": "Pregunta de prueba"},
        headers=LOCAL_HEADERS,
    )

    assert response.status_code == 200
    assert main.PUBLIC_QUERY_ERROR in response.text
    assert "TOKEN_SUPER_SECRETO" not in response.text


def test_status_redacts_errors_from_dependencies(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        main,
        "get_ollama_status",
        lambda: {"accessible": False, "error": "TOKEN_SUPER_SECRETO"},
    )

    response = client.get("/api/status", headers=LOCAL_HEADERS)

    assert response.status_code == 200
    assert response.json()["ollama"]["error"] == main.PUBLIC_OPERATION_ERROR
    assert "TOKEN_SUPER_SECRETO" not in response.text

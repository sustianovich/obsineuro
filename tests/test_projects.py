from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.db import (
    DEFAULT_PROJECT_ID,
    create_project,
    delete_project,
    get_connection,
    get_conversation,
    init_db,
    list_conversations,
    list_projects,
    project_exists,
    save_conversation_turn,
    utc_now,
)


@pytest.fixture()
def temporary_database(tmp_path: Path):
    original_database_path = settings.database_path
    settings.database_path = tmp_path / "projects.sqlite3"
    init_db()
    try:
        yield
    finally:
        settings.database_path = original_database_path


def test_delete_project_moves_conversations_to_default(
    temporary_database,
):
    project = create_project("PDPCM pruebas")
    conversation_id, _ = save_conversation_turn(
        conversation_id=None,
        project_id=project["id"],
        question="Pregunta de prueba",
        answer="Respuesta de prueba",
        sources=[],
        chat_model="test-model",
    )
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO project_memory(
                project_id, summary, summarized_until_turn_id,
                summarized_turn_count, summary_model, updated_at
            )
            VALUES (?, ?, 1, 1, ?, ?)
            """,
            (project["id"], "Resumen compartido", "test-model", utc_now()),
        )
        connection.commit()

    result = delete_project(project["id"])

    assert result == {
        "deleted": True,
        "project_id": project["id"],
        "project_name": "PDPCM pruebas",
        "moved_conversations": 1,
        "fallback_project_id": DEFAULT_PROJECT_ID,
    }
    assert not project_exists(project["id"])
    assert {item["id"] for item in list_projects()} == {
        DEFAULT_PROJECT_ID
    }
    assert [item["id"] for item in list_conversations(DEFAULT_PROJECT_ID)] == [
        conversation_id
    ]
    assert get_conversation(conversation_id)["project_id"] == (
        DEFAULT_PROJECT_ID
    )


def test_delete_project_rejects_default(temporary_database):
    with pytest.raises(ValueError, match="General"):
        delete_project(DEFAULT_PROJECT_ID)

    assert project_exists(DEFAULT_PROJECT_ID)

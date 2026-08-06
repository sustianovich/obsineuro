"""Conversaciones, sus turnos y la memoria conversacional resumida."""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from uuid import uuid4

from app.db.schema import DEFAULT_PROJECT_ID, get_connection, transaction, utc_now


def conversation_title(question: str, max_length: int = 72) -> str:
    normalized = " ".join(question.split()).strip()
    if not normalized:
        return "Nueva conversación"
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def conversation_exists(conversation_id: str) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return row is not None


def _memory_status_from_connection(
    connection: sqlite3.Connection,
    conversation_id: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
            c.memory_enabled,
            COUNT(t.id) AS total_turns,
            COALESCE(m.summary, '') AS summary,
            COALESCE(m.summarized_until_turn_id, 0)
                AS summarized_until_turn_id,
            COALESCE(m.summarized_turn_count, 0)
                AS summarized_turn_count,
            COALESCE(m.summary_model, '') AS summary_model,
            m.last_error,
            m.updated_at AS memory_updated_at
        FROM conversations AS c
        LEFT JOIN conversation_turns AS t
            ON t.conversation_id = c.id
        LEFT JOIN conversation_memory AS m
            ON m.conversation_id = c.id
        WHERE c.id = ?
        GROUP BY c.id
        """,
        (conversation_id,),
    ).fetchone()
    if row is None:
        return None
    total_turns = int(row["total_turns"])
    summarized_turns = int(row["summarized_turn_count"])
    return {
        "enabled": bool(row["memory_enabled"]),
        "has_summary": bool(str(row["summary"]).strip()),
        "summary": str(row["summary"]),
        "summarized_until_turn_id": int(
            row["summarized_until_turn_id"]
        ),
        "summarized_turns": summarized_turns,
        "pending_turns": max(0, total_turns - summarized_turns),
        "total_turns": total_turns,
        "summary_model": str(row["summary_model"]),
        "last_error": (
            str(row["last_error"]) if row["last_error"] else None
        ),
        "updated_at": (
            str(row["memory_updated_at"])
            if row["memory_updated_at"]
            else None
        ),
    }


def get_conversation_memory_status(
    conversation_id: str,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        return _memory_status_from_connection(connection, conversation_id)


def set_conversation_memory_enabled(
    conversation_id: str,
    enabled: bool,
) -> dict[str, Any] | None:
    with transaction() as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET memory_enabled = ?, updated_at = ?
            WHERE id = ?
            """,
            (int(enabled), utc_now(), conversation_id),
        )
        if cursor.rowcount == 0:
            return None
        return _memory_status_from_connection(connection, conversation_id)


def load_conversation_memory_context(
    conversation_id: str,
    *,
    recent_turn_limit: int,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        status = _memory_status_from_connection(
            connection,
            conversation_id,
        )
        if status is None:
            return None
        rows = connection.execute(
            """
            SELECT id, question, answer
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, recent_turn_limit),
        ).fetchall()
    turns = [
        {
            "id": int(row["id"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
        }
        for row in reversed(rows)
    ]
    return {
        "enabled": status["enabled"],
        "summary": status["summary"],
        "turns": turns,
    }


def get_conversation_memory_batch(
    conversation_id: str,
    *,
    batch_size: int,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        conversation = connection.execute(
            """
            SELECT memory_enabled
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None
        memory = connection.execute(
            """
            SELECT
                summary,
                summarized_until_turn_id,
                summarized_turn_count
            FROM conversation_memory
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        summarized_until = (
            int(memory["summarized_until_turn_id"]) if memory else 0
        )
        rows = connection.execute(
            """
            SELECT id, question, answer
            FROM conversation_turns
            WHERE conversation_id = ? AND id > ?
            ORDER BY id
            LIMIT ?
            """,
            (conversation_id, summarized_until, batch_size),
        ).fetchall()
    if not bool(conversation["memory_enabled"]) or len(rows) < batch_size:
        return None
    return {
        "previous_summary": str(memory["summary"]) if memory else "",
        "previous_summarized_turn_count": (
            int(memory["summarized_turn_count"]) if memory else 0
        ),
        "turns": [
            {
                "id": int(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }
            for row in rows
        ],
    }


def save_conversation_memory_summary(
    *,
    conversation_id: str,
    summary: str,
    summarized_until_turn_id: int,
    summarized_turn_count: int,
    summary_model: str,
) -> bool:
    now = utc_now()
    with transaction() as connection:
        current = connection.execute(
            """
            SELECT summarized_until_turn_id
            FROM conversation_memory
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        current_turn_id = (
            int(current["summarized_until_turn_id"]) if current else 0
        )
        if summarized_until_turn_id <= current_turn_id:
            return False
        connection.execute(
            """
            INSERT INTO conversation_memory(
                conversation_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                last_error,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                summary = excluded.summary,
                summarized_until_turn_id =
                    excluded.summarized_until_turn_id,
                summarized_turn_count = excluded.summarized_turn_count,
                summary_model = excluded.summary_model,
                last_error = NULL,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                now,
            ),
        )
        return True


def set_conversation_memory_error(
    conversation_id: str,
    error: str,
) -> None:
    now = utc_now()
    safe_error = " ".join(error.split()).strip()[:500]
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO conversation_memory(
                conversation_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                last_error,
                updated_at
            )
            VALUES (?, '', 0, 0, '', ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            """,
            (conversation_id, safe_error, now),
        )


def save_conversation_turn(
    *,
    conversation_id: str | None,
    project_id: str | None = None,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    chat_model: str,
    memory_enabled: bool = True,
    agent_metrics: dict[str, Any] | None = None,
) -> tuple[str, int]:
    now = utc_now()
    with transaction() as connection:
        current_turns = 0
        if conversation_id:
            row = connection.execute(
                "SELECT id, project_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(
                    f"No existe la conversación: {conversation_id}"
                )
            current_turns = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM conversation_turns
                    WHERE conversation_id = ?
                    """,
                    (conversation_id,),
                ).fetchone()["total"]
            )
        else:
            project_id = project_id or DEFAULT_PROJECT_ID
            project = connection.execute(
                "SELECT id FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if project is None:
                raise KeyError(f"No existe el proyecto: {project_id}")
            conversation_id = uuid4().hex
            connection.execute(
                """
                INSERT INTO conversations(
                    id, project_id, title, memory_enabled,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    project_id,
                    conversation_title(question),
                    int(memory_enabled),
                    now,
                    now,
                ),
            )

        cursor = connection.execute(
            """
            INSERT INTO conversation_turns(
                conversation_id, question, answer,
                sources_json, chat_model, agent_metrics_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                question,
                answer,
                json.dumps(sources, ensure_ascii=False, default=str),
                chat_model,
                json.dumps(
                    agent_metrics or {},
                    ensure_ascii=False,
                    default=str,
                ),
                now,
            ),
        )
        if current_turns == 0:
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (conversation_title(question), now, conversation_id),
            )
        else:
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (now, conversation_id),
            )
        connection.execute(
            """
            UPDATE conversations
            SET memory_enabled = ?
            WHERE id = ?
            """,
            (int(memory_enabled), conversation_id),
        )
        connection.execute(
            """
            UPDATE projects
            SET updated_at = ?
            WHERE id = (
                SELECT project_id
                FROM conversations
                WHERE id = ?
            )
            """,
            (now, conversation_id),
        )
        return conversation_id, int(cursor.lastrowid)


def list_conversations(
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.id,
                c.project_id,
                c.title,
                c.memory_enabled,
                c.created_at,
                c.updated_at,
                COUNT(t.id) AS turn_count
            FROM conversations AS c
            LEFT JOIN conversation_turns AS t
                ON t.conversation_id = c.id
            WHERE (? IS NULL OR c.project_id = ?)
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """,
            (project_id, project_id),
        ).fetchall()
    return [
        {
            "id": str(row["id"]),
            "project_id": str(row["project_id"]),
            "title": str(row["title"]),
            "memory_enabled": bool(row["memory_enabled"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "turn_count": int(row["turn_count"]),
        }
        for row in rows
    ]


_AGENT_CONTEXT_FIELDS = (
    "model",
    "model_context_window",
    "configured_context_window",
    "effective_context_window",
    "context_source",
)


def _normalize_agent_role_metrics(raw: Any) -> dict[str, Any] | None:
    """Decora una entrada verifier/writer con una marca de completitud.

    No reescribe nada en la base de datos: un turno guardado antes de que
    existieran estos campos (o guardado con un modelo distinto al activo
    ahora) sigue mostrando exactamente lo que se guardó, sólo que
    marcado como incompleto en vez de mostrar huecos silenciosos.
    """
    if not isinstance(raw, dict):
        return None
    normalized = dict(raw)
    normalized["incomplete_context_data"] = any(
        normalized.get(field) is None for field in _AGENT_CONTEXT_FIELDS
    )
    return normalized


def _normalize_agent_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    normalized = dict(metrics)
    for role in ("verifier", "writer"):
        role_metrics = _normalize_agent_role_metrics(metrics.get(role))
        if role_metrics is not None:
            normalized[role] = role_metrics
    return normalized


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        conversation = connection.execute(
            """
            SELECT
                id, project_id, title, memory_enabled,
                created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None
        rows = connection.execute(
            """
            SELECT
                id, question, answer, sources_json,
                chat_model, agent_metrics_json, created_at
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()

    turns: list[dict[str, Any]] = []
    for row in rows:
        try:
            sources = json.loads(row["sources_json"] or "[]")
        except json.JSONDecodeError:
            sources = []
        try:
            agent_metrics = json.loads(
                row["agent_metrics_json"] or "{}"
            )
        except json.JSONDecodeError:
            agent_metrics = {}
        turns.append(
            {
                "id": int(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "sources": sources if isinstance(sources, list) else [],
                "chat_model": str(row["chat_model"]),
                "agent_metrics": _normalize_agent_metrics(agent_metrics),
                "created_at": str(row["created_at"]),
            }
        )
    memory = get_conversation_memory_status(conversation_id)
    return {
        "id": str(conversation["id"]),
        "project_id": str(conversation["project_id"]),
        "title": str(conversation["title"]),
        "memory": memory,
        "created_at": str(conversation["created_at"]),
        "updated_at": str(conversation["updated_at"]),
        "turns": turns,
    }


def delete_conversation(conversation_id: str) -> bool:
    with transaction() as connection:
        row = connection.execute(
            "SELECT project_id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return False
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM project_memory WHERE project_id = ?",
            (str(row["project_id"]),),
        )
        return cursor.rowcount > 0

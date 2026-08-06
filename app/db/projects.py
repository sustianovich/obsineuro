"""Proyectos, su memoria compartida y los ajustes de los agentes."""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from uuid import uuid4

from app.config import settings
from app.db.schema import (
    DEFAULT_PROJECT_ID,
    DEFAULT_VERIFIER_CONTEXT_TOKENS,
    DEFAULT_WRITER_CONTEXT_TOKENS,
    MAX_AGENT_CONTEXT_TOKENS,
    MIN_AGENT_CONTEXT_TOKENS,
    get_connection,
    transaction,
    utc_now,
)


def project_exists(project_id: str) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
    return row is not None


def _validate_context_tokens(value: int, label: str) -> int:
    normalized = int(value)
    if not MIN_AGENT_CONTEXT_TOKENS <= normalized <= MAX_AGENT_CONTEXT_TOKENS:
        raise ValueError(
            f"{label} debe estar entre {MIN_AGENT_CONTEXT_TOKENS} y "
            f"{MAX_AGENT_CONTEXT_TOKENS} tokens."
        )
    return normalized


def _project_agent_settings_from_row(
    row: sqlite3.Row,
) -> dict[str, Any]:
    return {
        "verification_enabled": bool(row["verification_enabled"]),
        "project_memory_enabled": bool(row["project_memory_enabled"]),
        "verifier_context_tokens": int(
            row["verifier_context_tokens"]
        ),
        "writer_context_tokens": int(row["writer_context_tokens"]),
    }


def get_project_agent_settings(
    project_id: str,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                verification_enabled,
                project_memory_enabled,
                verifier_context_tokens,
                writer_context_tokens
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()
    return _project_agent_settings_from_row(row) if row else None


def update_project_agent_settings(
    project_id: str,
    *,
    verification_enabled: bool,
    project_memory_enabled: bool,
    verifier_context_tokens: int,
    writer_context_tokens: int,
) -> dict[str, Any] | None:
    verifier_tokens = _validate_context_tokens(
        verifier_context_tokens,
        "La ventana del verificador",
    )
    writer_tokens = _validate_context_tokens(
        writer_context_tokens,
        "La ventana del redactor",
    )
    with transaction() as connection:
        cursor = connection.execute(
            """
            UPDATE projects
            SET
                verification_enabled = ?,
                project_memory_enabled = ?,
                verifier_context_tokens = ?,
                writer_context_tokens = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                int(verification_enabled),
                int(project_memory_enabled),
                verifier_tokens,
                writer_tokens,
                utc_now(),
                project_id,
            ),
        )
        if cursor.rowcount == 0:
            return None
    return get_project_agent_settings(project_id)


def _project_memory_status_from_connection(
    connection: sqlite3.Connection,
    project_id: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
            p.project_memory_enabled,
            COUNT(t.id) AS total_turns,
            COALESCE(m.summary, '') AS summary,
            COALESCE(m.summarized_until_turn_id, 0)
                AS summarized_until_turn_id,
            COALESCE(m.summarized_turn_count, 0)
                AS summarized_turn_count,
            COALESCE(m.summary_model, '') AS summary_model,
            m.last_error,
            m.updated_at AS memory_updated_at
        FROM projects AS p
        LEFT JOIN conversations AS c ON c.project_id = p.id
        LEFT JOIN conversation_turns AS t
            ON t.conversation_id = c.id
        LEFT JOIN project_memory AS m ON m.project_id = p.id
        WHERE p.id = ?
        GROUP BY p.id
        """,
        (project_id,),
    ).fetchone()
    if row is None:
        return None
    total_turns = int(row["total_turns"])
    summarized_turns = int(row["summarized_turn_count"])
    return {
        "enabled": bool(row["project_memory_enabled"]),
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


def get_project_memory_status(
    project_id: str,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        return _project_memory_status_from_connection(
            connection,
            project_id,
        )


def load_project_memory_context(
    project_id: str,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        status = _project_memory_status_from_connection(
            connection,
            project_id,
        )
    if status is None:
        return None
    return {
        "enabled": status["enabled"],
        "summary": status["summary"],
    }


def get_project_memory_batch(
    project_id: str,
    *,
    batch_size: int,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        project = connection.execute(
            """
            SELECT project_memory_enabled
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()
        if project is None:
            return None
        memory = connection.execute(
            """
            SELECT
                summary,
                summarized_until_turn_id,
                summarized_turn_count
            FROM project_memory
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        summarized_until = (
            int(memory["summarized_until_turn_id"]) if memory else 0
        )
        rows = connection.execute(
            """
            SELECT
                t.id,
                c.title AS conversation_title,
                t.question,
                t.answer
            FROM conversation_turns AS t
            JOIN conversations AS c ON c.id = t.conversation_id
            WHERE c.project_id = ? AND t.id > ?
            ORDER BY t.id
            LIMIT ?
            """,
            (project_id, summarized_until, batch_size),
        ).fetchall()
    if (
        not bool(project["project_memory_enabled"])
        or len(rows) < batch_size
    ):
        return None
    return {
        "previous_summary": str(memory["summary"]) if memory else "",
        "previous_summarized_turn_count": (
            int(memory["summarized_turn_count"]) if memory else 0
        ),
        "turns": [
            {
                "id": int(row["id"]),
                "conversation_title": str(row["conversation_title"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }
            for row in rows
        ],
    }


def save_project_memory_summary(
    *,
    project_id: str,
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
            FROM project_memory
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        current_turn_id = (
            int(current["summarized_until_turn_id"]) if current else 0
        )
        if summarized_until_turn_id <= current_turn_id:
            return False
        connection.execute(
            """
            INSERT INTO project_memory(
                project_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                last_error,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                summary = excluded.summary,
                summarized_until_turn_id =
                    excluded.summarized_until_turn_id,
                summarized_turn_count = excluded.summarized_turn_count,
                summary_model = excluded.summary_model,
                last_error = NULL,
                updated_at = excluded.updated_at
            """,
            (
                project_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                now,
            ),
        )
        return True


def set_project_memory_error(project_id: str, error: str) -> None:
    now = utc_now()
    safe_error = " ".join(error.split()).strip()[:500]
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO project_memory(
                project_id,
                summary,
                summarized_until_turn_id,
                summarized_turn_count,
                summary_model,
                last_error,
                updated_at
            )
            VALUES (?, '', 0, 0, '', ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            """,
            (project_id, safe_error, now),
        )


def _agent_usage_point(
    metrics: dict[str, Any],
    role: str,
) -> dict[str, Any] | None:
    raw = metrics.get(role)
    if not isinstance(raw, dict):
        return None
    if raw.get("status") not in {"completed", "degraded"}:
        return None
    context_window = int(raw.get("context_window_tokens") or 0)
    prompt_tokens = int(raw.get("prompt_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or 0)
    total_tokens = int(
        raw.get("total_tokens")
        or prompt_tokens + completion_tokens
    )
    if context_window <= 0 or total_tokens < 0:
        return None
    return {
        "context_window_tokens": context_window,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "usage_percent": round(
            min(100.0, total_tokens / context_window * 100),
            2,
        ),
        "estimated": bool(raw.get("estimated", False)),
    }


def _project_context_usage(
    metrics_rows: list[sqlite3.Row],
) -> dict[str, Any]:
    role_points: dict[str, list[dict[str, Any]]] = {
        "verifier": [],
        "writer": [],
    }
    latest_metrics: dict[str, Any] = {}
    for row in metrics_rows:
        try:
            metrics = json.loads(row["agent_metrics_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(metrics, dict):
            continue
        latest_metrics = metrics
        for role in role_points:
            point = _agent_usage_point(metrics, role)
            if point is not None:
                role_points[role].append(point)

    output: dict[str, Any] = {}
    latest_percentages: list[float] = []
    maximum_percentages: list[float] = []
    all_percentages: list[float] = []
    for role, points in role_points.items():
        if not points:
            output[role] = {
                "latest": None,
                "maximum_percent": None,
                "average_percent": None,
                "samples": 0,
            }
            continue
        percentages = [
            float(point["usage_percent"]) for point in points
        ]
        latest = _agent_usage_point(latest_metrics, role)
        if latest is not None:
            latest_percentages.append(float(latest["usage_percent"]))
        maximum_percentages.append(max(percentages))
        all_percentages.extend(percentages)
        output[role] = {
            "latest": latest,
            "maximum_percent": round(max(percentages), 2),
            "average_percent": round(
                sum(percentages) / len(percentages),
                2,
            ),
            "samples": len(points),
        }

    output.update(
        {
            "latest_percent": (
                round(max(latest_percentages), 2)
                if latest_percentages
                else None
            ),
            "maximum_percent": (
                round(max(maximum_percentages), 2)
                if maximum_percentages
                else None
            ),
            "average_percent": (
                round(sum(all_percentages) / len(all_percentages), 2)
                if all_percentages
                else None
            ),
        }
    )
    return output


def _validated_project_name(name: str) -> str:
    normalized = " ".join(name.split()).strip()
    if not normalized:
        raise ValueError("El nombre del proyecto no puede estar vacío.")
    if len(normalized) > 80:
        raise ValueError(
            "El nombre del proyecto no puede superar 80 caracteres."
        )
    return normalized


def create_project(name: str) -> dict[str, Any]:
    normalized = _validated_project_name(name)
    project_id = uuid4().hex
    now = utc_now()
    verifier_context_tokens = _validate_context_tokens(
        getattr(
            settings,
            "default_verifier_context_tokens",
            DEFAULT_VERIFIER_CONTEXT_TOKENS,
        ),
        "La ventana del verificador",
    )
    writer_context_tokens = _validate_context_tokens(
        getattr(
            settings,
            "default_writer_context_tokens",
            DEFAULT_WRITER_CONTEXT_TOKENS,
        ),
        "La ventana del redactor",
    )
    try:
        with transaction() as connection:
            connection.execute(
                """
                INSERT INTO projects(
                    id,
                    name,
                    verification_enabled,
                    project_memory_enabled,
                    verifier_context_tokens,
                    writer_context_tokens,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, 1, 1, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    normalized,
                    verifier_context_tokens,
                    writer_context_tokens,
                    now,
                    now,
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"Ya existe un proyecto llamado «{normalized}»."
        ) from exc
    return {
        "id": project_id,
        "name": normalized,
        "created_at": now,
        "updated_at": now,
        "conversation_count": 0,
        "agent_settings": {
            "verification_enabled": True,
            "project_memory_enabled": True,
            "verifier_context_tokens": (
                verifier_context_tokens
            ),
            "writer_context_tokens": writer_context_tokens,
        },
        "context_usage": _project_context_usage([]),
    }


def list_projects() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                p.id,
                p.name,
                p.verification_enabled,
                p.project_memory_enabled,
                p.verifier_context_tokens,
                p.writer_context_tokens,
                p.created_at,
                p.updated_at,
                COUNT(c.id) AS conversation_count
            FROM projects AS p
            LEFT JOIN conversations AS c ON c.project_id = p.id
            GROUP BY p.id
            ORDER BY
                CASE WHEN p.id = ? THEN 0 ELSE 1 END,
                p.name COLLATE NOCASE
            """,
            (DEFAULT_PROJECT_ID,),
        ).fetchall()
        metric_rows = connection.execute(
            """
            SELECT c.project_id, t.agent_metrics_json
            FROM conversation_turns AS t
            JOIN conversations AS c ON c.id = t.conversation_id
            ORDER BY t.id
            """
        ).fetchall()
        memory_rows = connection.execute(
            """
            SELECT
                p.id AS project_id,
                COUNT(t.id) AS total_turns,
                COALESCE(m.summary, '') AS summary,
                COALESCE(m.summarized_turn_count, 0)
                    AS summarized_turn_count,
                m.last_error
            FROM projects AS p
            LEFT JOIN conversations AS c ON c.project_id = p.id
            LEFT JOIN conversation_turns AS t
                ON t.conversation_id = c.id
            LEFT JOIN project_memory AS m ON m.project_id = p.id
            GROUP BY p.id
            """
        ).fetchall()
    metrics_by_project: dict[str, list[sqlite3.Row]] = {}
    for metric_row in metric_rows:
        metrics_by_project.setdefault(
            str(metric_row["project_id"]),
            [],
        ).append(metric_row)
    memory_by_project = {
        str(row["project_id"]): {
            "enabled": True,
            "has_summary": bool(str(row["summary"]).strip()),
            "summarized_turns": int(row["summarized_turn_count"]),
            "pending_turns": max(
                0,
                int(row["total_turns"])
                - int(row["summarized_turn_count"]),
            ),
            "total_turns": int(row["total_turns"]),
            "last_error": (
                str(row["last_error"]) if row["last_error"] else None
            ),
        }
        for row in memory_rows
    }
    projects = []
    for row in rows:
        project_id = str(row["id"])
        project_memory = memory_by_project.get(project_id, {})
        project_memory["enabled"] = bool(
            row["project_memory_enabled"]
        )
        projects.append(
            {
            "id": str(row["id"]),
            "name": str(row["name"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "conversation_count": int(row["conversation_count"]),
            "agent_settings": _project_agent_settings_from_row(row),
            "context_usage": _project_context_usage(
                metrics_by_project.get(project_id, [])
            ),
            "memory": project_memory,
        }
        )
    return projects


def rename_project(project_id: str, name: str) -> dict[str, Any] | None:
    normalized = _validated_project_name(name)
    now = utc_now()
    try:
        with transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE projects
                SET name = ?, updated_at = ?
                WHERE id = ?
                """,
                (normalized, now, project_id),
            )
            if cursor.rowcount == 0:
                return None
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM conversations
                    WHERE project_id = ?
                    """,
                    (project_id,),
                ).fetchone()["total"]
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"Ya existe un proyecto llamado «{normalized}»."
        ) from exc
    return {
        "id": project_id,
        "name": normalized,
        "updated_at": now,
        "conversation_count": count,
    }


def delete_project(project_id: str) -> dict[str, Any] | None:
    if project_id == DEFAULT_PROJECT_ID:
        raise ValueError("El proyecto General no se puede eliminar.")

    now = utc_now()
    with transaction() as connection:
        row = connection.execute(
            "SELECT id, name FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            return None

        conversation_count = int(
            connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM conversations
                WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()["total"]
        )
        connection.execute(
            """
            UPDATE conversations
            SET project_id = ?, updated_at = ?
            WHERE project_id = ?
            """,
            (DEFAULT_PROJECT_ID, now, project_id),
        )
        connection.execute(
            """
            DELETE FROM project_memory
            WHERE project_id = ?
            """,
            (project_id,),
        )
        cursor = connection.execute(
            "DELETE FROM projects WHERE id = ?",
            (project_id,),
        )
        connection.execute(
            """
            UPDATE projects
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, DEFAULT_PROJECT_ID),
        )

    return {
        "deleted": cursor.rowcount > 0,
        "project_id": project_id,
        "project_name": str(row["name"]),
        "moved_conversations": conversation_count,
        "fallback_project_id": DEFAULT_PROJECT_ID,
    }

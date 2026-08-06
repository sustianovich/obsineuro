"""Fachada pública del paquete de base de datos.

`app/db.py` era un único módulo de 2803 líneas; se dividió en submódulos
por responsabilidad (`schema`, `fts`, `links`, `documents`,
`conversations`, `projects`). Este archivo re-exporta exactamente los
mismos nombres públicos que antes ofrecía `app.db`, así que ningún import
externo (`from app.db import ...`) tiene que cambiar.
"""

from __future__ import annotations

from app.db.schema import (
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_NAME,
    DEFAULT_VERIFIER_CONTEXT_TOKENS,
    DEFAULT_WRITER_CONTEXT_TOKENS,
    FTS_ERROR_META_KEY,
    FTS_INDEX_VERSION,
    FTS_VERSION_META_KEY,
    GRAPH_INDEX_VERSION,
    GRAPH_VERSION_META_KEY,
    MAX_AGENT_CONTEXT_TOKENS,
    MIN_AGENT_CONTEXT_TOKENS,
    get_connection,
    get_meta,
    init_db,
    transaction,
    utc_now,
)
from app.db.fts import get_fts5_status, search_chunks_fts
from app.db.links import _rebuild_document_links, get_document_graph_status
from app.db.documents import (
    commit_staged_index,
    delete_documents_not_in,
    discard_staged_index,
    get_document_hashes,
    get_document_statuses,
    get_embedding_dimensions,
    get_stats,
    replace_document,
    reset_staged_index,
    resolve_staged_document_links,
    stage_document,
)
from app.db.conversations import (
    conversation_exists,
    conversation_title,
    delete_conversation,
    get_conversation,
    get_conversation_memory_batch,
    get_conversation_memory_status,
    list_conversations,
    load_conversation_memory_context,
    save_conversation_memory_summary,
    save_conversation_turn,
    set_conversation_memory_enabled,
    set_conversation_memory_error,
)
# `_rebuild_document_links` no es parte de la API pública (por eso no está
# en `__all__`): sólo se reexporta porque tests/test_graph_retrieval.py hace
# `monkeypatch.setattr(db, "_rebuild_document_links", ...)` sobre este
# módulo. Si se borra, ese test deja de poder engancharse aquí.
from app.db.projects import (
    create_project,
    delete_project,
    get_project_agent_settings,
    get_project_memory_batch,
    get_project_memory_status,
    list_projects,
    load_project_memory_context,
    project_exists,
    rename_project,
    save_project_memory_summary,
    set_project_memory_error,
    update_project_agent_settings,
)

__all__ = [
    # schema
    "DEFAULT_PROJECT_ID",
    "DEFAULT_PROJECT_NAME",
    "DEFAULT_VERIFIER_CONTEXT_TOKENS",
    "DEFAULT_WRITER_CONTEXT_TOKENS",
    "FTS_ERROR_META_KEY",
    "FTS_INDEX_VERSION",
    "FTS_VERSION_META_KEY",
    "GRAPH_INDEX_VERSION",
    "GRAPH_VERSION_META_KEY",
    "MAX_AGENT_CONTEXT_TOKENS",
    "MIN_AGENT_CONTEXT_TOKENS",
    "get_connection",
    "get_meta",
    "init_db",
    "transaction",
    "utc_now",
    # fts
    "get_fts5_status",
    "search_chunks_fts",
    # links
    "get_document_graph_status",
    # documents
    "commit_staged_index",
    "delete_documents_not_in",
    "discard_staged_index",
    "get_document_hashes",
    "get_document_statuses",
    "get_embedding_dimensions",
    "get_stats",
    "replace_document",
    "reset_staged_index",
    "resolve_staged_document_links",
    "stage_document",
    # conversations
    "conversation_exists",
    "conversation_title",
    "delete_conversation",
    "get_conversation",
    "get_conversation_memory_batch",
    "get_conversation_memory_status",
    "list_conversations",
    "load_conversation_memory_context",
    "save_conversation_memory_summary",
    "save_conversation_turn",
    "set_conversation_memory_enabled",
    "set_conversation_memory_error",
    # projects
    "create_project",
    "delete_project",
    "get_project_agent_settings",
    "get_project_memory_batch",
    "get_project_memory_status",
    "list_projects",
    "load_project_memory_context",
    "project_exists",
    "rename_project",
    "save_project_memory_summary",
    "set_project_memory_error",
    "update_project_agent_settings",
]

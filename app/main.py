from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from threading import Thread
from typing import Any, Iterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import display_configured_path, settings
from app.http_security import reject_unsafe_request
from app.db import (
    DEFAULT_PROJECT_ID,
    conversation_exists,
    create_project,
    delete_conversation,
    delete_project,
    get_conversation,
    get_conversation_memory_status,
    get_document_statuses,
    get_project_agent_settings,
    get_project_memory_status,
    init_db,
    list_conversations,
    list_projects,
    project_exists,
    rename_project,
    save_conversation_turn,
    set_conversation_memory_enabled,
    set_conversation_memory_error,
    set_project_memory_error,
    update_project_agent_settings,
)
from app.model_profiles import (
    active_chat_model_profile,
    configure_chat_model_profile,
    list_chat_model_profiles,
    resolve_model_context,
)
from app.rag.agents import (
    run_rag_agent_pipeline,
    stream_rag_agent_pipeline,
)
from app.rag.indexer import (
    get_index_status,
    index_progress,
    index_vault,
    start_background_indexing,
)
from app.rag.memory import (
    build_conversation_context_overview,
    build_memory_aware_retrieval_query,
    get_memory_context,
    get_project_memory_context,
    maybe_summarize_conversation,
    maybe_summarize_project,
)
from app.rag.citations import validate_answer
from app.rag.ollama_client import get_ollama_status, warmup_models
from app.rag.retrieval import get_retrieval_status, retrieve
from app.rag.vector_store import vector_store
from app.schemas import (
    ChatModelProfileRequest,
    ConversationMemoryRequest,
    IndexResponse,
    ProjectAgentSettingsRequest,
    ProjectNameRequest,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from app.vault_config import choose_and_configure_vault


logger = logging.getLogger(__name__)
PUBLIC_OPERATION_ERROR = "No se pudo completar la operación."
PUBLIC_QUERY_ERROR = "No se pudo procesar la consulta."
PUBLIC_MEMORY_ERROR = "No se pudo actualizar la memoria."


def _safe_http_error(
    action: str,
    exc: Exception,
    *,
    status_code: int = 400,
    detail: str = PUBLIC_OPERATION_ERROR,
) -> HTTPException:
    logger.exception("request_operation_failed", extra={"action": action})
    return HTTPException(status_code=status_code, detail=detail)


def _public_payload(value: Any) -> Any:
    """Elimina detalles internos de errores antes de serializarlos al cliente."""
    if isinstance(value, dict):
        payload: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"error", "last_error", "embedding_error", "chat_error"}:
                payload[key] = PUBLIC_OPERATION_ERROR if item else item
            elif key == "errors" and isinstance(item, list):
                payload[key] = [PUBLIC_OPERATION_ERROR for _ in item]
            else:
                payload[key] = _public_payload(item)
        return payload
    if isinstance(value, list):
        return [_public_payload(item) for item in value]
    return value


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    # Precarga la matriz de embeddings y los modelos en un hilo aparte:
    # la primera consulta deja de pagar el arranque en frío.
    Thread(target=_warm_start, name="rag-warmup", daemon=True).start()
    yield


def _warm_start() -> None:
    try:
        vector_store.ensure_loaded()
    except Exception:
        pass
    try:
        warmup_models()
    except Exception:
        pass


app = FastAPI(
    title="RAG local para Obsidian",
    version="1.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def local_request_security(
    request: Request,
    call_next: Any,
) -> Any:
    rejected = reject_unsafe_request(request)
    if rejected is not None:
        return rejected
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


app.mount(
    "/static",
    StaticFiles(directory=settings.base_dir / "app" / "static"),
    name="static",
)
templates = Jinja2Templates(
    directory=settings.base_dir / "app" / "templates"
)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    context_info = resolve_model_context(settings.chat_model)
    return _public_payload({
        "vault_path": str(settings.vault_path),
        "vault_display_path": display_configured_path(settings.vault_path),
        "vault_exists": settings.vault_path.exists(),
        "database_path": str(settings.database_path),
        "capabilities": {
            "conversation_history": True,
            "projects": True,
            "conversation_memory": True,
            "hybrid_search": True,
            "graph_search": True,
            "project_agents": True,
            "project_context_usage": True,
            "project_memory": True,
            "streaming": settings.stream_enabled,
            "background_indexing": True,
            "temporal_validity": True,
            "tag_filters": True,
            "citation_validation": True,
            "context_inspector": True,
        },
        "memory": {
            "summary_interval": settings.memory_summary_interval,
            "recent_turns": settings.memory_recent_turns,
            "max_context_chars": settings.memory_max_context_chars,
        },
        "chat_profiles": list_chat_model_profiles(),
        "active_chat_profile": active_chat_model_profile(),
        "model_context": {
            "model": context_info.model,
            "profile_id": context_info.profile_id,
            "model_context_window": context_info.model_context_window,
            "source": context_info.source,
            "verified": context_info.verified,
            "warning": context_info.warning,
        },
        "ollama": get_ollama_status(),
        "index": get_index_status(),
        "retrieval": get_retrieval_status(),
        "document_statuses": get_document_statuses(),
    })


@app.post("/api/config/chat-profile")
def select_chat_profile(payload: ChatModelProfileRequest) -> dict:
    try:
        profile = configure_chat_model_profile(payload.profile)
    except Exception as exc:
        raise _safe_http_error("cambiar el perfil de chat", exc) from exc
    return {
        "profile": profile,
        "chat_model": profile["model"],
        "reindex_required": False,
    }


@app.post("/api/vault/select")
def select_vault() -> dict:
    try:
        selected_path = choose_and_configure_vault()
    except Exception as exc:
        raise _safe_http_error("seleccionar el vault", exc) from exc
    effective_path = selected_path or settings.vault_path
    return {
        "selected": selected_path is not None,
        "vault_path": str(effective_path),
        "vault_display_path": display_configured_path(effective_path),
        "vault_exists": effective_path.exists(),
    }


@app.post("/api/index", response_model=IndexResponse)
def run_indexing() -> IndexResponse:
    try:
        result = index_vault()
        return IndexResponse(**_public_payload(result))
    except Exception as exc:
        raise _safe_http_error("indexar el vault", exc) from exc


@app.post("/api/index/start")
def start_indexing() -> dict:
    """Lanza la indexación sin bloquear la petición HTTP."""
    try:
        return start_background_indexing()
    except Exception as exc:
        raise _safe_http_error("iniciar la indexación", exc) from exc


@app.get("/api/index/progress")
def indexing_progress() -> dict:
    return _public_payload(index_progress.snapshot())


@app.get("/api/conversations")
def conversation_list(project_id: str | None = None) -> dict:
    return _public_payload({"conversations": list_conversations(project_id)})


@app.get("/api/projects")
def project_list() -> dict:
    return _public_payload({
        "projects": list_projects(),
        "default_project_id": DEFAULT_PROJECT_ID,
    })


@app.post("/api/projects", status_code=201)
def add_project(payload: ProjectNameRequest) -> dict:
    try:
        return create_project(payload.name)
    except ValueError as exc:
        raise _safe_http_error(
            "crear el proyecto",
            exc,
            status_code=409,
        ) from exc


@app.patch("/api/projects/{project_id}")
def update_project(
    project_id: str,
    payload: ProjectNameRequest,
) -> dict:
    try:
        project = rename_project(project_id, payload.name)
    except ValueError as exc:
        raise _safe_http_error(
            "renombrar el proyecto",
            exc,
            status_code=409,
        ) from exc
    if project is None:
        raise HTTPException(
            status_code=404,
            detail="No existe el proyecto solicitado.",
        )
    return project


@app.delete("/api/projects/{project_id}")
def remove_project(project_id: str) -> dict:
    try:
        result = delete_project(project_id)
    except ValueError as exc:
        raise _safe_http_error("eliminar el proyecto", exc) from exc
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No existe el proyecto solicitado.",
        )
    return result


@app.patch("/api/projects/{project_id}/agent-settings")
def update_agent_settings(
    project_id: str,
    payload: ProjectAgentSettingsRequest,
) -> dict:
    try:
        agent_settings = update_project_agent_settings(
            project_id,
            verification_enabled=payload.verification_enabled,
            project_memory_enabled=payload.project_memory_enabled,
            verifier_context_tokens=payload.verifier_context_tokens,
            writer_context_tokens=payload.writer_context_tokens,
        )
    except ValueError as exc:
        raise _safe_http_error(
            "actualizar los agentes del proyecto",
            exc,
        ) from exc
    if agent_settings is None:
        raise HTTPException(
            status_code=404,
            detail="No existe el proyecto solicitado.",
        )
    return _public_payload({
        "project_id": project_id,
        "agent_settings": agent_settings,
        "memory": get_project_memory_status(project_id) or {},
    })


@app.get("/api/conversations/{conversation_id}")
def conversation_detail(conversation_id: str) -> dict:
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="No existe la conversación solicitada.",
        )
    return _public_payload(conversation)


@app.get("/api/conversations/{conversation_id}/context")
def conversation_context(conversation_id: str) -> dict:
    overview = build_conversation_context_overview(conversation_id)
    if overview is None:
        raise HTTPException(
            status_code=404,
            detail="No existe la conversación solicitada.",
        )
    return _public_payload(overview)


@app.delete("/api/conversations/{conversation_id}")
def remove_conversation(conversation_id: str) -> dict:
    if not delete_conversation(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="No existe la conversación solicitada.",
        )
    return {"deleted": True, "conversation_id": conversation_id}


@app.patch("/api/conversations/{conversation_id}/memory")
def update_conversation_memory(
    conversation_id: str,
    payload: ConversationMemoryRequest,
) -> dict:
    memory = set_conversation_memory_enabled(
        conversation_id,
        payload.enabled,
    )
    if memory is None:
        raise HTTPException(
            status_code=404,
            detail="No existe la conversación solicitada.",
        )
    return _public_payload(
        {"conversation_id": conversation_id, "memory": memory}
    )


# ----------------------------------------------------------------------
# Consulta: preparación común a la vía síncrona y a la de streaming
# ----------------------------------------------------------------------
def _resolve_project(payload: QueryRequest) -> str:
    response_project_id = payload.project_id or DEFAULT_PROJECT_ID
    if (
        payload.conversation_id
        and not conversation_exists(payload.conversation_id)
    ):
        raise HTTPException(
            status_code=404,
            detail="La conversación ya no existe.",
        )
    if payload.conversation_id:
        existing = get_conversation(payload.conversation_id)
        if existing is not None:
            response_project_id = str(
                existing.get("project_id", DEFAULT_PROJECT_ID)
            )
    if payload.project_id and not project_exists(payload.project_id):
        raise HTTPException(
            status_code=404,
            detail="El proyecto ya no existe.",
        )
    return response_project_id


def _default_agent_settings() -> dict[str, Any]:
    return {
        "verification_enabled": True,
        "project_memory_enabled": True,
        "verifier_context_tokens": settings.default_verifier_context_tokens,
        "writer_context_tokens": settings.default_writer_context_tokens,
    }


def _prepare_query(payload: QueryRequest) -> dict[str, Any]:
    """Memoria + recuperación. No invoca todavía al redactor."""
    response_project_id = _resolve_project(payload)

    memory_context = ""
    if payload.use_memory and payload.conversation_id:
        memory_context = get_memory_context(payload.conversation_id)

    agent_settings = (
        get_project_agent_settings(response_project_id)
        or _default_agent_settings()
    )

    project_memory_context = ""
    if agent_settings["project_memory_enabled"]:
        project_memory_context = get_project_memory_context(
            response_project_id
        )

    retrieval_memory = "\n\n".join(
        value
        for value in (memory_context, project_memory_context)
        if value.strip()
    )
    retrieval_question = build_memory_aware_retrieval_query(
        payload.question,
        retrieval_memory,
    )

    hits = retrieve(
        payload.question,
        top_k=payload.top_k,
        min_similarity=settings.min_similarity,
        status=payload.status,
        expand_links=payload.expand_links,
        lexical_question=payload.question,
        contextual_question=(
            retrieval_question
            if retrieval_question.strip() != payload.question.strip()
            else None
        ),
        vigencia=payload.vigencia,
        tags=payload.tags,
    )
    return {
        "project_id": response_project_id,
        "memory_context": memory_context,
        "project_memory_context": project_memory_context,
        "agent_settings": agent_settings,
        "hits": hits,
    }


def _refine_question_with_feedback(
    original_question: str,
    feedback: str,
) -> str:
    """Refina la pregunta usando el feedback del usuario.

    Añade el feedback al contexto de la pregunta para guiar la siguiente
    iteración del modelo. El cliente (frontend) decidirá cuándo lanzar
    una nueva iteración basándose en la respuesta que se le devuelva.
    """
    return f"""{original_question}

--- Feedback del usuario ---
{feedback}

--- Instrucción para la próxima iteración ---
Por favor, proporcione una respuesta que aborde las preocupaciones arriba
mencionadas, amplíe la información donde falte y mantenga todas las
referencias documentales. Si la evidencia documental no cubre algún punto
del feedback, indíquelo claramente."""


def _build_sources(hits: list[dict[str, Any]]) -> list[SourceItem]:
    return [
        SourceItem(
            reference=f"[{index}]",
            title=hit["title"],
            path=hit["path"],
            heading=hit["heading"],
            score=round(float(hit["score"]), 4),
            semantic_score=(
                round(float(hit["semantic_score"]), 4)
                if hit.get("semantic_score") is not None
                else None
            ),
            lexical_score=(
                round(float(hit["lexical_score"]), 4)
                if hit.get("lexical_score") is not None
                else None
            ),
            fusion_score=(
                round(float(hit["fusion_score"]), 4)
                if hit.get("fusion_score") is not None
                else None
            ),
            reason=hit["reason"],
            metadata=hit.get("metadata") or {},
            content=hit["content"],
            matched_content=hit.get("matched_content"),
            context_expanded=bool(hit.get("context_expanded")),
            matched_chunk_ids=[
                int(value) for value in hit.get("matched_chunk_ids") or []
            ],
            tags=hit.get("tags") or [],
            vigencia=hit.get("vigencia") or {},
        )
        for index, hit in enumerate(hits, start=1)
    ]


def _agent_abstained(metrics: dict[str, Any]) -> bool:
    """Indica que el verificador cerró el turno sin invocar al redactor."""
    writer = metrics.get("writer") or {}
    return writer.get("status") == "skipped_insufficient_evidence"


def _persist_turn(
    payload: QueryRequest,
    context: dict[str, Any],
    answer: str,
    sources: list[SourceItem],
    agent_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Guarda el turno y resume la memoria; nunca rompe la respuesta."""
    hits = context["hits"]
    response_project_id = context["project_id"]
    agent_settings = context["agent_settings"]

    conversation_id, turn_id = save_conversation_turn(
        conversation_id=payload.conversation_id,
        project_id=payload.project_id,
        question=payload.question,
        answer=answer,
        sources=[source.model_dump() for source in sources],
        chat_model=settings.chat_model,
        memory_enabled=payload.use_memory,
        agent_metrics=agent_metrics,
    )

    summary_updated = False
    memory_warning = None
    if hits and payload.use_memory:
        try:
            summary_updated = maybe_summarize_conversation(conversation_id)
        except Exception as exc:
            logger.exception(
                "conversation_memory_update_failed",
                extra={"conversation_id": conversation_id},
            )
            memory_warning = (
                "La respuesta se guardó, pero no se pudo actualizar "
                "el resumen de memoria."
            )
            try:
                set_conversation_memory_error(
                    conversation_id,
                    PUBLIC_MEMORY_ERROR,
                )
            except Exception:
                pass

    project_summary_updated = False
    project_memory_warning = None
    if hits and agent_settings["project_memory_enabled"]:
        try:
            project_summary_updated = maybe_summarize_project(
                response_project_id
            )
        except Exception as exc:
            logger.exception(
                "project_memory_update_failed",
                extra={"project_id": response_project_id},
            )
            project_memory_warning = (
                "La respuesta se guardó, pero no se pudo actualizar "
                "la memoria compartida del proyecto."
            )
            try:
                set_project_memory_error(
                    response_project_id,
                    PUBLIC_MEMORY_ERROR,
                )
            except Exception:
                pass

    try:
        memory_status = get_conversation_memory_status(conversation_id) or {}
    except Exception:
        memory_status = {}
    memory_status.update(
        {
            "enabled": payload.use_memory,
            "used_context": bool(context["memory_context"]),
            "summary_updated": summary_updated,
            "warning": memory_warning,
        }
    )

    try:
        project_memory_status = (
            get_project_memory_status(response_project_id) or {}
        )
    except Exception:
        project_memory_status = {}
    project_memory_status.update(
        {
            "enabled": bool(agent_settings["project_memory_enabled"]),
            "used_context": bool(context["project_memory_context"]),
            "summary_updated": project_summary_updated,
            "warning": project_memory_warning,
        }
    )

    return {
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "memory": _public_payload(memory_status),
        "project_memory": _public_payload(project_memory_status),
    }


@app.post("/api/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
    try:
        context = _prepare_query(payload)
        agent_settings = context["agent_settings"]
        agent_result = run_rag_agent_pipeline(
            payload.question,
            context["hits"],
            conversation_memory=context["memory_context"],
            project_memory=context["project_memory_context"],
            verification_enabled=bool(
                agent_settings["verification_enabled"]
            ),
            verifier_context_tokens=int(
                agent_settings["verifier_context_tokens"]
            ),
            writer_context_tokens=int(
                agent_settings["writer_context_tokens"]
            ),
        )
        sources = _build_sources(context["hits"])
        answer, citation_report, citation_warning = validate_answer(
            agent_result.answer,
            len(sources),
            abstained=_agent_abstained(agent_result.metrics),
        )

        # --- LÓGICA DE ITERACIÓN ---
        # Si hay feedback del usuario o se solicita una iteración,
        # usamos el feedback para refinar la pregunta y volvemos a ejecutar.
        # Si iteration=0 y no hay feedback, es la primera pasada: guardamos y devolvemos.
        max_iterations = 5
        should_iterate = payload.iteration > 0 or bool(payload.iteration_feedback.strip())

        if should_iterate and payload.iteration < max_iterations:
            # Refina la pregunta usando el feedback del usuario
            refined_question = _refine_question_with_feedback(
                payload.question,
                payload.iteration_feedback,
            )
            # Crea un payload con la pregunta refinada para la persistency
            refined_payload = QueryRequest(
                question=refined_question,
                top_k=payload.top_k,
                status=payload.status,
                expand_links=payload.expand_links,
                vigencia=payload.vigencia,
                tags=payload.tags,
                iteration=payload.iteration + 1,
                iteration_feedback="",
            )
            # Ejecuta una nueva iteración con la pregunta refinada
            context2 = _prepare_query(refined_payload)
            agent_settings2 = context2["agent_settings"]
            agent_result2 = run_rag_agent_pipeline(
                refined_question,
                context2["hits"],
                conversation_memory=context2["memory_context"],
                project_memory=context2["project_memory_context"],
                verification_enabled=bool(
                    agent_settings2["verification_enabled"]
                ),
                verifier_context_tokens=int(
                    agent_settings2["verifier_context_tokens"]
                ),
                writer_context_tokens=int(
                    agent_settings2["writer_context_tokens"]
                ),
            )
            sources2 = _build_sources(context2["hits"])
            answer2, citation_report2, citation_warning2 = validate_answer(
                agent_result2.answer,
                len(sources2),
                abstained=_agent_abstained(agent_result2.metrics),
            )
            saved2 = _persist_turn(
                refined_payload,
                context2,
                answer2,
                sources2,
                agent_result2.metrics,
            )
            agent_metrics2 = _public_payload(dict(agent_result2.metrics))
            agent_metrics2["warning"] = " ".join(
                part
                for part in (agent_result2.warning, citation_warning2)
                if part
            ) or None
            agent_metrics2["iteration"] = refined_payload.iteration
            agent_metrics2["iteration_available"] = (
                refined_payload.iteration < max_iterations
            )
            agent_metrics2["request_feedback"] = False
            return QueryResponse(
                answer=answer2,
                sources=sources2,
                conversation_id=saved2["conversation_id"],
                project_id=context2["project_id"],
                turn_id=saved2["turn_id"],
                chat_model=settings.chat_model,
                memory=saved2["memory"],
                project_memory=saved2["project_memory"],
                agents=agent_metrics2,
                citations=citation_report2.as_dict(),
            )
        else:
            # Primera pasada o se alcanzó el máximo de iteraciones: guardar y devolver
            saved = _persist_turn(
                payload,
                context,
                answer,
                sources,
                agent_result.metrics,
            )
            agent_metrics = _public_payload(dict(agent_result.metrics))
            agent_metrics["warning"] = " ".join(
                part
                for part in (agent_result.warning, citation_warning)
                if part
            ) or None
            # Añadimos metadatos de iteración para la UI
            agent_metrics["iteration"] = payload.iteration
            agent_metrics["iteration_available"] = (
                payload.iteration < max_iterations and not should_iterate
            )
            agent_metrics["request_feedback"] = (
                not should_iterate and payload.iteration == 0
            )
            return QueryResponse(
                answer=answer,
                sources=sources,
                conversation_id=saved["conversation_id"],
                project_id=context["project_id"],
                turn_id=saved["turn_id"],
                chat_model=settings.chat_model,
                memory=saved["memory"],
                project_memory=saved["project_memory"],
                agents=agent_metrics,
                citations=citation_report.as_dict(),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise _safe_http_error(
            "procesar la consulta",
            exc,
            detail=PUBLIC_QUERY_ERROR,
        ) from exc


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _query_event_stream(payload: QueryRequest) -> Iterator[str]:
    """Genera la respuesta como eventos SSE.

    Las fuentes se envían en cuanto termina la recuperación, mucho antes
    de que el modelo escriba la primera palabra. El usuario ve evidencia
    útil de inmediato en lugar de un indicador de carga.
    """
    try:
        context = _prepare_query(payload)
    except HTTPException as exc:
        yield _sse("error", {"detail": exc.detail})
        return
    except Exception as exc:
        logger.exception("stream_query_prepare_failed")
        yield _sse("error", {"detail": PUBLIC_QUERY_ERROR})
        return

    sources = _build_sources(context["hits"])
    yield _sse(
        "retrieval",
        {
            "sources": [source.model_dump() for source in sources],
            "count": len(sources),
            "project_id": context["project_id"],
        },
    )

    agent_settings = context["agent_settings"]
    answer = ""
    metrics: dict[str, Any] = {}
    warning: str | None = None

    try:
        for event in stream_rag_agent_pipeline(
            payload.question,
            context["hits"],
            conversation_memory=context["memory_context"],
            project_memory=context["project_memory_context"],
            verification_enabled=bool(
                agent_settings["verification_enabled"]
            ),
            verifier_context_tokens=int(
                agent_settings["verifier_context_tokens"]
            ),
            writer_context_tokens=int(
                agent_settings["writer_context_tokens"]
            ),
        ):
            kind = event["type"]
            if kind == "stage":
                yield _sse("stage", {"stage": event["stage"]})
            elif kind == "delta":
                yield _sse("delta", {"text": event["text"]})
            elif kind == "final":
                answer = event["answer"]
                metrics = event["metrics"]
                warning = event["warning"]
    except Exception as exc:
        logger.exception("stream_query_generation_failed")
        yield _sse("error", {"detail": PUBLIC_QUERY_ERROR})
        return

    answer, citation_report, citation_warning = validate_answer(
        answer,
        len(sources),
        abstained=_agent_abstained(metrics),
    )
    if citation_warning:
        warning = " ".join(
            part for part in (warning, citation_warning) if part
        )

    try:
        saved = _persist_turn(payload, context, answer, sources, metrics)
    except Exception as exc:
        logger.exception("stream_query_persist_failed")
        yield _sse(
            "error",
            {"detail": "No se pudo guardar la respuesta."},
        )
        return

    agent_metrics = _public_payload(dict(metrics))
    agent_metrics["warning"] = warning
    yield _sse(
        "done",
        {
            "answer": answer,
            "conversation_id": saved["conversation_id"],
            "project_id": context["project_id"],
            "turn_id": saved["turn_id"],
            "chat_model": settings.chat_model,
            "memory": saved["memory"],
            "project_memory": saved["project_memory"],
            "agents": agent_metrics,
            "citations": citation_report.as_dict(),
        },
    )


@app.post("/api/query/stream")
def query_documents_stream(payload: QueryRequest) -> StreamingResponse:
    if not settings.stream_enabled:
        raise HTTPException(
            status_code=409,
            detail="El streaming está desactivado (RAG_STREAMING=false).",
        )
    return StreamingResponse(
        _query_event_stream(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

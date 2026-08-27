from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock, Thread
import time
from typing import Any
from uuid import uuid4

from app.config import settings
from app.db import (
    commit_staged_index,
    delete_documents_not_in,
    discard_staged_index,
    get_document_hashes,
    get_embedding_dimensions,
    get_meta,
    get_stats,
    init_db,
    replace_document,
    reset_staged_index,
    resolve_staged_document_links,
    stage_document,
)
from app.rag.embedding_tasks import resolve_scheme
from app.rag.graph import graph_store
from app.rag.markdown import calculate_sha256, parse_markdown_file
from app.rag.ollama_client import create_embeddings, get_embedding_dimension
from app.rag.vector_store import vector_store


INDEX_FORMAT_VERSION = 4
# El troceado estructurado cambia las fronteras de los fragmentos: al
# variar esta versión el índice se reconstruye de forma atómica.
CHUNKER_VERSION = "structured-2"
INDEX_FINGERPRINT_KEY = "index_fingerprint"
EMBEDDING_PROVIDER = "ollama"
EXCLUDED_VAULT_DIRECTORIES = {".obsidian", "_plantillas", "_templates"}
_indexing_lock = Lock()
logger = logging.getLogger(__name__)
PUBLIC_INDEX_ERROR = "No se pudo indexar este documento."
PUBLIC_INDEX_RUN_ERROR = "No se pudo completar la indexación."


# ----------------------------------------------------------------------
# Progreso de indexación
# ----------------------------------------------------------------------
class IndexProgress:
    """Estado observable de la indexación en curso.

    Permite lanzar la indexación en segundo plano y que la interfaz
    consulte el avance, en lugar de dejar una petición HTTP bloqueada
    durante minutos sin ninguna señal.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._state: dict[str, Any] = {
            "running": False,
            "phase": "idle",
            "processed": 0,
            "total": 0,
            "current_file": "",
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    def start(self, total: int) -> None:
        with self._lock:
            self._state.update(
                {
                    "running": True,
                    "phase": "scanning",
                    "processed": 0,
                    "total": total,
                    "current_file": "",
                    "started_at": time.time(),
                    "finished_at": None,
                    "result": None,
                    "error": None,
                }
            )

    def advance(self, current_file: str, phase: str = "embedding") -> None:
        with self._lock:
            self._state["processed"] += 1
            self._state["current_file"] = current_file
            self._state["phase"] = phase

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._state["phase"] = phase

    def finish(
        self,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._state.update(
                {
                    "running": False,
                    "phase": "error" if error else "done",
                    "finished_at": time.time(),
                    "result": result,
                    "error": error,
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        started = state.get("started_at")
        finished = state.get("finished_at")
        if started:
            end = finished or time.time()
            state["elapsed_seconds"] = round(end - started, 1)
        else:
            state["elapsed_seconds"] = 0.0
        total = state["total"] or 0
        state["percent"] = (
            round(state["processed"] / total * 100, 1) if total else 0.0
        )
        return state


index_progress = IndexProgress()



def build_index_configuration(
    *,
    vault_path: Path,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimension: int,
    chunk_size: int,
    chunk_overlap: int,
    embedding_prefix_scheme: str,
    chunker_version: str,
    parent_child_chunking_enabled: bool = False,
    parent_chunk_size: int = 6000,
    child_chunk_size: int = 700,
    child_chunk_overlap: int = 100,
) -> dict[str, Any]:
    configuration = {
        "format_version": INDEX_FORMAT_VERSION,
        "vault_path": str(vault_path.resolve()),
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        # El prefijo de tarea forma parte del espacio vectorial: si
        # cambia, el índice deja de ser comparable y se reconstruye.
        "embedding_prefix_scheme": embedding_prefix_scheme,
        "chunker_version": chunker_version,
    }
    if parent_child_chunking_enabled:
        configuration.update(
            {
                "chunking_mode": "parent_child-1",
                "parent_chunk_size": parent_chunk_size,
                "child_chunk_size": child_chunk_size,
                "child_chunk_overlap": child_chunk_overlap,
            }
        )
    return configuration


def serialize_index_configuration(configuration: dict[str, Any]) -> str:
    return json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def changed_configuration_fields(
    stored_value: str | None,
    current: dict[str, Any],
) -> list[str]:
    if not stored_value:
        return ["configuración anterior no registrada"]
    try:
        stored = json.loads(stored_value)
    except (json.JSONDecodeError, TypeError):
        return ["configuración anterior no válida"]
    if not isinstance(stored, dict):
        return ["configuración anterior no válida"]
    changes = [
        key
        for key, value in current.items()
        if stored.get(key) != value
    ]
    if "chunking_mode" in stored and "chunking_mode" not in current:
        changes.append("chunking_mode")
    return changes


def discover_markdown_files(vault_path: Path) -> list[Path]:
    if not vault_path.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de Obsidian: {vault_path}"
        )
    if not vault_path.is_dir():
        raise NotADirectoryError(
            f"La ruta de Obsidian no es una carpeta: {vault_path}"
        )

    output: list[Path] = []
    for path in vault_path.rglob("*.md"):
        relative_parts = path.relative_to(vault_path).parts
        directory_names = {
            part.casefold() for part in relative_parts[:-1]
        }
        if directory_names.intersection(EXCLUDED_VAULT_DIRECTORIES):
            continue
        if any(part.startswith(".") for part in relative_parts):
            continue
        output.append(path)

    return sorted(output, key=lambda item: str(item).casefold())


def get_index_status() -> dict[str, Any]:
    stats = get_stats()
    stored_value = get_meta(INDEX_FINGERPRINT_KEY)
    dimensions = get_embedding_dimensions()
    reasons: list[str] = []
    stored: dict[str, Any] | None = None

    if stored_value:
        try:
            parsed = json.loads(stored_value)
            stored = parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            stored = None

    if stats["chunks"] > 0:
        if stored is None:
            reasons.append("configuración anterior no registrada o no válida")
        else:
            expected_without_live_dimension = {
                "format_version": INDEX_FORMAT_VERSION,
                "vault_path": str(settings.vault_path.resolve()),
                "embedding_provider": EMBEDDING_PROVIDER,
                "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_prefix_scheme": resolve_scheme(
                    settings.embedding_model,
                    settings.embedding_prefix_scheme,
                ).id,
                "chunker_version": CHUNKER_VERSION,
            }
            if settings.parent_child_chunking_enabled:
                expected_without_live_dimension.update(
                    {
                        "chunking_mode": "parent_child-1",
                        "parent_chunk_size": settings.parent_chunk_size,
                        "child_chunk_size": settings.child_chunk_size,
                        "child_chunk_overlap": settings.child_chunk_overlap,
                    }
                )
            reasons.extend(
                key
                for key, value in expected_without_live_dimension.items()
                if stored.get(key) != value
            )
            if (
                "chunking_mode" in stored
                and "chunking_mode" not in expected_without_live_dimension
            ):
                reasons.append("chunking_mode")
            stored_dimension = stored.get("embedding_dimension")
            if (
                len(dimensions) != 1
                or not isinstance(stored_dimension, int)
                or dimensions[0] != stored_dimension
            ):
                reasons.append("embedding_dimension")

    reasons = list(dict.fromkeys(reasons))
    if stats["chunks"] == 0:
        state = "vacío"
    elif reasons:
        state = "requiere_reconstrucción"
    else:
        state = "listo"

    return {
        "state": state,
        "needs_rebuild": bool(reasons),
        "rebuild_reasons": reasons,
        "embedding_dimensions": dimensions,
        "fingerprint_registered": stored_value is not None,
        "chunking": {
            "mode": (
                "parent_child"
                if settings.parent_child_chunking_enabled
                else "flat"
            ),
            "parent_chunk_size": (
                settings.parent_chunk_size
                if settings.parent_child_chunking_enabled
                else None
            ),
            "child_chunk_size": (
                settings.child_chunk_size
                if settings.parent_child_chunking_enabled
                else settings.chunk_size
            ),
            "child_chunk_overlap": (
                settings.child_chunk_overlap
                if settings.parent_child_chunking_enabled
                else settings.chunk_overlap
            ),
        },
        "stats": stats,
    }


def prepare_document(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    document = parse_markdown_file(
        path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        parent_child_enabled=settings.parent_child_chunking_enabled,
        parent_chunk_size=settings.parent_chunk_size,
        child_chunk_size=settings.child_chunk_size,
        child_chunk_overlap=settings.child_chunk_overlap,
    )
    scheme = resolve_scheme(
        settings.embedding_model,
        settings.embedding_prefix_scheme,
    )
    # El suplemento lleva etiquetas y avisos al espacio vectorial sin
    # ensuciar el contenido que se muestra y se cita.
    supplement = (
        f"\n\n{document.supplement}" if document.supplement else ""
    )
    embedding_inputs = [
        (
            f"{scheme.document}"
            f"Título: {document.title}\n"
            f"Sección: {chunk.heading}\n\n"
            f"{chunk.content}"
            f"{supplement}"
        )
        for chunk in document.chunks
    ]
    vectors = create_embeddings(embedding_inputs)
    prepared_chunks = [
        {
            "chunk_index": chunk.index,
            "heading": chunk.heading,
            "content": chunk.content,
            "parent_index": chunk.parent_index,
            "embedding": vector,
        }
        for chunk, vector in zip(document.chunks, vectors, strict=True)
    ]
    return document, prepared_chunks


def _atomic_rebuild(
    *,
    files: list[Path],
    serialized_configuration: str,
    configuration_changes: list[str],
    stored_configuration: str | None,
    existing_stats: dict[str, int],
    known_hashes: dict[str, str],
) -> dict:
    run_id = uuid4().hex
    reset_staged_index()

    indexed = 0
    chunks_created = 0
    parents_created = 0
    links_created = 0
    errors: list[str] = []
    relative_paths = {
        path.relative_to(settings.vault_path).as_posix() for path in files
    }

    for path in files:
        relative_path = path.relative_to(settings.vault_path).as_posix()
        try:
            index_progress.advance(relative_path)
            document, prepared_chunks = prepare_document(path)
            stage_document(
                run_id=run_id,
                path=relative_path,
                title=document.title,
                sha256=document.sha256,
                mtime=document.mtime,
                metadata=document.metadata,
                links=document.links,
                chunks=prepared_chunks,
                parents=[
                    {
                        "parent_index": parent.index,
                        "heading": parent.heading,
                        "content": parent.content,
                    }
                    for parent in document.parents
                ],
            )
            indexed += 1
            chunks_created += len(prepared_chunks)
            parents_created += len(document.parents)
        except Exception as exc:
            logger.exception(
                "index_document_failed",
                extra={"path": relative_path, "mode": "rebuild"},
            )
            errors.append(f"{relative_path}: {PUBLIC_INDEX_ERROR}")

    if not errors:
        try:
            index_progress.set_phase("resolving_links")
            links_created = resolve_staged_document_links(run_id)
        except Exception:
            logger.exception("index_graph_staging_failed")
            errors.append(
                "No se pudo preparar el grafo de enlaces del índice."
            )

    rebuilt = bool(
        stored_configuration is not None or existing_stats["documents"] > 0
    )
    if errors:
        try:
            discard_staged_index(run_id)
        except Exception as exc:
            logger.exception("index_rebuild_cleanup_failed")
            errors.append("No se pudo limpiar la preparación fallida.")
        errors.append(
            "Reconstrucción atómica cancelada: el índice anterior "
            "se ha conservado sin cambios."
        )
        return {
            "scanned": len(files),
            "indexed": 0,
            "unchanged": 0,
            "deleted": 0,
            "chunks_created": 0,
            "parents_created": 0,
            "errors": errors,
            "rebuilt": False,
            "rebuild_reasons": configuration_changes,
            "stats": get_stats(),
        }

    try:
        commit_staged_index(
            run_id=run_id,
            fingerprint_key=INDEX_FINGERPRINT_KEY,
            fingerprint_value=serialized_configuration,
            expected_documents=indexed,
            expected_chunks=chunks_created,
            expected_links=links_created,
            expected_parents=parents_created,
        )
    except Exception as exc:
        logger.exception("index_rebuild_commit_failed")
        try:
            discard_staged_index(run_id)
        except Exception as cleanup_exc:
            logger.exception("index_rebuild_cleanup_failed")
            errors.append("No se pudo limpiar la preparación fallida.")
        errors.append(
            "No se pudo activar el índice preparado; el índice anterior "
            "se ha conservado sin cambios."
        )
        return {
            "scanned": len(files),
            "indexed": 0,
            "unchanged": 0,
            "deleted": 0,
            "chunks_created": 0,
            "parents_created": 0,
            "errors": errors,
            "rebuilt": False,
            "rebuild_reasons": configuration_changes,
            "stats": get_stats(),
        }

    return {
        "scanned": len(files),
        "indexed": indexed,
        "unchanged": 0,
        "deleted": len(set(known_hashes).difference(relative_paths)),
        "chunks_created": chunks_created,
        "parents_created": parents_created,
        "errors": [],
        "rebuilt": rebuilt,
        "rebuild_reasons": configuration_changes if rebuilt else [],
        "stats": get_stats(),
    }


def _update_incrementally(
    *,
    files: list[Path],
    known_hashes: dict[str, str],
) -> dict:
    relative_paths = {
        path.relative_to(settings.vault_path).as_posix() for path in files
    }
    deleted = delete_documents_not_in(relative_paths)
    indexed = 0
    unchanged = 0
    chunks_created = 0
    parents_created = 0
    errors: list[str] = []

    for path in files:
        relative_path = path.relative_to(settings.vault_path).as_posix()
        try:
            current_hash = calculate_sha256(path)
            if known_hashes.get(relative_path) == current_hash:
                unchanged += 1
                index_progress.advance(relative_path, phase="unchanged")
                continue
            index_progress.advance(relative_path)

            document, prepared_chunks = prepare_document(path)
            replace_document(
                path=relative_path,
                title=document.title,
                sha256=document.sha256,
                mtime=document.mtime,
                metadata=document.metadata,
                links=document.links,
                chunks=prepared_chunks,
                parents=[
                    {
                        "parent_index": parent.index,
                        "heading": parent.heading,
                        "content": parent.content,
                    }
                    for parent in document.parents
                ],
            )
            indexed += 1
            chunks_created += len(prepared_chunks)
            parents_created += len(document.parents)
        except Exception as exc:
            logger.exception(
                "index_document_failed",
                extra={"path": relative_path, "mode": "incremental"},
            )
            errors.append(f"{relative_path}: {PUBLIC_INDEX_ERROR}")

    return {
        "scanned": len(files),
        "indexed": indexed,
        "unchanged": unchanged,
        "deleted": deleted,
        "chunks_created": chunks_created,
        "parents_created": parents_created,
        "errors": errors,
        "rebuilt": False,
        "rebuild_reasons": [],
        "stats": get_stats(),
    }


def _index_vault() -> dict:
    init_db()
    files = discover_markdown_files(settings.vault_path)
    index_progress.start(len(files))
    embedding_dimension = get_embedding_dimension()

    configuration = build_index_configuration(
        vault_path=settings.vault_path,
        embedding_provider=EMBEDDING_PROVIDER,
        embedding_model=settings.embedding_model,
        embedding_dimension=embedding_dimension,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_prefix_scheme=resolve_scheme(
            settings.embedding_model,
            settings.embedding_prefix_scheme,
        ).id,
        chunker_version=CHUNKER_VERSION,
        parent_child_chunking_enabled=(
            settings.parent_child_chunking_enabled
        ),
        parent_chunk_size=settings.parent_chunk_size,
        child_chunk_size=settings.child_chunk_size,
        child_chunk_overlap=settings.child_chunk_overlap,
    )
    serialized_configuration = serialize_index_configuration(configuration)
    stored_configuration = get_meta(INDEX_FINGERPRINT_KEY)
    configuration_changes = changed_configuration_fields(
        stored_configuration,
        configuration,
    )
    existing_stats = get_stats()
    known_hashes = get_document_hashes()

    if configuration_changes:
        return _atomic_rebuild(
            files=files,
            serialized_configuration=serialized_configuration,
            configuration_changes=configuration_changes,
            stored_configuration=stored_configuration,
            existing_stats=existing_stats,
            known_hashes=known_hashes,
        )
    return _update_incrementally(files=files, known_hashes=known_hashes)


def index_vault() -> dict:
    if not _indexing_lock.acquire(blocking=False):
        raise RuntimeError("Ya hay una indexación en curso.")
    try:
        result = _index_vault()
    except Exception as exc:
        logger.exception("index_run_failed")
        index_progress.finish(error=PUBLIC_INDEX_RUN_ERROR)
        raise
    else:
        # La matriz y el grafo en memoria deben reflejar el índice recién
        # escrito.
        vector_store.invalidate()
        graph_store.invalidate()
        index_progress.finish(result=result)
        return result
    finally:
        _indexing_lock.release()


def start_background_indexing() -> dict[str, Any]:
    """Lanza la indexación en un hilo y devuelve el progreso inicial."""
    if index_progress.snapshot()["running"]:
        return index_progress.snapshot()

    def _run() -> None:
        try:
            index_vault()
        except Exception:
            # index_vault ya ha registrado el error en el progreso.
            pass

    index_progress.start(0)
    Thread(target=_run, name="rag-indexer", daemon=True).start()
    return index_progress.snapshot()

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dataclasses import dataclass
import json
import logging
import math
import re
from typing import Any, Iterator

import httpx

from app.config import settings
from app.model_profiles import active_think_value
from app.rag.token_estimate import estimate_token_count


logger = logging.getLogger(__name__)
PUBLIC_OLLAMA_ERROR = "No se pudo conectar con Ollama local."


EMBEDDING_BATCH_SIZE = 64
STATUS_TIMEOUT_SECONDS = 3.0
# Un pool pequeño evita saturar Ollama y a la vez elimina el tiempo
# muerto entre lotes consecutivos durante la indexación.
PARALLEL_EMBEDDING_THRESHOLD = 2


class OllamaError(RuntimeError):
    """Error base para fallos claros y seguros de Ollama."""


class OllamaConnectionError(OllamaError):
    """Ollama no está accesible en la URL local configurada."""


class OllamaModelNotFoundError(OllamaError):
    """El modelo solicitado no está instalado en Ollama."""


class OllamaResponseError(OllamaError):
    """Ollama devolvió una respuesta inesperada o inválida."""


@dataclass(frozen=True)
class GenerationResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    estimated: bool

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@lru_cache(maxsize=1)
def get_ollama_client() -> httpx.Client:
    timeout = httpx.Timeout(
        settings.ollama_timeout_seconds,
        connect=min(10.0, settings.ollama_timeout_seconds),
        write=min(30.0, settings.ollama_timeout_seconds),
        pool=min(10.0, settings.ollama_timeout_seconds),
    )
    return httpx.Client(
        base_url=f"{settings.ollama_base_url}/",
        timeout=timeout,
        trust_env=False,
        headers={"Accept": "application/json"},
        limits=httpx.Limits(
            max_connections=max(4, settings.embedding_workers + 2),
            max_keepalive_connections=max(4, settings.embedding_workers + 2),
        ),
    )


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()
    if isinstance(payload, dict):
        return str(payload.get("error", "")).strip()
    return ""


def _request_json(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    model: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {}
    if json_body is not None:
        request_kwargs["json"] = json_body
    if timeout is not None:
        request_kwargs["timeout"] = timeout
    try:
        response = get_ollama_client().request(
            method,
            path,
            **request_kwargs,
        )
    except httpx.TimeoutException as exc:
        effective_timeout = (
            timeout
            if timeout is not None
            else settings.ollama_timeout_seconds
        )
        raise OllamaConnectionError(
            f"Ollama agotó el tiempo de espera en "
            f"{settings.ollama_base_url}. Comprueba que el servicio esté "
            f"iniciado; la primera carga de un modelo puede tardar. "
            f"Tiempo de espera: {effective_timeout:g} s."
        ) from exc
    except httpx.RequestError as exc:
        raise OllamaConnectionError(
            f"No se puede conectar con Ollama en "
            f"{settings.ollama_base_url}. Instala Ollama, inicia la "
            f"aplicación o ejecuta 'ollama serve', y vuelve a intentarlo."
        ) from exc

    detail = _error_detail(response)
    if response.status_code == 404 and model:
        raise OllamaModelNotFoundError(
            f"El modelo local '{model}' no está instalado en Ollama. "
            f"Ejecuta: ollama pull {model}"
        )
    if response.is_error:
        suffix = f": {detail}" if detail else ""
        raise OllamaResponseError(
            f"Ollama devolvió HTTP {response.status_code}{suffix}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise OllamaResponseError(
            "Ollama devolvió una respuesta que no es JSON válido."
        ) from exc
    if not isinstance(payload, dict):
        raise OllamaResponseError(
            "Ollama devolvió una respuesta JSON con un formato inesperado."
        )
    return payload


def _validate_embeddings(
    raw_embeddings: Any,
    *,
    expected_count: int,
) -> list[list[float]]:
    if not isinstance(raw_embeddings, list):
        raise OllamaResponseError(
            "La respuesta de Ollama no contiene una lista de embeddings."
        )
    if len(raw_embeddings) != expected_count:
        raise OllamaResponseError(
            "Ollama devolvió un número de embeddings distinto al número "
            "de textos enviados."
        )

    vectors: list[list[float]] = []
    for raw_vector in raw_embeddings:
        if not isinstance(raw_vector, list) or not raw_vector:
            raise OllamaResponseError(
                "Ollama devolvió un embedding vacío o inválido."
            )
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError) as exc:
            raise OllamaResponseError(
                "Ollama devolvió un embedding con valores no numéricos."
            ) from exc
        if not all(math.isfinite(value) for value in vector):
            raise OllamaResponseError(
                "Ollama devolvió un embedding con valores no finitos."
            )
        vectors.append(vector)

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise OllamaResponseError(
            "Ollama devolvió embeddings con dimensiones diferentes."
        )
    return vectors


def _embed_batch(batch: list[str]) -> list[list[float]]:
    payload = _request_json(
        "POST",
        "api/embed",
        json_body={
            "model": settings.embedding_model,
            "input": batch,
            "truncate": True,
            "keep_alive": settings.ollama_keep_alive,
        },
        model=settings.embedding_model,
    )
    return _validate_embeddings(
        payload.get("embeddings"),
        expected_count=len(batch),
    )


def create_embeddings(texts: list[str]) -> list[list[float]]:
    """Crea embeddings por lotes conservando el orden de entrada.

    Los lotes se envían en paralelo con un pool acotado: Ollama atiende
    la siguiente petición mientras devuelve la anterior, lo que reduce
    de forma notable el tiempo total de indexación sin sobrecargarlo.
    """
    if not texts:
        return []

    batches = [
        texts[start : start + EMBEDDING_BATCH_SIZE]
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE)
    ]

    if (
        len(batches) < PARALLEL_EMBEDDING_THRESHOLD
        or settings.embedding_workers <= 1
    ):
        results = [_embed_batch(batch) for batch in batches]
    else:
        workers = min(settings.embedding_workers, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_embed_batch, batches))

    vectors: list[list[float]] = []
    expected_dimension: int | None = None
    for batch_vectors in results:
        batch_dimension = len(batch_vectors[0])
        if (
            expected_dimension is not None
            and batch_dimension != expected_dimension
        ):
            raise OllamaResponseError(
                "La dimensión de los embeddings cambió entre lotes."
            )
        expected_dimension = batch_dimension
        vectors.extend(batch_vectors)
    return vectors


def create_query_embedding(text: str) -> list[float]:
    return create_embeddings([text])[0]


def get_embedding_dimension() -> int:
    return len(create_query_embedding("comprobación de dimensión"))


def _without_internal_thinking(content: str) -> str:
    return re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def generate_rag_response(
    instructions: str,
    prompt: str,
    *,
    num_predict: int | None = None,
    temperature: float = 0.2,
    context_window_tokens: int | None = None,
) -> str:
    return generate_rag_response_with_metrics(
        instructions,
        prompt,
        num_predict=num_predict,
        temperature=temperature,
        context_window_tokens=context_window_tokens,
    ).content


def _response_token_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def generate_rag_response_with_metrics(
    instructions: str,
    prompt: str,
    *,
    num_predict: int | None = None,
    temperature: float = 0.2,
    context_window_tokens: int | None = None,
) -> GenerationResult:
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": (
            num_predict
            if num_predict is not None
            else settings.max_output_tokens
        ),
    }
    if context_window_tokens is not None:
        options["num_ctx"] = int(context_window_tokens)
    payload = _request_json(
        "POST",
        "api/chat",
        json_body={
            "model": settings.chat_model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": active_think_value(),
            "keep_alive": settings.ollama_keep_alive,
            "options": options,
        },
        model=settings.chat_model,
    )
    message = payload.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OllamaResponseError(
            "Ollama no devolvió contenido para la respuesta."
        )
    answer = _without_internal_thinking(content)
    if not answer:
        raise OllamaResponseError(
            "Ollama sólo devolvió razonamiento interno y ninguna respuesta."
        )
    prompt_tokens = _response_token_count(
        payload.get("prompt_eval_count")
    )
    completion_tokens = _response_token_count(payload.get("eval_count"))
    estimated = prompt_tokens is None or completion_tokens is None
    if prompt_tokens is None:
        # Ollama suele informar prompt_eval_count. Esta estimación mantiene
        # visible la métrica con versiones/modelos que no lo devuelvan.
        prompt_tokens = estimate_token_count(instructions + prompt)
    if completion_tokens is None:
        completion_tokens = estimate_token_count(answer)
    return GenerationResult(
        content=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated=estimated,
    )


def _canonical_model_name(value: str) -> str:
    normalized = value.strip().casefold()
    final_component = normalized.rsplit("/", 1)[-1]
    if ":" not in final_component:
        normalized += ":latest"
    return normalized


def _installed_model_names(payload: dict[str, Any]) -> set[str]:
    models = payload.get("models")
    if not isinstance(models, list):
        raise OllamaResponseError(
            "Ollama devolvió una lista de modelos con formato inesperado."
        )
    names: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                names.add(_canonical_model_name(value))
    return names


def get_model_show(model: str) -> dict[str, Any] | None:
    """Lee los metadatos locales de un modelo instalado (`api/show`).

    No descarga nada: sólo consulta un modelo que ya está en Ollama. Se
    usa para saber la ventana de contexto real de un modelo, incluidos
    los que no tienen perfil propio. Un fallo (Ollama apagado, modelo no
    instalado, respuesta inesperada) se degrada a `None`, nunca se lanza.
    """
    try:
        payload = _request_json(
            "POST",
            "api/show",
            json_body={"model": model},
            model=model,
            timeout=STATUS_TIMEOUT_SECONDS,
        )
    except OllamaError:
        return None
    return payload if isinstance(payload, dict) else None


def get_ollama_status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "accessible": False,
        "base_url": settings.ollama_base_url,
        "version": None,
        "chat_model": {
            "name": settings.chat_model,
            "installed": None,
        },
        "embedding_model": {
            "name": settings.embedding_model,
            "installed": None,
        },
        "error": None,
    }
    try:
        version_payload = _request_json(
            "GET",
            "api/version",
            timeout=STATUS_TIMEOUT_SECONDS,
        )
    except OllamaError as exc:
        logger.exception("ollama_version_status_failed")
        result["error"] = PUBLIC_OLLAMA_ERROR
        return result

    result["accessible"] = True
    version = version_payload.get("version")
    result["version"] = version if isinstance(version, str) else None

    try:
        models_payload = _request_json(
            "GET",
            "api/tags",
            timeout=STATUS_TIMEOUT_SECONDS,
        )
        installed = _installed_model_names(models_payload)
    except OllamaError as exc:
        logger.exception("ollama_models_status_failed")
        result["error"] = PUBLIC_OLLAMA_ERROR
        return result

    result["chat_model"]["installed"] = (
        _canonical_model_name(settings.chat_model) in installed
    )
    result["embedding_model"]["installed"] = (
        _canonical_model_name(settings.embedding_model) in installed
    )
    return result


# ----------------------------------------------------------------------
# Generación en streaming
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class StreamChunk:
    """Fragmento emitido durante la generación en streaming."""

    text: str
    done: bool
    metrics: GenerationResult | None = None


def stream_rag_response(
    instructions: str,
    prompt: str,
    *,
    num_predict: int | None = None,
    temperature: float = 0.2,
    context_window_tokens: int | None = None,
) -> Iterator[StreamChunk]:
    """Emite la respuesta token a token.

    Filtra los bloques `<think>` sobre la marcha, de modo que el
    razonamiento interno de modelos como Qwen nunca llega a la interfaz
    aunque el modelo lo produzca.
    """
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": (
            num_predict
            if num_predict is not None
            else settings.max_output_tokens
        ),
    }
    if context_window_tokens is not None:
        options["num_ctx"] = int(context_window_tokens)

    body = {
        "model": settings.chat_model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "think": active_think_value(),
        "keep_alive": settings.ollama_keep_alive,
        "options": options,
    }

    filter_state = _ThinkFilter()
    emitted = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    try:
        with get_ollama_client().stream(
            "POST",
            "api/chat",
            json=body,
        ) as response:
            if response.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"El modelo local '{settings.chat_model}' no está "
                    f"instalado en Ollama. Ejecuta: "
                    f"ollama pull {settings.chat_model}"
                )
            if response.is_error:
                response.read()
                raise OllamaResponseError(
                    f"Ollama devolvió HTTP {response.status_code}"
                )

            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(payload, dict):
                    continue

                message = payload.get("message")
                if isinstance(message, dict):
                    piece = message.get("content")
                    if isinstance(piece, str) and piece:
                        visible = filter_state.feed(piece)
                        if visible:
                            emitted += len(visible)
                            yield StreamChunk(text=visible, done=False)

                if payload.get("done"):
                    prompt_tokens = _response_token_count(
                        payload.get("prompt_eval_count")
                    )
                    completion_tokens = _response_token_count(
                        payload.get("eval_count")
                    )
                    break
    except httpx.TimeoutException as exc:
        raise OllamaConnectionError(
            f"Ollama agotó el tiempo de espera en "
            f"{settings.ollama_base_url}. La primera carga de un modelo "
            f"puede tardar."
        ) from exc
    except httpx.RequestError as exc:
        raise OllamaConnectionError(
            f"No se puede conectar con Ollama en "
            f"{settings.ollama_base_url}. Inicia el servicio y reinténtalo."
        ) from exc

    tail = filter_state.flush()
    if tail:
        emitted += len(tail)
        yield StreamChunk(text=tail, done=False)

    if emitted == 0:
        raise OllamaResponseError(
            "Ollama no devolvió contenido para la respuesta."
        )

    estimated = prompt_tokens is None or completion_tokens is None
    if prompt_tokens is None:
        prompt_tokens = estimate_token_count(instructions + prompt)
    if completion_tokens is None:
        completion_tokens = max(1, math.ceil(emitted / 4))

    yield StreamChunk(
        text="",
        done=True,
        metrics=GenerationResult(
            content="",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated=estimated,
        ),
    )


class _ThinkFilter:
    """Suprime bloques <think>...</think> en un flujo incremental."""

    OPEN = "<think"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, piece: str) -> str:
        self._buffer += piece
        output: list[str] = []

        while self._buffer:
            if self._inside:
                end = self._buffer.find(self.CLOSE)
                if end == -1:
                    keep = len(self.CLOSE) - 1
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self._buffer = self._buffer[end + len(self.CLOSE):]
                self._inside = False
                continue

            start = self._buffer.find(self.OPEN)
            if start == -1:
                # Retiene una cola por si la etiqueta llega partida.
                keep = len(self.OPEN) - 1
                if len(self._buffer) > keep:
                    output.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                break

            output.append(self._buffer[:start])
            closing = self._buffer.find(">", start)
            if closing == -1:
                self._buffer = self._buffer[start:]
                break
            self._buffer = self._buffer[closing + 1:]
            self._inside = True

        return "".join(output)

    def flush(self) -> str:
        if self._inside:
            self._buffer = ""
            return ""
        remaining = self._buffer
        self._buffer = ""
        return remaining


def warmup_models() -> dict[str, Any]:
    """Precarga los modelos en memoria para que la demostración no espere."""
    report: dict[str, Any] = {"chat": False, "embedding": False}
    try:
        _request_json(
            "POST",
            "api/embed",
            json_body={
                "model": settings.embedding_model,
                "input": ["calentamiento"],
                "keep_alive": settings.ollama_keep_alive,
            },
            model=settings.embedding_model,
            timeout=settings.ollama_timeout_seconds,
        )
        report["embedding"] = True
    except OllamaError as exc:
        logger.exception("ollama_embedding_warmup_failed")
        report["embedding_error"] = PUBLIC_OLLAMA_ERROR
    try:
        _request_json(
            "POST",
            "api/chat",
            json_body={
                "model": settings.chat_model,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "think": active_think_value(),
                "keep_alive": settings.ollama_keep_alive,
                "options": {"num_predict": 1},
            },
            model=settings.chat_model,
            timeout=settings.ollama_timeout_seconds,
        )
        report["chat"] = True
    except OllamaError as exc:
        logger.exception("ollama_chat_warmup_failed")
        report["chat_error"] = PUBLIC_OLLAMA_ERROR
    return report

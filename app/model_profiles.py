from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from app.config import BASE_DIR, settings, validate_local_model_name
from app.env_config import update_env_value


CHAT_MODEL_ENV_KEY = "OLLAMA_CHAT_MODEL"
RERANK_ENV_KEY = "RAG_RERANK"
RERANK_BACKEND_ENV_KEY = "RAG_RERANK_BACKEND"


@dataclass(frozen=True)
class ChatModelProfile:
    id: str
    label: str
    model: str
    description: str
    ram: str
    # Techo real del modelo. Pedir más contexto del que se entrenó no da
    # error: degrada la respuesta en silencio, que es peor.
    context_window: int = 32768
    # Modelos de razonamiento: emiten cadena de pensamiento que no debe
    # llegar a la interfaz ni contar como respuesta.
    reasoning: bool = False
    # Valor del parámetro `think` de Ollama para este modelo.
    think: bool | str = False
    # Los perfiles intermedios compensan una segunda pasada local.
    rerank_enabled: bool = False
    rerank_backend: str = "onnx"


CHAT_MODEL_PROFILES = {
    "light": ChatModelProfile(
        id="light",
        label="Ligero",
        model="qwen3.5:0.8b",
        description="Más rápido y con menor consumo.",
        ram="8 GB recomendados",
    ),
    "balanced": ChatModelProfile(
        id="balanced",
        label="Equilibrado",
        model="qwen3.5:2b",
        rerank_enabled=True,
        description="Mejor comprensión con consumo moderado.",
        ram="8–12 GB recomendados",
    ),
    "quality": ChatModelProfile(
        id="quality",
        label="Calidad",
        model="qwen3.5:4b",
        rerank_enabled=True,
        description="Mejor síntesis y seguimiento de instrucciones.",
        ram="12–16 GB recomendados",
        context_window=32768,
    ),
    "qwen3_14b": ChatModelProfile(
        id="qwen3_14b",
        label="Qwen3 14B",
        model="qwen3:14b",
        description=(
            "Modelo denso más grande, con razonamiento propio activable. "
            "Mejor síntesis que los perfiles Qwen3.5, a costa de más "
            "memoria y de respuestas algo más lentas."
        ),
        ram="16–20 GB recomendados",
        context_window=40960,
        reasoning=True,
        # A diferencia de gpt-oss (que usa un nivel de esfuerzo), Qwen3
        # activa o desactiva el razonamiento con un booleano en Ollama.
        think=True,
    ),
    "llama3": ChatModelProfile(
        id="llama3",
        label="Llama 3",
        model="llama3:latest",
        description=(
            "Buena redacción en castellano. Contexto corto: 8K, así que "
            "el redactor trabaja con menos fragmentos."
        ),
        ram="16 GB recomendados",
        context_window=8192,
        reasoning=False,
        think=False,
    ),
    "gpt_oss": ChatModelProfile(
        id="gpt_oss",
        label="GPT-OSS",
        model="gpt-oss:latest",
        description=(
            "Contexto muy amplio (128K) y razonamiento propio. Es el "
            "más exigente en memoria de todos los perfiles."
        ),
        ram="24 GB recomendados (16 GB de VRAM)",
        context_window=131072,
        reasoning=True,
        # gpt-oss razona siempre por diseño; el esfuerzo mínimo es lo
        # más rápido y el razonamiento se filtra igualmente.
        think="low",
    ),
}
_profile_lock = Lock()


DEFAULT_CONTEXT_WINDOW = 32768

# Ollama nombra la capacidad de contexto según la arquitectura del modelo
# (p.ej. `llama.context_length`, `qwen2.context_length`), nunca con una
# clave fija: hay que buscarla por sufijo.
_CONTEXT_LENGTH_SUFFIX = ".context_length"
_MIN_PLAUSIBLE_CONTEXT_WINDOW = 512
_MAX_PLAUSIBLE_CONTEXT_WINDOW = 10_000_000

ContextSource = Literal["profile", "ollama", "profile+ollama", "fallback"]


@dataclass(frozen=True)
class ModelContextInfo:
    """Capacidad de contexto resuelta para un modelo, con su procedencia.

    `verified` distingue un dato respaldado por el perfil o por Ollama de
    un fallback prudente para un modelo desconocido: la interfaz no debe
    presentar ambos casos como si tuvieran la misma certeza.
    """

    model: str
    profile_id: str | None
    model_context_window: int
    source: ContextSource
    verified: bool
    warning: str | None = None


def list_chat_model_profiles() -> list[dict[str, Any]]:
    return [asdict(profile) for profile in CHAT_MODEL_PROFILES.values()]


def _profile_for_model(model_name: str) -> ChatModelProfile | None:
    normalized = model_name.strip().casefold()
    for profile in CHAT_MODEL_PROFILES.values():
        if profile.model.casefold() == normalized:
            return profile
    return None


def active_profile() -> ChatModelProfile | None:
    """Perfil que corresponde al modelo configurado ahora mismo."""
    return _profile_for_model(settings.chat_model)


def context_window_from_model_info(model_info: Any) -> int | None:
    """Extrae la ventana de contexto de `model_info` (respuesta de Ollama).

    Se valida tipo y rango antes de aceptar cualquier valor: unos
    metadatos corruptos o disparatados no deben colarse como techo real.
    Si aparece más de una clave `*.context_length` se toma la más baja,
    coherente con la política general de preferir el valor conservador.
    """
    if not isinstance(model_info, dict):
        return None
    candidates: list[int] = []
    for key, value in model_info.items():
        if not isinstance(key, str) or not key.endswith(
            _CONTEXT_LENGTH_SUFFIX
        ):
            continue
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if _MIN_PLAUSIBLE_CONTEXT_WINDOW <= parsed <= _MAX_PLAUSIBLE_CONTEXT_WINDOW:
            candidates.append(parsed)
    return min(candidates) if candidates else None


def _fetch_ollama_context_window(model_name: str) -> int | None:
    """Intenta leer la ventana real desde los metadatos locales de Ollama.

    Nunca lanza ni descarga nada: usa `api/show`, que sólo lee metadatos
    de un modelo ya instalado. Si Ollama está apagado, el modelo no está
    instalado o la respuesta es inesperada, se degrada a `None` sin
    interrumpir la resolución de contexto.
    """
    try:
        from app.rag.ollama_client import get_model_show

        payload = get_model_show(model_name)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return context_window_from_model_info(payload.get("model_info"))


def resolve_model_context(model_name: str) -> ModelContextInfo:
    """Resuelve la capacidad de contexto real o conservadora de un modelo.

    Orden de resolución:
    1. Metadatos locales de Ollama (`api/show`), cuando estén disponibles.
    2. Perfil conocido en `CHAT_MODEL_PROFILES`.
    3. Si ambos existen y difieren, se usa el valor más conservador y se
       adjunta una advertencia explicando el porqué.
    4. Sin ninguno de los dos, fallback prudente de `DEFAULT_CONTEXT_WINDOW`
       marcado como no verificado.
    """
    normalized_model = model_name.strip()
    profile = _profile_for_model(normalized_model)
    profile_window = profile.context_window if profile else None
    profile_id = profile.id if profile else None
    ollama_window = _fetch_ollama_context_window(normalized_model)

    if profile_window is not None and ollama_window is not None:
        if profile_window == ollama_window:
            return ModelContextInfo(
                model=normalized_model,
                profile_id=profile_id,
                model_context_window=profile_window,
                source="profile+ollama",
                verified=True,
                warning=None,
            )
        conservative = min(profile_window, ollama_window)
        return ModelContextInfo(
            model=normalized_model,
            profile_id=profile_id,
            model_context_window=conservative,
            source="profile+ollama",
            verified=True,
            warning=(
                f"El perfil declara {profile_window} tokens y Ollama "
                f"informa {ollama_window}; se usa el valor más "
                "conservador."
            ),
        )
    if ollama_window is not None:
        return ModelContextInfo(
            model=normalized_model,
            profile_id=profile_id,
            model_context_window=ollama_window,
            source="ollama",
            verified=True,
            warning=None,
        )
    if profile_window is not None:
        return ModelContextInfo(
            model=normalized_model,
            profile_id=profile_id,
            model_context_window=profile_window,
            source="profile",
            verified=True,
            warning=None,
        )
    return ModelContextInfo(
        model=normalized_model,
        profile_id=None,
        model_context_window=DEFAULT_CONTEXT_WINDOW,
        source="fallback",
        verified=False,
        warning=(
            "Modelo sin perfil conocido ni metadatos de Ollama; se aplica "
            f"un límite prudente de {DEFAULT_CONTEXT_WINDOW} tokens."
        ),
    )


def effective_context_window(
    configured_context_window: int,
    model_context_window: int,
) -> int:
    """Ventana realmente enviada a Ollama: nunca por encima de la capacidad."""
    return max(1, min(int(configured_context_window), int(model_context_window)))


def active_think_value() -> bool | str:
    profile = active_profile()
    return profile.think if profile else False


def active_chat_model_profile() -> str | None:
    selected_model = settings.chat_model.casefold()
    for profile in CHAT_MODEL_PROFILES.values():
        if profile.model.casefold() == selected_model:
            return profile.id
    return None


def configure_chat_model_profile(
    profile_id: str,
    *,
    env_path: Path | None = None,
) -> dict[str, Any]:
    profile = CHAT_MODEL_PROFILES.get(profile_id)
    if profile is None:
        available = ", ".join(CHAT_MODEL_PROFILES)
        raise ValueError(
            f"Perfil de chat desconocido: {profile_id}. "
            f"Perfiles disponibles: {available}."
        )

    model = validate_local_model_name(profile.model, CHAT_MODEL_ENV_KEY)
    with _profile_lock:
        update_env_value(
            CHAT_MODEL_ENV_KEY,
            model,
            env_path=env_path or BASE_DIR / ".env",
            quote_value=False,
        )
        update_env_value(
            RERANK_ENV_KEY,
            "true" if profile.rerank_enabled else "false",
            env_path=env_path or BASE_DIR / ".env",
            quote_value=False,
        )
        update_env_value(
            RERANK_BACKEND_ENV_KEY,
            profile.rerank_backend,
            env_path=env_path or BASE_DIR / ".env",
            quote_value=False,
        )
        settings.chat_model = model
        settings.rerank_enabled = profile.rerank_enabled
        settings.rerank_backend = profile.rerank_backend
    return asdict(profile)

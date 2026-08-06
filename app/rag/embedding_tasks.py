"""Prefijos de tarea para modelos de embedding asimétricos.

Modelos como `nomic-embed-text`, la familia E5 o BGE se entrenaron con
prefijos distintos para el documento y para la consulta. Omitirlos degrada
la recuperación de forma medible porque consulta y pasaje quedan en
regiones distintas del espacio vectorial.

El esquema forma parte de la huella del índice (`index_fingerprint`), de
modo que cambiarlo fuerza una reconstrucción atómica automática.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixScheme:
    id: str
    document: str
    query: str


PREFIX_SCHEMES: dict[str, PrefixScheme] = {
    "none": PrefixScheme(id="none", document="", query=""),
    "nomic": PrefixScheme(
        id="nomic",
        document="search_document: ",
        query="search_query: ",
    ),
    "e5": PrefixScheme(id="e5", document="passage: ", query="query: "),
    "bge": PrefixScheme(
        id="bge",
        document="",
        query=(
            "Represent this sentence for searching relevant passages: "
        ),
    ),
}

_AUTO_RULES: tuple[tuple[str, str], ...] = (
    ("nomic-embed", "nomic"),
    ("multilingual-e5", "e5"),
    ("intfloat/e5", "e5"),
    ("e5-", "e5"),
    ("bge-", "bge"),
    ("bge:", "bge"),
)


def detect_scheme_id(model_name: str) -> str:
    """Deduce el esquema de prefijos a partir del nombre del modelo."""
    normalized = model_name.strip().casefold()
    for needle, scheme_id in _AUTO_RULES:
        if needle in normalized:
            return scheme_id
    return "none"


def resolve_scheme(model_name: str, configured: str = "auto") -> PrefixScheme:
    """Resuelve el esquema efectivo respetando la configuración explícita."""
    requested = (configured or "auto").strip().casefold()
    if requested in {"", "auto"}:
        return PREFIX_SCHEMES[detect_scheme_id(model_name)]
    scheme = PREFIX_SCHEMES.get(requested)
    if scheme is None:
        available = ", ".join(sorted(PREFIX_SCHEMES))
        raise ValueError(
            f"RAG_EMBEDDING_PREFIX_SCHEME no válido: {configured}. "
            f"Valores admitidos: auto, {available}."
        )
    return scheme

"""Estimador de tokens único para todo el proyecto.

Se usa exclusivamente cuando Ollama no informa contadores reales o para
proyectar el uso de un prompt que todavía no se ha enviado. El resultado
debe presentarse siempre como estimado, nunca como medido: es una
aproximación de ~4 caracteres por token, no un recuento real del
tokenizador del modelo.
"""

from __future__ import annotations

import math


def estimate_token_count(text: str) -> int:
    normalized = text or ""
    if not normalized.strip():
        return 0
    return max(1, math.ceil(len(normalized) / 4))

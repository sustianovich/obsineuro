"""Vigencia temporal y metadatos derivados del documento.

En un corpus normativo el estado no basta. Un protocolo derogado en marzo
sigue siendo recuperable si nadie tocó el campo `estado`, y una norma con
entrada en vigor futura no debería responder como si ya rigiera.

Aquí se calcula la vigencia a partir de las fechas del frontmatter y se
deja registrada junto al documento, de modo que la recuperación pueda
filtrar por ella y la interfaz pueda avisar aunque no se filtre.

Claves reconocidas (se admiten variantes en castellano e inglés):

  entrada en vigor : fecha_vigencia, vigente_desde, valid_from, fecha_inicio
  derogación       : fecha_derogacion, derogado_el, valid_until, fecha_fin
  revisión prevista: fecha_revision, revisar_el, review_date
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any


VALID_FROM_KEYS = (
    "fecha_vigencia",
    "vigente_desde",
    "fecha_inicio",
    "valid_from",
    "effective_date",
)
VALID_UNTIL_KEYS = (
    "fecha_derogacion",
    "fecha_derogación",
    "derogado_el",
    "fecha_fin",
    "valid_until",
    "expiry_date",
)
REVIEW_KEYS = (
    "fecha_revision",
    "fecha_revisión",
    "revisar_el",
    "review_date",
)

VIGENCIA_VIGENTE = "vigente"
VIGENCIA_FUTURA = "futura"
VIGENCIA_CADUCADA = "caducada"
VIGENCIA_DESCONOCIDA = "desconocida"


@dataclass(frozen=True)
class Vigencia:
    estado_temporal: str
    valid_from: str | None
    valid_until: str | None
    review_date: str | None
    revision_vencida: bool
    derogado_por_callout: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_date(value: Any) -> date | None:
    """Acepta date, datetime y las cadenas ISO habituales."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("/", "-")
    for pattern in ("%Y-%m-%d", "%d-%m-%Y", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(text[: len(pattern) + 6], pattern).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text[:19]).date()
    except ValueError:
        return None


def _first_date(metadata: dict[str, Any], keys: tuple[str, ...]) -> date | None:
    for key in keys:
        if key in metadata:
            parsed = parse_date(metadata[key])
            if parsed is not None:
                return parsed
    return None


def compute_vigencia(
    metadata: dict[str, Any],
    *,
    deprecated_callout: bool = False,
    today: date | None = None,
) -> Vigencia:
    """Determina el estado temporal del documento."""
    reference = today or date.today()
    valid_from = _first_date(metadata, VALID_FROM_KEYS)
    valid_until = _first_date(metadata, VALID_UNTIL_KEYS)
    review = _first_date(metadata, REVIEW_KEYS)

    if valid_until is not None and valid_until <= reference:
        estado = VIGENCIA_CADUCADA
    elif valid_from is not None and valid_from > reference:
        estado = VIGENCIA_FUTURA
    elif valid_from is not None or valid_until is not None:
        estado = VIGENCIA_VIGENTE
    else:
        estado = VIGENCIA_DESCONOCIDA

    # Un callout de derogación es una declaración explícita del autor y
    # pesa más que la ausencia de fecha de fin.
    if deprecated_callout and estado != VIGENCIA_FUTURA:
        estado = VIGENCIA_CADUCADA

    return Vigencia(
        estado_temporal=estado,
        valid_from=valid_from.isoformat() if valid_from else None,
        valid_until=valid_until.isoformat() if valid_until else None,
        review_date=review.isoformat() if review else None,
        revision_vencida=bool(review and review <= reference),
        derogado_por_callout=deprecated_callout,
    )


def build_enrichment(
    metadata: dict[str, Any],
    *,
    tags: list[str],
    vigencia: Vigencia,
    anchors: dict[str, str],
    callouts: list[dict[str, str]],
) -> dict[str, Any]:
    """Metadatos derivados que se guardan junto al documento.

    Van bajo claves con guion bajo porque `answer.build_context` ya
    considera internas esas claves y no las vuelca al modelo: son para
    filtrar y para avisar en la interfaz, no para que el redactor las
    cite como si fueran contenido.
    """
    enriched = dict(metadata)
    enriched["_tags"] = tags
    enriched["_vigencia"] = vigencia.as_dict()
    enriched["_anclas"] = anchors
    enriched["_callouts"] = callouts
    return enriched


def document_tags(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("_tags", [])
    return [str(value) for value in raw] if isinstance(raw, list) else []


def document_vigencia(metadata: dict[str, Any]) -> str:
    raw = metadata.get("_vigencia")
    if isinstance(raw, dict):
        return str(raw.get("estado_temporal", VIGENCIA_DESCONOCIDA))
    return VIGENCIA_DESCONOCIDA


def vigencia_details(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("_vigencia")
    return dict(raw) if isinstance(raw, dict) else {}

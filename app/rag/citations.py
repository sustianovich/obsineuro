"""Validación de las referencias [n] que emite el redactor.

El modelo puede citar `[9]` habiendo recuperado seis fragmentos. La
interfaz oculta esas citas al renderizar, pero la respuesta se guardaba
en el historial con la referencia inventada dentro, y de ahí pasaba a
las exportaciones y a la memoria conversacional.

Aquí se comprueban las referencias antes de persistir. Una cita fuera de
rango no se borra en silencio: se marca, se cuenta y se informa. Borrarla
dejaría una afirmación sin respaldo aparente, que es peor que una cita
visiblemente rota.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CITATION_RE = re.compile(r"\[(\d{1,3})\]")
INVALID_MARK = "[?]"


@dataclass(frozen=True)
class CitationReport:
    total: int = 0
    valid: int = 0
    invalid_references: tuple[int, ...] = ()
    unused_sources: tuple[int, ...] = ()
    uncited_answer: bool = False

    @property
    def has_invalid(self) -> bool:
        return bool(self.invalid_references)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "valid": self.valid,
            "invalid_references": list(self.invalid_references),
            "unused_sources": list(self.unused_sources),
            "uncited_answer": self.uncited_answer,
            "has_invalid": self.has_invalid,
        }


def analyse_citations(answer: str, source_count: int) -> CitationReport:
    """Comprueba qué referencias del texto existen de verdad."""
    if not answer.strip():
        return CitationReport()

    found = [int(match.group(1)) for match in CITATION_RE.finditer(answer)]
    valid_range = set(range(1, source_count + 1))

    invalid = sorted({value for value in found if value not in valid_range})
    used = {value for value in found if value in valid_range}
    unused = sorted(valid_range - used)

    return CitationReport(
        total=len(found),
        valid=len(found) - sum(1 for value in found if value in invalid),
        invalid_references=tuple(invalid),
        unused_sources=tuple(unused),
        uncited_answer=bool(source_count and not used),
    )


def mark_invalid_citations(answer: str, source_count: int) -> str:
    """Sustituye las referencias inexistentes por una marca visible."""
    valid_range = set(range(1, source_count + 1))

    def _replace(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return match.group(0) if value in valid_range else INVALID_MARK

    return CITATION_RE.sub(_replace, answer)


def validate_answer(
    answer: str,
    source_count: int,
    *,
    abstained: bool = False,
) -> tuple[str, CitationReport, str | None]:
    """Devuelve la respuesta saneada, el informe y un aviso si procede.

    Una abstención dirigida por el verificador puede conservar fuentes en la
    interfaz para que la persona inspeccione la evidencia rechazada. Esa
    respuesta no pretende formular afirmaciones documentales y, por tanto,
    no debe marcarse como una respuesta ordinaria sin citas.
    """
    effective_source_count = 0 if abstained else source_count
    report = analyse_citations(answer, effective_source_count)
    if not report.has_invalid:
        return answer, report, None

    cleaned = mark_invalid_citations(answer, effective_source_count)
    listed = ", ".join(f"[{value}]" for value in report.invalid_references)
    warning = (
        f"La respuesta citaba {listed}, que no corresponde a ninguna "
        f"fuente recuperada ({effective_source_count} disponibles). "
        "Esas referencias se han marcado como no verificables."
    )
    return cleaned, report, warning

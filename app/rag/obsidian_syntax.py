"""Extracción de la sintaxis propia de Obsidian.

El vault ya contiene metadatos que el indexador no estaba aprovechando:

  - `#etiquetas` en el cuerpo y `tags:` en el frontmatter;
  - callouts `> [!tipo] Título`, que marcan advertencias y avisos de
    derogación;
  - `![[embeds]]`, que son transclusiones y no simples enlaces;
  - anclas de bloque `^identificador`, que permiten citar con precisión
    y resolver enlaces `[[nota#^ancla]]`.

Las etiquetas se normalizan (sin acentos, en minúscula) para que sirvan
como filtro fiable con independencia de cómo se escribieran.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


INLINE_TAG_RE = re.compile(r"(?<![\w&])#([A-Za-z\u00C0-\u024F][\w\u00C0-\u024F/-]*)")
CALLOUT_RE = re.compile(
    r"^\s{0,3}>\s*\[!(?P<tipo>[A-Za-z-]+)\](?P<plegado>[+-]?)\s*(?P<titulo>.*)$"
)
BLOCK_ANCHOR_RE = re.compile(r"(?:^|\s)\^([A-Za-z0-9][A-Za-z0-9-]*)\s*$")
CODE_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

# Los callouts que anuncian que un documento ya no rige son señal fuerte
# en un corpus normativo.
DEPRECATION_CALLOUTS = {"caution", "danger", "failure", "bug", "deprecated"}


@dataclass(frozen=True)
class Callout:
    tipo: str
    titulo: str
    contenido: str


def normalize_tag(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.strip().casefold())
    stripped = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return stripped.strip("/#-")


def _strip_code_regions(body: str) -> str:
    """Sustituye el interior de las vallas por líneas en blanco.

    Evita confundir comentarios `# encabezado` de Python o Bash con
    etiquetas de Obsidian. Conserva el número de líneas para que los
    índices sigan siendo comparables.
    """
    output: list[str] = []
    inside = False
    marker = ""
    for line in body.split("\n"):
        fence = CODE_FENCE_RE.match(line)
        if fence:
            if not inside:
                inside = True
                marker = fence.group(1)[0]
            elif fence.group(1)[0] == marker:
                inside = False
            output.append("")
            continue
        output.append("" if inside else line)
    return "\n".join(output)


def extract_inline_tags(body: str) -> list[str]:
    """Etiquetas `#etiqueta` del cuerpo, fuera de bloques de código."""
    clean = _strip_code_regions(body)
    tags: list[str] = []
    seen: set[str] = set()
    for match in INLINE_TAG_RE.finditer(clean):
        tag = normalize_tag(match.group(1))
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def extract_frontmatter_tags(metadata: dict[str, Any]) -> list[str]:
    """Etiquetas declaradas en el frontmatter (`tags` o `etiquetas`)."""
    raw = metadata.get("tags", metadata.get("etiquetas", []))
    if isinstance(raw, str):
        candidates = re.split(r"[,\s]+", raw)
    elif isinstance(raw, (list, tuple)):
        candidates = [str(value) for value in raw]
    else:
        return []

    tags: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        tag = normalize_tag(str(candidate))
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def collect_tags(metadata: dict[str, Any], body: str) -> list[str]:
    """Unión ordenada de las etiquetas del frontmatter y del cuerpo."""
    combined = extract_frontmatter_tags(metadata) + extract_inline_tags(body)
    seen: set[str] = set()
    result: list[str] = []
    for tag in combined:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result


def extract_callouts(body: str) -> list[Callout]:
    """Callouts `> [!tipo] Título` con su contenido."""
    callouts: list[Callout] = []
    lines = _strip_code_regions(body).split("\n")
    index = 0
    while index < len(lines):
        match = CALLOUT_RE.match(lines[index])
        if not match:
            index += 1
            continue
        contenido: list[str] = []
        index += 1
        while index < len(lines) and re.match(r"^\s{0,3}>", lines[index]):
            contenido.append(re.sub(r"^\s{0,3}>\s?", "", lines[index]))
            index += 1
        callouts.append(
            Callout(
                tipo=match.group("tipo").strip().casefold(),
                titulo=match.group("titulo").strip(),
                contenido="\n".join(contenido).strip(),
            )
        )
    return callouts


def has_deprecation_callout(callouts: list[Callout]) -> bool:
    return any(callout.tipo in DEPRECATION_CALLOUTS for callout in callouts)


def extract_block_anchors(body: str) -> dict[str, str]:
    """Anclas `^identificador` y la línea que identifican."""
    anchors: dict[str, str] = {}
    for line in _strip_code_regions(body).split("\n"):
        match = BLOCK_ANCHOR_RE.search(line)
        if not match:
            continue
        anchor = match.group(1).casefold()
        text = BLOCK_ANCHOR_RE.sub("", line).strip()
        if anchor and anchor not in anchors:
            anchors[anchor] = text
    return anchors


def searchable_supplement(
    tags: list[str],
    callouts: list[Callout],
) -> str:
    """Texto añadido al embedding para que etiquetas y avisos pesen.

    Sin esto, una etiqueta `#sicc-2025` es invisible para la búsqueda
    semántica aunque describa el documento mejor que su prosa.
    """
    parts: list[str] = []
    if tags:
        parts.append("Etiquetas: " + ", ".join(tags))
    for callout in callouts:
        encabezado = callout.titulo or callout.tipo
        parts.append(f"Aviso ({callout.tipo}): {encabezado}")
    return "\n".join(parts)

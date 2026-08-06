"""Troceado consciente de la estructura del Markdown.

El troceado anterior partía por párrafos y, si un párrafo excedía el
tamaño, por número de caracteres. Eso rompía bloques de código a mitad
(dejando vallas sin cerrar) y podía decapitar tablas, de modo que el
modelo recibía Markdown sintácticamente roto.

Aquí el documento se descompone primero en bloques —valla de código,
tabla, callout, lista, párrafo— y los bloques se empaquetan enteros. Sólo
se parte un bloque cuando por sí solo excede el tamaño máximo, y en ese
caso se parte por sus propias costuras:

  - el código se corta por líneas y cada parte se vuelve a vallar;
  - la tabla se corta por filas y se repite la cabecera en cada parte;
  - el resto cae al reparto por párrafos y, en último extremo, por
    caracteres.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator


FENCE_RE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
TABLE_DELIMITER_RE = re.compile(
    r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$"
)
LIST_ITEM_RE = re.compile(r"^\s{0,3}([-*+]|\d{1,9}[.)])\s+")
QUOTE_RE = re.compile(r"^\s{0,3}>")


@dataclass
class Block:
    """Unidad estructural del documento."""

    kind: str
    lines: list[str] = field(default_factory=list)
    fence: str = ""
    info: str = ""
    indent: str = ""
    header: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip("\n")

    def __len__(self) -> int:
        return len(self.text)


def _is_table_start(lines: list[str], index: int) -> bool:
    if "|" not in lines[index]:
        return False
    if index + 1 >= len(lines):
        return False
    return bool(TABLE_DELIMITER_RE.match(lines[index + 1]))


def parse_blocks(body: str) -> list[Block]:
    """Descompone el texto en bloques estructurales."""
    lines = body.split("\n")
    blocks: list[Block] = []
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]

        if not line.strip():
            index += 1
            continue

        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group("fence")
            character = marker[0]
            length = len(marker)
            block = Block(
                kind="code",
                lines=[line],
                fence=marker,
                info=fence_match.group("info").strip(),
                indent=fence_match.group("indent"),
            )
            index += 1
            while index < total:
                candidate = lines[index]
                block.lines.append(candidate)
                closing = FENCE_RE.match(candidate)
                if (
                    closing
                    and closing.group("fence")[0] == character
                    and len(closing.group("fence")) >= length
                    and not closing.group("info").strip()
                ):
                    index += 1
                    break
                index += 1
            blocks.append(block)
            continue

        if _is_table_start(lines, index):
            block = Block(
                kind="table",
                lines=[lines[index], lines[index + 1]],
                header=[lines[index], lines[index + 1]],
            )
            index += 2
            while index < total and "|" in lines[index] and lines[index].strip():
                block.lines.append(lines[index])
                index += 1
            blocks.append(block)
            continue

        if QUOTE_RE.match(line):
            block = Block(kind="callout", lines=[])
            while index < total and (
                QUOTE_RE.match(lines[index]) or lines[index].strip()
            ):
                if not QUOTE_RE.match(lines[index]) and not lines[
                    index
                ].startswith("  "):
                    break
                block.lines.append(lines[index])
                index += 1
            blocks.append(block)
            continue

        if LIST_ITEM_RE.match(line):
            block = Block(kind="list", lines=[])
            while index < total and lines[index].strip():
                if (
                    not LIST_ITEM_RE.match(lines[index])
                    and not lines[index].startswith(" ")
                ):
                    break
                block.lines.append(lines[index])
                index += 1
            blocks.append(block)
            continue

        block = Block(kind="paragraph", lines=[])
        while index < total and lines[index].strip():
            if (
                FENCE_RE.match(lines[index])
                or _is_table_start(lines, index)
                or QUOTE_RE.match(lines[index])
            ):
                break
            block.lines.append(lines[index])
            index += 1
        if block.lines:
            blocks.append(block)
        else:
            index += 1

    return [block for block in blocks if block.text]


# ----------------------------------------------------------------------
# Partido de bloques que por sí solos exceden el tamaño
# ----------------------------------------------------------------------
def _split_code_block(block: Block, max_chars: int) -> list[str]:
    """Parte código por líneas y vuelve a vallar cada trozo."""
    inner = block.lines[1:]
    if inner and FENCE_RE.match(inner[-1]):
        inner = inner[:-1]

    opening = f"{block.indent}{block.fence}{block.info}"
    closing = f"{block.indent}{block.fence[0] * len(block.fence)}"
    overhead = len(opening) + len(closing) + 2
    budget = max(80, max_chars - overhead)

    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in inner:
        addition = len(line) + 1
        if current and size + addition > budget:
            parts.append("\n".join([opening, *current, closing]))
            current = []
            size = 0
        if addition > budget and not current:
            # Una sola línea desmesurada: se parte por caracteres.
            for start in range(0, len(line), budget):
                parts.append(
                    "\n".join([opening, line[start : start + budget], closing])
                )
            continue
        current.append(line)
        size += addition
    if current:
        parts.append("\n".join([opening, *current, closing]))
    return parts or [block.text]


def _split_table_block(block: Block, max_chars: int) -> list[str]:
    """Parte la tabla por filas repitiendo la cabecera."""
    header = block.header or block.lines[:2]
    rows = block.lines[len(header):]
    header_text = "\n".join(header)
    budget = max(80, max_chars - len(header_text) - 1)

    parts: list[str] = []
    current: list[str] = []
    size = 0
    for row in rows:
        addition = len(row) + 1
        if current and size + addition > budget:
            parts.append("\n".join([header_text, *current]))
            current = []
            size = 0
        current.append(row)
        size += addition
    if current:
        parts.append("\n".join([header_text, *current]))
    return parts or [block.text]


def _split_plain(text: str, max_chars: int) -> list[str]:
    """Reparto por líneas y, en último extremo, por caracteres."""
    lines = text.split("\n")
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        addition = len(line) + 1
        if current and size + addition > max_chars:
            parts.append("\n".join(current).strip())
            current = []
            size = 0
        if addition > max_chars and not current:
            for start in range(0, len(line), max_chars):
                parts.append(line[start : start + max_chars].strip())
            continue
        current.append(line)
        size += addition
    if current:
        joined = "\n".join(current).strip()
        if joined:
            parts.append(joined)
    return [part for part in parts if part]


def split_block(block: Block, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block.text]
    if block.kind == "code":
        return _split_code_block(block, max_chars)
    if block.kind == "table":
        return _split_table_block(block, max_chars)
    return _split_plain(block.text, max_chars)


# ----------------------------------------------------------------------
# Empaquetado
# ----------------------------------------------------------------------
def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Solapamiento por líneas completas, nunca a mitad de palabra."""
    if overlap_chars <= 0 or not text:
        return ""
    lines = text.split("\n")
    tail: list[str] = []
    size = 0
    for line in reversed(lines):
        addition = len(line) + 1
        if size + addition > overlap_chars:
            break
        tail.insert(0, line)
        size += addition
    candidate = "\n".join(tail).strip()
    # Un solapamiento que abre una valla sin cerrarla envenenaría el
    # siguiente fragmento: en ese caso se prescinde de él.
    if candidate.count("```") % 2 or candidate.count("~~~") % 2:
        return ""
    return candidate


def split_structured_text(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Divide respetando la estructura del Markdown."""
    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    blocks = parse_blocks(clean)
    if not blocks:
        return _split_plain(clean, max_chars)

    pieces: list[str] = []
    for block in blocks:
        pieces.extend(split_block(block, max_chars))

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for piece in pieces:
        addition = len(piece) + 2
        if current and size + addition > max_chars:
            joined = "\n\n".join(current).strip()
            chunks.append(joined)
            tail = _overlap_tail(joined, overlap_chars)
            current = [tail] if tail else []
            size = len(tail) + 2 if tail else 0
        current.append(piece)
        size += addition

    if current:
        joined = "\n\n".join(current).strip()
        if joined and (not chunks or joined != chunks[-1]):
            chunks.append(joined)

    return chunks


def iter_unbalanced(chunks: list[str]) -> Iterator[str]:
    """Diagnóstico: fragmentos con vallas de código sin cerrar."""
    for chunk in chunks:
        if chunk.count("```") % 2 or chunk.count("~~~") % 2:
            yield chunk

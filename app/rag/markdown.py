from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.rag.chunking import split_structured_text
from app.rag.document_meta import build_enrichment, compute_vigencia
from app.rag.obsidian_syntax import (
    collect_tags,
    extract_block_anchors,
    extract_callouts,
    has_deprecation_callout,
    searchable_supplement,
)


FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*(?:\n|$)",
    flags=re.DOTALL,
)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", flags=re.MULTILINE)
WIKI_LINK_RE = re.compile(r"(!?)\[\[([^\]]+)\]\]")
CODE_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"(`+)(.*?)\1")


@dataclass
class MarkdownChunk:
    index: int
    heading: str
    content: str
    parent_index: int | None = None


@dataclass
class MarkdownParent:
    index: int
    heading: str
    content: str


@dataclass
class MarkdownDocument:
    path: Path
    title: str
    metadata: dict[str, Any]
    links: list[dict[str, Any]]
    chunks: list[MarkdownChunk]
    sha256: str
    mtime: float
    parents: list[MarkdownParent] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    supplement: str = ""


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    raw = match.group(1)
    try:
        parsed = yaml.safe_load(raw) or {}
        metadata = parsed if isinstance(parsed, dict) else {}
    except yaml.YAMLError:
        metadata = {"_frontmatter_error": "No se pudo interpretar el YAML"}

    body = text[match.end():]
    return metadata, body


def extract_title(
    metadata: dict[str, Any],
    body: str,
    fallback: str,
) -> str:
    for key in ("titulo", "title", "nombre"):
        value = metadata.get(key)
        if value:
            return str(value).strip()

    match = re.search(r"^#\s+(.+?)\s*$", body, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()

    return fallback


def _strip_markdown_code(body: str) -> str:
    """Oculta código cercado y en línea antes de buscar wikienlaces."""
    output: list[str] = []
    inside_fence = False
    fence_marker = ""
    for line in body.split("\n"):
        fence = CODE_FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if not inside_fence:
                inside_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                inside_fence = False
            output.append("")
            continue
        output.append("" if inside_fence else INLINE_CODE_RE.sub("", line))
    return "\n".join(output)


def extract_wiki_links(body: str) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, bool]] = set()

    for match in WIKI_LINK_RE.finditer(_strip_markdown_code(body)):
        embedded = bool(match.group(1))
        # Dentro de una tabla Markdown, Obsidian escribe el separador del
        # alias como ``\|`` para que no cierre la celda. Sigue siendo el
        # mismo wikienlace ``[[Nota|alias]]`` y no parte del destino.
        raw = match.group(2).strip().replace("\\|", "|")

        target_part, alias = (
            raw.split("|", 1) if "|" in raw else (raw, "")
        )
        target_part = target_part.strip()
        alias = alias.strip()

        target, section = (
            target_part.split("#", 1)
            if "#" in target_part
            else (target_part, "")
        )
        target = target.strip()
        section = section.strip()

        if not target:
            continue

        key = (target, section, alias, embedded)
        if key in seen:
            continue
        seen.add(key)

        links.append(
            {
                "target": target,
                "section": section,
                "alias": alias,
                "embedded": embedded,
            }
        )

    return links


def split_sections(body: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(body))
    if not matches:
        return [("Documento", body.strip())] if body.strip() else []

    sections: list[tuple[str, str]] = []
    intro = body[: matches[0].start()].strip()
    if intro:
        sections.append(("Introducción", intro))

    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(body)
        )
        content = body[start:end].strip()
        if content:
            sections.append((heading, content))

    return sections


def parse_markdown_file(
    path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    parent_child_enabled: bool = False,
    parent_chunk_size: int = 6000,
    child_chunk_size: int = 700,
    child_chunk_overlap: int = 100,
) -> MarkdownDocument:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    metadata, body = parse_frontmatter(text)
    title = extract_title(metadata, body, path.stem)
    links = extract_wiki_links(body)

    chunks: list[MarkdownChunk] = []
    parents: list[MarkdownParent] = []
    index = 0
    parent_index = 0
    for heading, section_content in split_sections(body):
        if parent_child_enabled:
            parent_parts = split_structured_text(
                section_content,
                max_chars=parent_chunk_size,
                overlap_chars=0,
            )
            for parent_content in parent_parts:
                parents.append(
                    MarkdownParent(
                        index=parent_index,
                        heading=heading,
                        content=parent_content,
                    )
                )
                for part in split_structured_text(
                    parent_content,
                    max_chars=child_chunk_size,
                    overlap_chars=child_chunk_overlap,
                ):
                    chunks.append(
                        MarkdownChunk(
                            index=index,
                            heading=heading,
                            content=part,
                            parent_index=parent_index,
                        )
                    )
                    index += 1
                parent_index += 1
        else:
            for part in split_structured_text(
                section_content,
                max_chars=chunk_size,
                overlap_chars=chunk_overlap,
            ):
                chunks.append(
                    MarkdownChunk(
                        index=index,
                        heading=heading,
                        content=part,
                    )
                )
                index += 1

    if not chunks and body.strip():
        fallback_parent = 0 if parent_child_enabled else None
        if parent_child_enabled:
            parents.append(
                MarkdownParent(
                    index=0,
                    heading="Documento",
                    content=body.strip(),
                )
            )
        chunks.append(
            MarkdownChunk(
                index=0,
                heading="Documento",
                content=body.strip(),
                parent_index=fallback_parent,
            )
        )

    tags = collect_tags(metadata, body)
    callouts = extract_callouts(body)
    anchors = extract_block_anchors(body)
    vigencia = compute_vigencia(
        metadata,
        deprecated_callout=has_deprecation_callout(callouts),
    )
    enriched = build_enrichment(
        metadata,
        tags=tags,
        vigencia=vigencia,
        anchors=anchors,
        callouts=[
            {
                "tipo": callout.tipo,
                "titulo": callout.titulo,
                "contenido": callout.contenido,
            }
            for callout in callouts
        ],
    )

    return MarkdownDocument(
        path=path,
        title=title,
        metadata=enriched,
        links=links,
        chunks=chunks,
        sha256=calculate_sha256(path),
        mtime=path.stat().st_mtime,
        parents=parents,
        tags=tags,
        supplement=searchable_supplement(tags, callouts),
    )

from __future__ import annotations

from app.rag.indexer import discover_markdown_files


def test_discovery_excludes_obsidian_and_template_directories(tmp_path):
    visible = tmp_path / "notas" / "Visible.md"
    hidden = tmp_path / ".privado" / "Oculta.md"
    obsidian = tmp_path / ".obsidian" / "Configuracion.md"
    spanish_template = tmp_path / "_plantillas" / "Plantilla.md"
    english_template = tmp_path / "_templates" / "Template.md"

    for path in (
        visible,
        hidden,
        obsidian,
        spanish_template,
        english_template,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Nota", encoding="utf-8")

    discovered = discover_markdown_files(tmp_path)

    assert discovered == [visible]

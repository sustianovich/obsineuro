from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "app" / "templates" / "index.html").read_text(
    encoding="utf-8"
)
JS = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "app" / "static" / "styles.css").read_text(
    encoding="utf-8"
)


def test_referencias_dom_existen_en_la_plantilla():
    ids = set(re.findall(r'id="([^"]+)"', HTML))
    references = set(
        re.findall(r"""getElementById\(\s*["']([^"']+)["']""", JS)
    )
    assert references <= ids


def test_variables_css_usadas_estan_definidas():
    defined = set(re.findall(r"^\s*(--[-\w]+)\s*:", CSS, re.MULTILINE))
    used = set(re.findall(r"var\((--[-\w]+)", CSS))
    assert used <= defined


def test_ui_expone_estado_y_navegacion_movil():
    assert 'id="document-count"' in HTML
    assert 'id="chunk-count"' in HTML
    assert 'id="stop-query-button"' in HTML
    assert 'id="ask-button"' in HTML
    assert 'id="theme-toggle"' in HTML
    assert 'id="delete-project-button"' in HTML
    assert 'class="mobile-tabbar"' in HTML
    assert 'data-mobile-tab="chat"' in HTML
    assert 'data-mobile-panel="status"' in HTML
    assert 'data-mobile-panel="filters"' in HTML


def test_ui_expone_selector_compacto_de_etiquetas():
    assert 'id="tag-filter-control"' in HTML
    assert 'id="tag-filter-search"' in HTML
    assert 'id="tag-filter-clear"' in HTML
    assert "tag-filter-option-count" in JS
    assert "function filterTagOptions()" in JS
    assert "function clearTagFilter()" in JS


def test_ui_expone_copia_de_respuestas():
    assert "function createCopyAnswerButton" in JS
    assert "copyTextToClipboard" in JS
    assert "copy-answer-button" in JS
    assert ".copy-answer-button" in CSS

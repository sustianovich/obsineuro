from __future__ import annotations

from datetime import date

import pytest

from app.rag.chunking import (
    iter_unbalanced,
    parse_blocks,
    split_structured_text,
)
from app.rag.citations import analyse_citations, validate_answer
from app.rag.document_meta import compute_vigencia, parse_date
from app.rag.obsidian_syntax import (
    collect_tags,
    extract_block_anchors,
    extract_callouts,
    extract_inline_tags,
    has_deprecation_callout,
    normalize_tag,
)


# ----------------------------------------------------------------------
# Troceado estructurado
# ----------------------------------------------------------------------
CODIGO = "```python\n" + "linea = 1\n" * 60 + "```"
TABLA = (
    "| Codigo | Indicador | Objetivo |\n"
    "| --- | --- | --- |\n"
    + "".join(f"| IP-{n:02} | Indicador {n} | {n} % |\n" for n in range(30))
)


def test_reconoce_los_tipos_de_bloque():
    texto = f"Parrafo inicial.\n\n{CODIGO}\n\n{TABLA}\n\n> [!note] Aviso\n> Cuerpo."
    tipos = [block.kind for block in parse_blocks(texto)]
    assert "code" in tipos
    assert "table" in tipos
    assert "callout" in tipos
    assert "paragraph" in tipos


def test_no_deja_vallas_sin_cerrar():
    for max_chars in (150, 300, 600, 1200):
        chunks = split_structured_text(
            f"Introduccion.\n\n{CODIGO}\n\nCierre.",
            max_chars=max_chars,
            overlap_chars=40,
        )
        assert not list(iter_unbalanced(chunks)), (
            f"valla rota con max_chars={max_chars}"
        )


def test_conserva_el_lenguaje_al_partir_codigo():
    chunks = split_structured_text(
        CODIGO, max_chars=200, overlap_chars=0
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith("```python")
        assert chunk.rstrip().endswith("```")


def test_repite_la_cabecera_al_partir_tablas():
    chunks = split_structured_text(TABLA, max_chars=250, overlap_chars=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "| Codigo | Indicador | Objetivo |" in chunk
        assert "| --- |" in chunk


def test_el_solapamiento_no_arrastra_media_valla():
    chunks = split_structured_text(
        f"Texto previo largo.\n\n{CODIGO}\n\nTexto posterior.",
        max_chars=300,
        overlap_chars=200,
    )
    assert not list(iter_unbalanced(chunks))


def test_texto_corto_no_se_parte():
    assert split_structured_text(
        "Una frase corta.", max_chars=500, overlap_chars=50
    ) == ["Una frase corta."]


def test_texto_vacio_devuelve_lista_vacia():
    assert split_structured_text("   \n  ", max_chars=100, overlap_chars=0) == []


# ----------------------------------------------------------------------
# Sintaxis de Obsidian
# ----------------------------------------------------------------------
def test_etiquetas_del_cuerpo_y_del_frontmatter():
    metadata = {"tags": ["Protocolo", "SICC-2025"]}
    body = "Contenido con #cribado y #trazabilidad."
    assert collect_tags(metadata, body) == [
        "protocolo",
        "sicc-2025",
        "cribado",
        "trazabilidad",
    ]


def test_ignora_almohadillas_dentro_de_codigo():
    body = (
        "Texto con #real.\n\n"
        "```bash\n# comentario, no etiqueta\necho '#tampoco'\n```\n\n"
        "Final con #tambien."
    )
    assert extract_inline_tags(body) == ["real", "tambien"]


def test_normaliza_acentos_y_mayusculas():
    assert normalize_tag("Validación") == "validacion"
    assert normalize_tag("#SICC-2025") == "sicc-2025"


def test_detecta_callout_de_derogacion():
    body = "> [!caution] Documento derogado\n> Sustituido por la version 3."
    callouts = extract_callouts(body)
    assert callouts[0].tipo == "caution"
    assert callouts[0].titulo == "Documento derogado"
    assert has_deprecation_callout(callouts)


def test_callout_informativo_no_marca_derogacion():
    callouts = extract_callouts("> [!note] Aviso\n> Sin importancia.")
    assert not has_deprecation_callout(callouts)


def test_extrae_anclas_de_bloque():
    body = "El intervalo es de 24 meses. ^poblacion-diana\nOtra linea."
    anclas = extract_block_anchors(body)
    assert "poblacion-diana" in anclas
    assert anclas["poblacion-diana"].endswith("24 meses.")


def test_frontmatter_tags_como_cadena():
    assert collect_tags({"tags": "uno, dos"}, "") == ["uno", "dos"]


# ----------------------------------------------------------------------
# Vigencia temporal
# ----------------------------------------------------------------------
HOY = date(2026, 7, 24)


@pytest.mark.parametrize(
    "valor, esperado",
    [
        ("2025-01-01", date(2025, 1, 1)),
        ("2025/01/01", date(2025, 1, 1)),
        (date(2024, 3, 2), date(2024, 3, 2)),
        ("", None),
        ("no es fecha", None),
        (None, None),
    ],
)
def test_lectura_de_fechas(valor, esperado):
    assert parse_date(valor) == esperado


def test_documento_vigente():
    v = compute_vigencia({"fecha_vigencia": "2025-01-01"}, today=HOY)
    assert v.estado_temporal == "vigente"


def test_documento_caducado_por_fecha():
    v = compute_vigencia(
        {"fecha_vigencia": "2021-01-01", "fecha_derogacion": "2025-01-01"},
        today=HOY,
    )
    assert v.estado_temporal == "caducada"
    assert v.valid_until == "2025-01-01"


def test_documento_con_vigencia_futura():
    v = compute_vigencia({"fecha_vigencia": "2026-09-01"}, today=HOY)
    assert v.estado_temporal == "futura"


def test_sin_fechas_queda_desconocida():
    assert compute_vigencia({}, today=HOY).estado_temporal == "desconocida"


def test_callout_de_derogacion_manda_sobre_la_ausencia_de_fecha():
    v = compute_vigencia(
        {"fecha_vigencia": "2020-01-01"},
        deprecated_callout=True,
        today=HOY,
    )
    assert v.estado_temporal == "caducada"


def test_callout_no_adelanta_una_vigencia_futura():
    # Un borrador con aviso no es una norma ya derogada.
    v = compute_vigencia(
        {"fecha_vigencia": "2027-01-01"},
        deprecated_callout=True,
        today=HOY,
    )
    assert v.estado_temporal == "futura"


def test_revision_vencida():
    v = compute_vigencia(
        {"fecha_vigencia": "2020-01-01", "fecha_revision": "2024-01-01"},
        today=HOY,
    )
    assert v.revision_vencida


# ----------------------------------------------------------------------
# Validación de citas
# ----------------------------------------------------------------------
def test_citas_validas_no_se_tocan():
    texto = "Segun [1] y [2], el plazo es de quince dias."
    limpio, informe, aviso = validate_answer(texto, 3)
    assert limpio == texto
    assert aviso is None
    assert not informe.has_invalid


def test_marca_las_citas_inexistentes():
    limpio, informe, aviso = validate_answer("Dice [1] y tambien [9].", 3)
    assert "[?]" in limpio
    assert "[1]" in limpio
    assert informe.invalid_references == (9,)
    assert aviso and "[9]" in aviso


def test_detecta_respuesta_sin_citar_nada():
    informe = analyse_citations("Respuesta sin referencias.", 4)
    assert informe.uncited_answer
    assert informe.unused_sources == (1, 2, 3, 4)


def test_sin_fuentes_no_hay_cita_huerfana():
    informe = analyse_citations("Texto.", 0)
    assert not informe.uncited_answer


def test_informa_de_fuentes_no_utilizadas():
    informe = analyse_citations("Solo uso [2].", 3)
    assert informe.unused_sources == (1, 3)

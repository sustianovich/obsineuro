"""Genera el vault de evaluación del programa (dominio PDPCM).

Contenido deliberadamente organizativo, no clínico: circuitos, plazos,
responsables e indicadores. Incluye a propósito toda la sintaxis que el
troceador y el extractor deben soportar: vallas de código, tablas,
callouts, etiquetas, enlaces con sección, referencias de bloque y
documentos derogados con fechas.
"""

from __future__ import annotations

from pathlib import Path

VAULT = Path(__file__).resolve().parent.parent / "vault_demo"

DOCUMENTS: dict[str, str] = {}

DOCUMENTS["protocolos/protocolo-cribado.md"] = """---
titulo: Protocolo de cribado poblacional
estado: vigente
fecha_vigencia: 2025-01-01
fecha_revision: 2027-01-01
version: "3.2"
tags: [protocolo, cribado, sicc-2025]
responsable: Coordinacion del Programa
---

# Protocolo de cribado poblacional

> [!important] Ambito
> Este documento describe el circuito organizativo del programa. No
> sustituye al criterio clinico ni constituye indicacion individual.

## Poblacion diana
La invitacion se dirige a mujeres de 50 a 69 anos empadronadas en el area
de salud. El intervalo entre rondas es de veinticuatro meses. ^poblacion-diana

## Fases del circuito
El circuito consta de cuatro fases: invitacion, realizacion de la prueba,
lectura y comunicacion de resultados. La fase de lectura se detalla en
[[Circuito de lectura doble]].

| Fase | Responsable | Plazo maximo |
| --- | --- | --- |
| Invitacion | Unidad administrativa | 30 dias |
| Realizacion | Unidad de radiologia | 45 dias |
| Lectura | Radiologia (doble ciego) | 10 dias |
| Comunicacion | Coordinacion | 15 dias |

## Criterios de exclusion
Quedan excluidas las participantes con seguimiento activo en consulta
especializada y aquellas que hayan solicitado su baja voluntaria del
programa. Ver [[Gestion de exclusiones]].

## Codificacion de resultados
La codificacion sigue el estandar interno del programa:

```sql
-- Consulta de resultados pendientes de comunicacion
SELECT p.nia,
       p.fecha_prueba,
       r.categoria,
       r.fecha_lectura
FROM participantes p
JOIN resultados r ON r.nia = p.nia
WHERE r.categoria IN ('R3', 'R4', 'R5')
  AND r.fecha_comunicacion IS NULL
  AND r.fecha_lectura < DATE('now', '-15 days')
ORDER BY r.fecha_lectura ASC;
```

Las categorias R3, R4 y R5 activan el circuito descrito en
[[Circuito de valoracion adicional]]. #cribado #trazabilidad
"""

DOCUMENTS["circuitos/lectura-doble.md"] = """---
titulo: Circuito de lectura doble
estado: vigente
fecha_vigencia: 2025-03-15
tags: [circuito, lectura, calidad]
responsable: Jefatura de Radiologia
---

# Circuito de lectura doble

## Principio
Toda prueba se somete a dos lecturas independientes y ciegas. Ningun
lector conoce el resultado del otro hasta que ambos han registrado su
categoria.

## Resolucion de discordancias
Cuando las dos lecturas difieren en categoria, interviene un tercer
lector con funcion dirimente. El plazo para la tercera lectura es de
cinco dias habiles. ^discordancia

> [!warning] Trazabilidad
> Las tres lecturas quedan registradas con identificador de lector y
> marca temporal. No se admite sobrescritura.

## Indicadores asociados
El rendimiento del circuito se mide con los indicadores recogidos en
[[Indicadores de proceso]]. #calidad
"""

DOCUMENTS["circuitos/valoracion-adicional.md"] = """---
titulo: Circuito de valoracion adicional
estado: vigente
fecha_vigencia: 2025-01-01
tags: [circuito, valoracion, sicc-2025]
---

# Circuito de valoracion adicional

## Activacion
Se activa ante una categoria R3, R4 o R5 confirmada tras la lectura doble
descrita en [[Circuito de lectura doble#Resolucion de discordancias]].

## Plazos
La citacion para valoracion adicional debe producirse en un plazo maximo
de quince dias naturales desde la confirmacion de la categoria. La
participante recibe comunicacion telefonica y por escrito. ^plazo-citacion

## Contenido de la comunicacion
La comunicacion indica el motivo de la citacion, el lugar y la fecha, y un
telefono de contacto del programa. No se comunican resultados por telefono
sin identificacion previa.
"""

DOCUMENTS["circuitos/gestion-exclusiones.md"] = """---
titulo: Gestion de exclusiones
estado: vigente
fecha_vigencia: 2025-01-01
tags: [circuito, exclusiones]
---

# Gestion de exclusiones

## Tipos de exclusion
Se distinguen tres tipos: exclusion temporal, exclusion definitiva y baja
voluntaria. Cada tipo tiene un procedimiento de registro distinto.

## Exclusion temporal
Se aplica durante un episodio de seguimiento activo. Se revisa en cada
ronda y caduca automaticamente a los veinticuatro meses.

## Baja voluntaria
Requiere solicitud firmada por la participante. Es reversible en cualquier
momento mediante nueva solicitud.
"""

DOCUMENTS["indicadores/indicadores-proceso.md"] = """---
titulo: Indicadores de proceso
estado: vigente
fecha_vigencia: 2025-01-01
tags: [indicadores, calidad, sicc-2025]
---

# Indicadores de proceso

## Tabla de indicadores

| Codigo | Indicador | Objetivo | Fuente |
| --- | --- | --- | --- |
| IP-01 | Cobertura de invitacion | >= 95 % | Padron |
| IP-02 | Tasa de participacion | >= 70 % | Sistema |
| IP-03 | Tasa de recitacion | < 5 % | Radiologia |
| IP-04 | Demora media de lectura | <= 10 dias | Sistema |
| IP-05 | Discordancia entre lectores | 5-15 % | Radiologia |

## Calculo de la tasa de participacion
La tasa de participacion se calcula sobre invitaciones efectivamente
entregadas, no sobre invitaciones emitidas:

```python
def tasa_participacion(pruebas_realizadas, invitaciones_entregadas):
    \"\"\"Devuelve la tasa en tanto por ciento.\"\"\"
    if invitaciones_entregadas == 0:
        return 0.0
    return pruebas_realizadas / invitaciones_entregadas * 100
```

## Periodicidad
Los indicadores se calculan con periodicidad trimestral y se consolidan
anualmente en la memoria del programa. #indicadores
"""

DOCUMENTS["protocolos/protocolo-cribado-v2.md"] = """---
titulo: Protocolo de cribado poblacional (v2)
estado: derogado
fecha_vigencia: 2021-01-01
fecha_derogacion: 2025-01-01
version: "2.4"
tags: [protocolo, historico]
sustituido_por: Protocolo de cribado poblacional
---

# Protocolo de cribado poblacional (v2)

> [!caution] Documento derogado
> Sustituido por la version 3.2 con efecto 1 de enero de 2025.

## Poblacion diana
La invitacion se dirigia a mujeres de 45 a 69 anos. El intervalo entre
rondas era de dieciocho meses.

## Fases del circuito
El circuito constaba de tres fases: invitacion, realizacion y
comunicacion. La lectura no era doble de forma sistematica.
"""

DOCUMENTS["referencias/normativa-sicc.md"] = """---
titulo: Referencia normativa SICC 2025
estado: vigente
fecha_vigencia: 2025-01-01
tags: [normativa, sicc-2025]
---

# Referencia normativa SICC 2025

## Ambito
El marco SICC 2025 establece los requisitos minimos de trazabilidad,
codificacion e indicadores de los programas de cribado poblacional.

## Requisitos de trazabilidad
Toda actuacion registrada debe conservar identificador de actuante, marca
temporal y motivo. Los registros son inmutables una vez cerrados.

## Correspondencia con el programa
La correspondencia entre los indicadores del programa y los exigidos por
SICC 2025 figura en [[Indicadores de proceso]].
"""

DOCUMENTS["referencias/borrador-teleradiologia.md"] = """---
titulo: Borrador de teleradiologia
estado: borrador
fecha_vigencia: 2026-09-01
tags: [borrador, teleradiologia]
---

# Borrador de teleradiologia

## Propuesta
Se propone habilitar la segunda lectura en modalidad remota para centros
sin dotacion de radiologia en jornada completa.

> [!note] Estado
> Documento en discusion. No aplicable hasta su aprobacion.

## Requisitos tecnicos pendientes
Latencia maxima, calibracion de monitores y politica de conservacion de
imagenes estan pendientes de definir.
"""

DOCUMENTS["circuitos/comunicacion-resultados.md"] = """---
titulo: Comunicacion de resultados
estado: vigente
fecha_vigencia: 2025-01-01
tags: [circuito, comunicacion]
---

# Comunicacion de resultados

## Resultado normal
Se comunica por carta en un plazo maximo de quince dias naturales desde la
lectura. La carta incluye la fecha prevista de la siguiente ronda.

## Resultado que requiere valoracion adicional
Se comunica por via telefonica y se confirma por escrito, siguiendo el
[[Circuito de valoracion adicional]]. Nunca se comunica el detalle por
correo electronico.

## Registro
Toda comunicacion queda registrada con fecha, via y persona que la
realiza, conforme a [[Referencia normativa SICC 2025]].
"""


def main() -> None:
    for relative, content in DOCUMENTS.items():
        path = VAULT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"{len(DOCUMENTS)} documentos escritos en {VAULT}")


if __name__ == "__main__":
    main()

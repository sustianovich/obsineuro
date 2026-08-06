# Comparación de recuperación con grafo

Fecha: 2026-07-29.

## Condiciones

- Mismo índice vectorial y mismo conjunto `evaluations/questions.json`.
- Vault de evaluación: `vault_demo` (9 notas, 28 fragmentos).
- Grafo apagado y encendido en procesos separados, sin reconstruir entre
  ambas ejecuciones.
- Parámetros del grafo: peso 0,5; 2 saltos; decaimiento 0,5; peso de backlink
  0,7; 4 documentos semilla; 20 candidatos máximos.
- Las preguntas relacionales son `E01` y `E02`; las otras 21 preguntas
  contestables se agrupan como factuales. Las 3 preguntas de abstención se
  informan aparte.

## Resultados

| Grupo | Grafo | Acierto exacto | Recall documental | MRR |
|---|---:|---:|---:|---:|
| Factuales (21) | apagado | 90,5 % | 100,0 % | 1,000 |
| Factuales (21) | encendido | 90,5 % | 100,0 % | 0,968 |
| Relacionales (2) | apagado | 100,0 % | 100,0 % | 1,000 |
| Relacionales (2) | encendido | 50,0 % | 75,0 % | 1,000 |
| Contestables (23) | apagado | 91,3 % | 100,0 % | 1,000 |
| Contestables (23) | encendido | 87,0 % | 97,8 % | 0,971 |

La precisión de abstención fue 0 % en ambas ejecuciones. El grafo no modificó
esa salvaguarda; el umbral semántico vigente tampoco abstuvo en los tres casos
fuera de dominio.

En las preguntas factuales el acierto agregado quedó igual, aunque el grafo
intercambió un fallo y un acierto (`D01` empeoró y `D02` mejoró) y redujo el
MRR por el descenso de `R04` del rango 1 al 3. En las relacionales, `E02`
conservó todos sus documentos y `E01` perdió `indicadores/indicadores-proceso.md`
del `top_k`.

## Interpretación

El vault de evaluación sólo tiene 0,89 enlaces resueltos por nota (mediana 1,
máximo 3), por debajo del umbral orientativo de 2. En esta densidad el grafo
aporta poca señal. Además, con el grafo apagado `expand_links` añadía el destino
de `E01` después del `top_k`; con el grafo encendido ese mecanismo se desactiva
y el candidato estructural no alcanzó los seis puestos fusionados con peso 0,5.
El resultado confirma que no conviene activar el grafo por defecto en este
vault.

## Densidad del vault configurado

El vault activo `INTERCONSULTA_PRIMARIA` tiene 16 notas, 73 enlaces resueltos y
3 rotos:

- media: 4,56 enlaces resueltos por nota;
- mediana: 3;
- máximo: 14 (`00 Indice.md`).

La densidad media sí permite que el grafo aporte señal, pero el índice central
roza el umbral de alta conectividad y el vault es pequeño. La configuración
inicial prudente es `RAG_GRAPH_MAX_HOPS=1`; sólo conviene probar 2 saltos con un
conjunto dorado propio que incluya preguntas relacionales del vault activo.

Los informes completos están en `evaluations/reports/graph_off/latest.json` y
`evaluations/reports/graph_on/latest.json`.

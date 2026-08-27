# Evaluación de recuperación del vault hospitalario

Fecha: 2026-08-10.

## Condiciones

- Dataset: `evaluations/hospital_questions.json`.
- 21 casos: 12 factuales, 5 relacionales, 2 híbridos y 2 de abstención.
- Índice: 18 documentos, 194 fragmentos y troceado plano 1800/250.
- Las carpetas `_plantillas` y `_templates` están excluidas del índice.
- Grafo saneado: 118 enlaces resueltos, 0 rotos y 0 alias ambiguos.
- Embeddings: `nomic-embed-text`; `top_k=6`; similitud mínima 0,30.
- No se generaron respuestas: primero se aisló la calidad de recuperación.

`graph_off` desactiva la propagación ponderada por grafo, pero conserva la
expansión directa de wikilinks solicitada por cada caso. Por tanto, esta
comparación mide el buscador de grafo frente al comportamiento actual de
expansión directa; no es una comparación contra un sistema sin enlaces.

## Resultados

| Estrategia | Acierto exacto | Recall documental | Precision@k | nDCG@k | MRR |
|---|---:|---:|---:|---:|---:|
| Grafo apagado | 89,5 % | 93,0 % | 42,2 % | 90,1 % | 0,921 |
| Grafo siempre activo, 1 salto | 89,5 % | 98,2 % | 36,8 % | 75,2 % | 0,655 |
| Router oráculo | 89,5 % | 93,0 % | 38,6 % | 82,5 % | 0,781 |
| Router real | 89,5 % | 96,9 % | 38,7 % | 86,2 % | 0,844 |

Con el grafo apagado, las preguntas relacionales obtienen 80 % de acierto
exacto, 93,3 % de recall y MRR 1,000. `HOSP-R05` no recupera Anatomía
patológica. En las factuales, `HOSP-F04` no recupera Laboratorio clínico y
`HOSP-F10` coloca el primer resultado relevante en rango 2.

El grafo a un salto aumenta el recall agregado, pero reduce mucho la calidad
del orden: el MRR baja de 0,921 a 0,655 y nDCG de 90,1 % a 75,2 %. El router
real conserva más recall que la línea base, pero sólo acierta el tipo de 7 de
21 consultas y también empeora MRR, nDCG y precisión.

## Robustez ortográfica

El dataset separado `hospital_typos_questions.json` contiene ocho consultas
con omisiones, transposiciones y errores fonéticos deliberados. Con la
configuración normal obtiene 100 % de acierto exacto, 100 % de recall, nDCG
90,8 % y MRR 0,875. Seis documentos correctos quedan en rango 1 y dos en
rango 2.

No se añadió autocorrección ni búsqueda difusa: no mejoraría el recall actual
y podría modificar siglas o terminología clínica válida. Forzar el router en
estas consultas reduce el recall a 87,5 % y el MRR a 0,488. Excluir las
plantillas sí fue una mejora segura y elevó el MRR de errores ortográficos de
0,854 a 0,875.

## Abstención

Los dos casos fuera de dominio recuperan fragmentos, por lo que el recall de
abstención es 0 %. La calibración por similitud propone provisionalmente 0,680
frente al 0,30 actual, pero las distribuciones se solapan: el cambio también
rechazaría una pregunta válida sobre errores preanalíticos. Con sólo dos
negativos no se debe convertir ese valor en configuración.

## Decisión

- Mantener desactivados el grafo ponderado y el router: el aumento de recall
  no compensa la pérdida de orden y precisión.
- Conservar la recuperación semántica como tolerancia principal ante errores
  ortográficos; no introducir autocorrección clínica sin fallos demostrados.
- Ampliar primero los negativos cercanos al dominio antes de calibrar la
  abstención posterior.
- Investigar la pérdida de documentos en preguntas multisección y la
  duplicación de fragmentos antes de ajustar pesos del grafo.
- Ejecutar `--generate-answers` cuando la recuperación y la abstención superen
  sus puertas de calidad; hacerlo ahora mezclaría fallos de recuperación con
  fallos del modelo generador.

Los detalles por caso están en `baseline/latest.json` y
`graph-on/latest.json`; la comparación agregada está en
`strategy_comparison.json`.

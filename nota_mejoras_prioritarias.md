# Nota: 3 puntos a mejorar ahora

Fecha: 2026-08-27.

## Visión general

El proyecto está en buen estado: el README documenta con honestidad sus
propios límites, cada decisión de diseño (grafo, router, reranking,
abstención posterior) está respaldada por una evaluación con números
concretos en vez de intuición, y hay disciplina de calibración (umbrales
mínimos de recall/MRR antes de aceptar un resultado). Eso es más raro de lo
que parece en proyectos de un solo desarrollador.

Lo que sigue no son problemas de arquitectura, sino tres frentes concretos
que ya están identificados en el propio repo pero no cerrados.

## 1. Mantener CI independiente de datasets retirados

Los conjuntos retirados no deben seguir siendo dependencias implícitas de
pruebas, scripts o documentación. CI usa `evaluations/ci_questions.json`, un
conjunto sintético ligado al vault generado por `scripts/build_demo_vault.py`.
Las evaluaciones adicionales se pasan explícitamente con `--dataset` y pueden
mantenerse fuera del repositorio cuando contienen material de un dominio real.

## 2. Investigar *por qué* el grafo y el router bajan el ranking, no volver a medirlo

Dos evaluaciones independientes —un conjunto anterior (16 preguntas) y ahora el vault
hospitalario (21 preguntas)— llegan a la misma conclusión cuantitativa:
activar el grafo mejora el recall pero hunde el orden de los resultados.
En el conjunto anterior el MRR cae de 0,938 a 0,474 con el grafo siempre activo; en el
hospital, de 0,921 a 0,655 y el nDCG de 90,1% a 75,2%. El router real
tampoco compensa (acierta el tipo de consulta en sólo 7 de 21 casos).

La decisión correcta hoy es mantenerlos apagados, y así está configurado.
El diagnóstico histórico señala el siguiente paso sin ejecutarlo todavía:
*"Investigar la pérdida de documentos
en preguntas multisección y la duplicación de fragmentos antes de ajustar
pesos del grafo."* Con el mismo patrón repitiéndose en dos vaults distintos,
vale más diagnosticar el mecanismo (fragmentos duplicados compitiendo en
MMR, selección de semillas, peso de backlinks) que seguir acumulando
comparaciones con la config ya descartada.

## 3. Los negativos de abstención no alcanzan masa crítica para calibrar nada

El vault hospitalario sólo tiene 2 preguntas fuera de dominio, y el propio
resumen lo marca explícitamente: *"Con sólo dos negativos no se debe
convertir ese valor en configuración."* El estudio de infraestructura de
abstención posterior (`study_abstention_Infrastructure.md`) plantea la misma
exigencia para cualquier conjunto de validación. `scripts/calibrate_posterior_threshold.py` ya impone
`--min-class-size 10` como puerta de calidad, así que ejecutarlo sobre
cualquiera de estos datasets no puede pasar sus propias garantías.

Antes de tocar de nuevo `RAG_POSTERIOR_ABSTENTION_THRESHOLD`, el trabajo que
falta es construir negativos reales por vault (con revisión del responsable
de contenido en el caso clínico) hasta llegar a ese mínimo, en vez de seguir
proponiendo umbrales provisionales que las propias herramientas del proyecto
consideran insuficientes.

---

Fuera de estos tres, `study_security_implementation.md` deja documentado un
cuarto frente (exponer el RAG a una LAN) que ya tiene diseño y prompt de
retomo listos, pero no me parece urgente mientras el uso siga siendo de un
solo equipo.

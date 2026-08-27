# Evaluación automática del RAG

- Estrategia: graph_on_1hop
- Troceado: plano (1800/250)
- Casos: 21
- Acierto de recuperación: 78.9%
- Recall documental: 90.4%
- Precision@k: 30.4%
- nDCG@k: 70.1%
- MRR: 0.639
- Cobertura del sistema (no abstuvo): 100.0%
- Precisión de abstención: N/A
- Recall de abstención: 0.0%
- Cobertura de términos en respuestas: N/A

| Caso | Tipo | Recuperación | Recall | Precision@k | Primer acierto |
|---|---|---:|---:|---:|---:|
| HOSP-F01-triaje-urgencias | factual | [OK] | 100.0% | 22.2% | 0.333 |
| HOSP-F02-asignacion-cama | factual | [OK] | 100.0% | 22.2% | 0.500 |
| HOSP-F03-soporte-uci | factual | [OK] | 100.0% | 33.3% | 1.000 |
| HOSP-F04-error-preanalitico | factual | [OK] | 100.0% | 11.1% | 1.000 |
| HOSP-F05-valores-criticos | factual | [OK] | 100.0% | 11.1% | 1.000 |
| HOSP-F06-imagen-sin-radiacion | factual | [OK] | 100.0% | 22.2% | 0.200 |
| HOSP-F07-preparacion-colonoscopia | factual | [OK] | 100.0% | 11.1% | 1.000 |
| HOSP-F08-biopsia-intraoperatoria | factual | [OK] | 100.0% | 11.1% | 1.000 |
| HOSP-F09-validacion-farmaceutica | factual | [OK] | 100.0% | 33.3% | 0.333 |
| HOSP-F10-codificacion-altas | factual | [OK] | 100.0% | 22.2% | 1.000 |
| HOSP-F11-interoperabilidad | factual | [OK] | 100.0% | 100.0% | 1.000 |
| HOSP-F12-sesion-hospital-dia | factual | [OK] | 100.0% | 33.3% | 0.333 |
| HOSP-R01-muestra-endoscopia-diagnostico | relational | [FAIL] | 50.0% | 22.2% | 0.200 |
| HOSP-R02-urgencias-uci-planta | relational | [FAIL] | 66.7% | 44.4% | 0.500 |
| HOSP-R03-tratamiento-hospital-dia | relational | [FAIL] | 50.0% | 44.4% | 0.333 |
| HOSP-R04-admision-sistemas-indicadores | relational | [FAIL] | 50.0% | 11.1% | 0.167 |
| HOSP-R05-cribado-confirmacion-diagnostico | relational | [OK] | 100.0% | 44.4% | 1.000 |
| HOSP-H01-postoperatorio-complejo | hybrid | [OK] | 100.0% | 33.3% | 0.250 |
| HOSP-H02-peticion-analitica-no-llega | hybrid | [OK] | 100.0% | 44.4% | 1.000 |
| HOSP-A01-catering | out_of_domain | [FAIL] | — | — | — |
| HOSP-A02-nominas-vacaciones | out_of_domain | [FAIL] | — | — | — |

# Evaluación automática del RAG

- Estrategia: configured
- Troceado: plano (1800/250)
- Casos: 8
- Acierto de recuperación: 100.0%
- Recall documental: 100.0%
- Precision@k: 22.2%
- nDCG@k: 90.8%
- MRR: 0.875
- Cobertura del sistema (no abstuvo): 100.0%
- Precisión de abstención: N/A
- Recall de abstención: N/A
- Cobertura de términos en respuestas: N/A

| Caso | Tipo | Recuperación | Recall | Precision@k | Primer acierto |
|---|---|---:|---:|---:|---:|
| HOSP-T01-urjencias-triaje | factual | [OK] | 100.0% | 22.2% | 0.500 |
| HOSP-T02-ospitalisacion-cama | factual | [OK] | 100.0% | 22.2% | 1.000 |
| HOSP-T03-farmasia-validacion | factual | [OK] | 100.0% | 33.3% | 1.000 |
| HOSP-T04-lavoratorio-valor-critico | factual | [OK] | 100.0% | 11.1% | 0.500 |
| HOSP-T05-diagnostico-imajen | factual | [OK] | 100.0% | 33.3% | 1.000 |
| HOSP-T06-biopcia-intraoperatoria | factual | [OK] | 100.0% | 11.1% | 1.000 |
| HOSP-T07-preparasion-colonoscopia | factual | [OK] | 100.0% | 33.3% | 1.000 |
| HOSP-T08-estandares-integrasion | factual | [OK] | 100.0% | 11.1% | 1.000 |

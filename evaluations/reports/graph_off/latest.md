# Evaluación automática del RAG

- Casos: 26
- Acierto de recuperación: 91.3%
- Recall documental: 100.0%
- MRR: 1.000
- Acierto de abstención: 0.0%
- Cobertura de términos en respuestas: —

| Caso | Recuperación | Recall | Primer acierto |
|---|---:|---:|---:|
| R01-poblacion-diana | [OK] | 100.0% | 1.000 |
| R02-plazo-lectura-tabla | [OK] | 100.0% | 1.000 |
| R03-discordancia-tercer-lector | [OK] | 100.0% | 1.000 |
| R04-plazo-citacion-valoracion | [OK] | 100.0% | 1.000 |
| R05-consulta-sql-codigo | [OK] | 100.0% | 1.000 |
| R06-tasa-participacion-formula | [OK] | 100.0% | 1.000 |
| R07-indicador-objetivo-tabla | [OK] | 100.0% | 1.000 |
| R08-exclusiones-baja-voluntaria | [OK] | 100.0% | 1.000 |
| R09-comunicacion-no-email | [OK] | 100.0% | 1.000 |
| R10-trazabilidad-sicc | [OK] | 100.0% | 1.000 |
| F01-vigencia-excluye-derogado | [OK] | 100.0% | 1.000 |
| F02-vigencia-historico | [OK] | 100.0% | 1.000 |
| F03-etiqueta-sicc | [OK] | 100.0% | 1.000 |
| F04-etiqueta-historico | [OK] | 100.0% | 1.000 |
| F05-estado-borrador | [OK] | 100.0% | 1.000 |
| F06-estado-vigente-excluye-borrador | [OK] | 100.0% | 1.000 |
| E01-enlace-lectura-a-indicadores | [OK] | 100.0% | 1.000 |
| E02-enlace-protocolo-a-exclusiones | [OK] | 100.0% | 1.000 |
| A01-fuera-de-dominio | [FAIL] | — | — |
| A02-sin-relacion | [FAIL] | — | — |
| A03-tema-ajeno | [FAIL] | — | — |
| D01-plazo-quince-dias-cual | [OK] | 100.0% | 1.000 |
| D02-carta-resultado-normal | [FAIL] | 100.0% | 1.000 |
| D03-protocolo-actual-no-anterior | [OK] | 100.0% | 1.000 |
| D04-quien-resuelve-discordancia | [FAIL] | 100.0% | 1.000 |
| D05-exclusion-caduca-sola | [OK] | 100.0% | 1.000 |

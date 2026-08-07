# P10.9 — Auditoría de consolidación del AI Module

## Objetivo

Construir una rama única de consolidación del AI Module preservando P10.6 como base operativa, incorporando P10.7 de manera aditiva y fail-closed, y trayendo de P10.8 únicamente el cierre normativo de alcance.

## Rama de consolidación

`enzo/p10-9-product-consolidation-ai`

Base deliberada:

- rama: `p10-6-cierre-y-nivel-axial`
- commit: `7b2ad5378bf508ee4846403366ed572b946e8f71`

La base P10.6 contiene, entre otros, de-identificación DICOM, persistencia del input registry, `sliceLevels`, export DICOM SEG/SR y runtime subarticular. Estos cambios no se reemplazan por versiones anteriores.

## P10.7 integrado

Fuente auditada:

- rama: `enzo/p10-7-runtime-integration`
- commit: `462b6a28ef9c614bc22409cf1ae5e3c0e2b07814`
- merge base con P10.6: `b0886b14ac8bac03e0bbaa94beeca5dbf8b40c45`
- estado previo: ramas divergidas; P10.7 estaba 1 commit adelante y 8 commits detrás de la rama P10.6.

Por ese motivo NO se toma la rama P10.7 completa como nueva base. Se integran selectivamente sus 10 cambios.

Archivos P10.7 incorporados:

- `ai_service/pfi_ai_service/contracts/disc_degenerative_findings.py`
- `ai_service/pfi_ai_service/disc_degenerative_runtime.py`
- `ai_service/tests/test_disc_degenerative_findings_contract.py`
- `ai_service/tests/test_disc_degenerative_runtime.py`
- `docs/P10_7_RUNTIME_INTEGRATION.md`
- `docs/contracts/disc-degenerative-findings-v1.schema.json`

Archivos aditivos reconciliados con P10.6:

- `ai_service/pfi_ai_service/api.py`
- `ai_service/pfi_ai_service/readiness.py`
- `ai_service/pfi_ai_service/real_inference_routes.py`
- `ai_service/pfi_ai_service/settings.py`

`api.py` se reconcilia manualmente porque P10.6 también lo modificó. Se preservan sus rutas DICOM SEG/SR, upload/de-identificación, subarticular y slices, y se agregan únicamente los hooks/endpoint P10.7.

También se documentan variables de entorno P10.7 en `.env.example` y `render.yaml`.

## Checkpoint P10.7

Artifact esperado fuera de Git:

`frozen_p10_7_spider_degenerative_multitask.pt`

SHA-256 esperado:

`16eccff327e6794b127fe372ecd03ea619a0f69d939b84ae1aa2e904191c6293`

El runtime debe fallar cerrado ante archivo ausente, hash incorrecto o metadata incompatible.

## Gates P10.7 que siguen bloqueados

Esta consolidación NO cambia a true los gates de producto:

- `PREPROCESSING_PARITY_VALIDATED = false`
- `AUTOMATIC_DISC_LOCALIZATION_VALIDATED = false`

Por lo tanto `POST /degenerative-findings/disc-multitask/predict` sigue bloqueado para requests productivos con `422 DISC_DEGENERATIVE_PREPROCESSING_NOT_AVAILABLE` hasta cerrar ambas condiciones con evidencia real.

No se debe cambiar esos flags por configuración ni para “hacer pasar” el E2E.

## P10.8 integrado

Fuente de research:

- rama: `enzo/p10-8-clinical-expansion-preflight`
- commit observado al cierre: `0cd72d1fda51fcb755f4cf555791f0534a90404d`

La comparación contra P10.6 mostró una rama de research con notebooks y duplicados históricos. No se importa masivamente a la rama de producto.

Se incorporan como fuente normativa:

- `docs/P10_8_PREFLIGHT_CLOSURE.md`
- `docs/P10_8_PREFLIGHT_CLOSURE.json`

Decisión P10.8:

`P10_8_PREFLIGHT_CLOSED_NO_TRAINING`

No existe checkpoint P10.8 y no se abre Notebook 77.

## Alcance P10.8 permitido para producto

Protocolos solamente:

- altura discal descriptiva con revisión profesional;
- diámetro AP descriptivo con revisión profesional;
- revisión multiframe de hernia sin reentrenamiento.

No habilitados como nuevas inferencias automáticas:

- facet hypertrophy;
- ligamentum flavum hypertrophy;
- annular tear;
- nerve root compression;
- epidural fat.

Alineación sagital↔axial automática permanece no validada.

## Próximos gates antes de E2E Backend↔AI

1. comprobar regresión completa P10.6;
2. montar y validar checkpoint subarticular real;
3. montar y validar checkpoint P10.7 real;
4. reproducir preprocessing P10.7 exactamente como entrenamiento/export;
5. derivar ROI/nivel discal desde segmentación existente con pruebas de paridad;
6. mantener de-identificación y guardas de PII;
7. ejecutar smoke HTTP real;
8. recién después conectar Backend y persistencia P10.7.

## Regla de merge

Esta rama no se fusiona a `main` hasta que los gates de regresión y E2E definidos para la consolidación estén verdes.

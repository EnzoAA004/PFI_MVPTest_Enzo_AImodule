# PF-008 a PF-011 - Inferencia productiva, entrada de archivo, overlays y mediciones

Estado: En progreso.

## Objetivo

Convertir la lógica experimental de inferencia E13 en módulos reutilizables dentro del servicio Python `pfi_ai_service`, preparados para ser consumidos por endpoints reales en el bloque siguiente.

## Tickets cubiertos

- PF-008: convertir pipeline E13 a módulos productivos.
- PF-009: soportar entrada de archivo.
- PF-010: generar overlays finales.
- PF-011: calcular mediciones geométricas mínimas.

## Alcance

Este bloque no expone todavía endpoints FastAPI definitivos. Su función es dejar módulos importables y testeables:

- `preprocessing.py`
- `measurements.py`
- `overlay.py`
- `inference.py`
- `pipeline.py`

## Entradas esperadas

- `config/data_freeze_config.json`
- `config/model_registry_final.json`
- modelos finales en `models/final/`

## Salidas esperadas

- módulos Python productivos en `ai_service/pfi_ai_service/`
- `results/PF008_PF011_product_inference_modules/PF008_PF011_module_inventory.csv`
- `results/PF008_PF011_product_inference_modules/PF008_PF011_smoke_test_report.json`
- `results/PF008_PF011_product_inference_modules/PF008_PF011_checks.csv`
- `docs/PF008_PF011_product_inference_modules.md`

## Criterio de aceptación

- Los módulos se escriben en el paquete `pfi_ai_service`.
- Los módulos se importan correctamente.
- Se puede cargar el registry final.
- Se puede ejecutar un smoke test sintético de preprocesamiento, mediciones y overlay.
- Queda una interfaz clara para que PF-012 a PF-019 agreguen endpoints reales.

## Decisión metodológica

El bloque prepara inferencia productiva modular sin prometer diagnóstico autónomo ni reconstrucción 3D real. El resultado sigue siendo asistivo y sujeto a revisión profesional.

## Próximo bloque

PF-012 a PF-019: agente endurecido y endpoints reales del servicio Python.

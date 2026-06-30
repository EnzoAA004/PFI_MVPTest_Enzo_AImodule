# PF-012 a PF-019 - Agente endurecido y endpoints del servicio Python

Estado: En progreso.

## Objetivo

Transformar los modulos productivos de inferencia generados en PF-008 a PF-011 en una capa de servicio Python expuesta por FastAPI, con politica de agente, reportes por caso, pruebas de regresion, manejo de errores y Dockerfile base.

## Tickets cubiertos

- PF-012: endurecer reglas del agente.
- PF-013: generar reportes por caso productivos.
- PF-014: agregar pruebas de regresion del agente.
- PF-015: endpoint de inferencia sagital.
- PF-016: endpoint de inferencia axial.
- PF-017: endpoint de pipeline completo.
- PF-018: manejo de errores y logs.
- PF-019: Dockerfile del servicio Python.

## Entradas esperadas

- `config/model_registry_final.json`
- `config/data_freeze_config.json`
- modulos `preprocessing.py`, `measurements.py`, `overlay.py`, `inference.py`, `pipeline.py`
- modelos finales en `models/final/`

## Salidas esperadas

- `ai_service/pfi_ai_service/agent_policy.py`
- `ai_service/pfi_ai_service/case_reporting.py`
- `ai_service/pfi_ai_service/service_endpoints_runtime.py`
- `ai_service/pfi_ai_service/api.py` actualizado
- `ai_service/Dockerfile`
- reportes smoke y checks en `results/PF012_PF019_service_endpoints_agent/`
- documentacion en `docs/PF012_PF019_service_endpoints_agent.md`

## Criterio de aceptacion

- El paquete importa correctamente.
- FastAPI levanta con TestClient.
- `/health` responde OK.
- `/models` devuelve los modelos finales registrados.
- `/inference/sagittal` responde sobre caso smoke.
- `/inference/axial` responde sobre caso smoke.
- `/pipeline/run` genera salida completa con decision del agente.
- `/agent/report/{run_id}` recupera reporte por caso.
- Existe Dockerfile base.
- La respuesta mantiene `human_review_required=true` y no emite diagnostico clinico autonomo.

## Decision metodologica

Los endpoints exponen un servicio de apoyo y revision. La salida es una respuesta tecnica trazable con prioridad, flags, mediciones y overlay, no un diagnostico clinico.

## Proximo bloque

PF-020 a PF-025: backend Spring Boot real, DTOs definitivos, persistencia minima y consumo del servicio Python.

# E15 - Traduccion a backend/MVP

Estado: En progreso.

## Objetivo

Traducir la cadena funcional construida en Colab a una estructura inicial de servicio Python para MVP. E15 no entrena modelos nuevos y no busca reconstruccion 3D real. Su objetivo es empaquetar la logica consolidada de E13/E14 en modulos reutilizables que luego puedan integrarse con el backend Spring Boot o exponerse como microservicio Python.

## Dependencias

- E10: modelo axial T2 final.
- E11: mapeo y decision metodologica de clases axiales.
- E12: modelo sagital final consolidado.
- E13: pipeline comun de inferencia multiplanar.
- E14: agente/orquestador IA y reportes de prioridad.

## Alcance

E15 debe generar una carpeta de servicio dentro del repo:

- `ai_service/`
- `ai_service/pfi_ai_service/`
- `ai_service/pfi_ai_service/settings.py`
- `ai_service/pfi_ai_service/schemas.py`
- `ai_service/pfi_ai_service/quality.py`
- `ai_service/pfi_ai_service/agent.py`
- `ai_service/pfi_ai_service/reporting.py`
- `ai_service/pfi_ai_service/api.py`
- `ai_service/requirements-ai-service.txt`
- `ai_service/README.md`

## Funciones esperadas

- Definir contratos de entrada/salida para inferencia y reporte.
- Centralizar rutas de modelos y resultados.
- Reutilizar reglas de quality flags.
- Reutilizar reglas del agente E14.
- Exponer endpoints iniciales estilo FastAPI:
  - `/health`
  - `/models`
  - `/agent/worklist`
  - `/agent/report`

## Salidas esperadas de Colab

- `results/E15_backend_mvp_translation/E15_service_file_inventory.csv`
- `results/E15_backend_mvp_translation/E15_smoke_test_report.json`
- `docs/E15_backend_mvp_translation_conclusion.md`
- `docs/E15_backend_mvp_codex_prompt.md`

## Criterios de aceptacion

- Se generan los archivos del servicio en el repo.
- Se puede importar el paquete `pfi_ai_service` desde Colab.
- El smoke test carga resultados E14.
- El smoke test genera resumen de worklist y prioridades.
- Queda documentado como base para backend/MVP.

## Decision metodologica

E15 representa la transicion desde notebooks experimentales hacia arquitectura de producto. El servicio sigue siendo de apoyo a decision y conserva revision humana obligatoria.

## Proximo paso posterior

E16 puede tomar dos caminos:

1. Integracion con backend Spring Boot: definir endpoint Java que invoque el servicio Python.
2. Spike 3D/geometrico: investigar series DICOM con axial y sagital del mismo paciente.

Recomendacion: avanzar primero con integracion backend/MVP antes del 3D real.

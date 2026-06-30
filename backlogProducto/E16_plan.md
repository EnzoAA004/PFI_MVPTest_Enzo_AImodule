# E16 - Integracion backend/MVP

Estado: En progreso.

## Objetivo

Validar la estructura `ai_service` generada en E15 como componente integrable al backend/MVP. E16 no entrena modelos nuevos: prueba el servicio FastAPI, define contratos JSON y deja un ejemplo de puente Spring Boot -> Python service.

## Dependencias

- E13: pipeline comun de inferencia multiplanar.
- E14: agente/orquestador IA.
- E15: paquete Python `pfi_ai_service` y API FastAPI inicial.

## Alcance

- Importar el paquete `pfi_ai_service` desde el repo.
- Instanciar la app FastAPI.
- Probar endpoints con `TestClient` sin necesidad de exponer puertos:
  - `/health`
  - `/models`
  - `/agent/worklist`
  - `/agent/report`
- Validar estructura de respuestas JSON.
- Generar contrato backend/frontend.
- Generar ejemplo de cliente Spring Boot.
- Generar reporte de smoke test.

## Salidas esperadas

- results/E16_backend_integration_smoke/E16_fastapi_endpoint_checks.csv
- results/E16_backend_integration_smoke/E16_agent_report_response_sample.json
- results/E16_backend_integration_smoke/E16_backend_contract_summary.json
- results/E16_backend_integration_smoke/E16_backend_integration_report.json
- docs/E16_backend_integration_conclusion.md
- docs/E16_spring_boot_bridge_contract.md
- docs/E16_frontend_payload_contract.md
- docs/E16_codex_backend_prompt.md

## Criterios de aceptacion

- El paquete `pfi_ai_service` importa correctamente.
- FastAPI `app` se instancia correctamente.
- `/health` responde 200.
- `/models` responde 200 y lista ambos modelos.
- `/agent/worklist` responde 200 y contiene 12 items.
- `/agent/report` responde 200 y conserva las distribuciones de E14.
- Se genera documentacion para backend y frontend.

## Decision metodologica

E16 representa el pasaje desde notebooks/servicio Python hacia integracion de producto. El servicio Python concentra IA y orquestacion; el backend Spring Boot puede consumirlo por HTTP y exponerlo al frontend sin mezclar inferencia pesada dentro de Java.

## Proximo paso posterior

E17 puede avanzar por dos caminos:

1. Implementar el bridge real en Spring Boot.
2. Preparar UI/frontend para visualizar worklist, overlays, prioridades y reportes.

Recomendacion: avanzar primero con el bridge Spring Boot minimo y luego con UI.

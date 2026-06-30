# PF-020 a PF-025 - Backend Spring Boot real y persistencia minima

Estado: En progreso.

## Objetivo

Construir la capa backend Spring Boot del producto final para consumir el servicio Python FastAPI validado en PF-012 a PF-019, exponer endpoints al frontend y preparar persistencia minima de reportes/estados de revision.

## Tickets cubiertos

- PF-020: crear proyecto backend real o integrar scaffold.
- PF-021: DTOs definitivos del contrato IA.
- PF-022: endpoint backend `/api/ai/agent/report`.
- PF-023: endpoint backend de ejecucion de caso.
- PF-024: modelo de datos minimo.
- PF-025: persistencia simple.

## Entradas esperadas

- Servicio Python FastAPI funcional.
- Contrato de endpoints PF-012 a PF-019.
- Frontend E18 como consumidor futuro.

## Salidas esperadas

- `backend_product_spring/` con scaffold backend ejecutable.
- DTOs Java del contrato IA.
- Cliente HTTP hacia servicio Python.
- Controlador REST para frontend.
- Modelo minimo de entidad/reporte/revision.
- Repositorio in-memory o persistencia H2/PostgreSQL preparada.
- Smoke test estatico de archivos y contrato.
- Documentacion del contrato backend.

## Endpoints backend esperados

- `GET /api/ai/health`
- `GET /api/ai/models`
- `POST /api/ai/pipeline/run`
- `GET /api/ai/agent/report/{runId}`
- `PATCH /api/ai/review/{runId}`

## Criterio de aceptacion

- Existe estructura Spring Boot compilable o integrable.
- DTOs reflejan los campos necesarios del servicio Python.
- El controller expone los endpoints esperados para el frontend.
- El servicio backend conserva `humanReviewRequired` y no transforma la salida en diagnostico clinico.
- Existe una estrategia de persistencia minima para reportes y estados de revision.

## Decision metodologica

Spring Boot funciona como backend de producto y no como motor IA. La inferencia queda encapsulada en Python/FastAPI; Java orquesta, persiste estados y expone contrato al frontend.

## Proximo bloque

PF-026 a PF-030: frontend final, vistas requeridas, integracion con backend y estados de revision.

# PF-026 a PF-030 - Frontend final, vistas requeridas y estados de revision

Estado: En progreso.

## Objetivo

Construir la capa frontend final del producto para consumir el backend Spring Boot generado en PF-020 a PF-025, reemplazar el mock inicial por contrato real y presentar las vistas necesarias para una demo de producto final.

## Tickets cubiertos

- PF-026: integrar frontend con backend real.
- PF-027: mejorar UI/UX de worklist.
- PF-028: vista de overlay / evidencia visual.
- PF-029: estados de revision en UI.
- PF-030: formulario de observaciones.

## Entradas esperadas

- contrato backend documentado en `docs/PF020_PF025_frontend_backend_contract.md`
- endpoints backend:
  - `GET /api/ai/health`
  - `GET /api/ai/models`
  - `POST /api/ai/pipeline/run`
  - `GET /api/ai/agent/report/{runId}`
  - `PATCH /api/ai/review/{runId}`

## Salidas esperadas

- `frontend_product_ui/` con aplicacion React/Vite/TypeScript.
- Vistas de dashboard, worklist, detalle de caso, overlay/evidencia visual y panel de revision.
- Cliente API contra Spring Boot.
- Mock fallback para demo si backend no esta levantado.
- Docs de ejecucion y contrato visual.
- Checks estaticos de archivos y componentes requeridos.

## Vistas minimas

1. Dashboard resumen del agente.
2. Worklist de casos/items.
3. Detalle de caso.
4. Vista de overlay/evidencia visual.
5. Panel de flags, confianza y accion recomendada.
6. Estado de revision: pendiente, aceptado, observado, descartado.
7. Formulario de observaciones.

## Criterio de aceptacion

- El frontend consume Spring Boot mediante `VITE_API_BASE_URL`.
- Existe fallback mock documentado para demo controlada.
- La UI mantiene la politica de revision profesional.
- Las vistas requeridas estan implementadas como componentes navegables.
- El formulario PATCH de revision queda preparado para actualizar estado y observaciones.

## Decision metodologica

El frontend no debe presentar la salida como diagnostico clinico. Debe presentarla como evidencia tecnica, priorizacion y apoyo a la revision profesional.

## Proximo bloque

PF-031 a PF-038: smoke tests end-to-end, demo script, validacion profesional, documentacion final y limpieza de repo.

# Alineacion de contrato AI Module - Backend

Este documento fija el contrato minimo para integrar el backend Spring Boot con el AI Module FastAPI.

## Payload esperado por el AI Module

El backend puede enviar camelCase:

```json
{
  "caseId": "case-001",
  "plane": "sagittal",
  "modelKey": "sagittal_spider",
  "inputPath": "studies/case-001",
  "metadata": {
    "source": "frontend-demo"
  }
}
```

Tambien se acepta snake_case:

```json
{
  "case_id": "case-001",
  "plane": "sagittal",
  "model_key": "sagittal_spider",
  "input_path": "studies/case-001",
  "metadata": {
    "source": "frontend-demo"
  }
}
```

## Equivalencias snake_case/camelCase

```text
case_id    <-> caseId
model_key  <-> modelKey
input_path <-> inputPath
run_id     <-> runId
overlay_path <-> overlayPath
agent_decision <-> agentDecision
human_review_required <-> humanReviewRequired
not_clinical_diagnosis <-> notClinicalDiagnosis
```

## Payload devuelto por POST /pipeline/run

La respuesta incluye ambas variantes para facilitar la integracion inicial:

```json
{
  "run_id": "abc123",
  "runId": "abc123",
  "case_id": "case-001",
  "caseId": "case-001",
  "plane": "sagittal",
  "model_key": "sagittal_spider",
  "modelKey": "sagittal_spider",
  "measurements": {
    "status": "pending_real_inference",
    "values": []
  },
  "overlay_path": null,
  "overlayPath": null,
  "agent_decision": {
    "agent_status": "requires_professional_review",
    "human_review_required": true,
    "not_clinical_diagnosis": true
  },
  "agentDecision": {
    "agent_status": "requires_professional_review",
    "human_review_required": true,
    "not_clinical_diagnosis": true
  },
  "human_review_required": true,
  "humanReviewRequired": true,
  "not_clinical_diagnosis": true,
  "notClinicalDiagnosis": true
}
```

## Endpoints consumidos por backend

- `GET /health`: health check.
- `GET /models`: modelos tecnicos disponibles/configurados.
- `POST /pipeline/run`: corrida principal del modulo IA.
- `POST /inference/sagittal`: acceso directo a pipeline sagital.
- `POST /inference/axial`: acceso directo a pipeline axial complementario.
- `GET /agent/report/{run_id}`: reporte tecnico por corrida si fue materializado.
- `GET /agent/regression-test`: smoke tecnico de politica asistiva.

## Regla human-in-the-loop

El AI Module no emite diagnostico clinico, no recomienda tratamientos y no toma decisiones medicas automaticas. Toda respuesta de inferencia, pipeline o agente debe conservar `human_review_required=true` y `not_clinical_diagnosis=true`.

El backend y el frontend deben presentar estos resultados como apoyo tecnico revisable y editable por profesionales.

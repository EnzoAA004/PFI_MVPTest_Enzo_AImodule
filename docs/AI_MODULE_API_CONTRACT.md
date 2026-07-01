# Contrato API del AI Module

Base local sugerida:

```text
http://localhost:8000
```

Todas las respuestas tecnicas que representen resultados de procesamiento o inferencia deben conservar la regla asistiva: `human_review_required=true` cuando corresponda.

## Endpoints implementados

### GET /health

Estado del servicio.

Respuesta actual:

```json
{
  "status": "ok",
  "pfi_root": "/content/drive/MyDrive/PFI_MVP",
  "human_review_required": true
}
```

### GET /models

Devuelve el registro tecnico de modelos definido en `pfi_ai_service.settings.MODEL_REGISTRY` y las rutas esperadas para pesos sagitales y axiales.

Respuesta actual:

```json
{
  "models": {},
  "paths": {
    "sagittal_model_path": "...",
    "axial_model_path": "..."
  }
}
```

Nota: el ejemplo resume la forma de respuesta. El contenido real de `models` depende del registro configurado en codigo.

### GET /agent/worklist

Lee `E14_agent_worklist.csv` desde:

```text
PFI_ROOT/results/E14_ai_agent_orchestrator/E14_agent_worklist.csv
```

Respuesta actual:

```json
{
  "rows": 0,
  "items": []
}
```

Si el archivo no existe, responde `404`.

### GET /agent/report

Genera un resumen tecnico del agente IA a partir de la worklist y, si existe, `E14_agent_metrics_summary.csv`.

Respuesta actual:

```json
{
  "summary": {},
  "markdown": "...",
  "items": []
}
```

## Endpoints solicitados pendientes

Los siguientes endpoints no existen actualmente en `ai_service/pfi_ai_service/api.py`. No deben documentarse como disponibles hasta implementarlos.

### POST /inference/sagittal

Estado: pendiente.

Propuesta: recibir una referencia controlada a imagen/serie sagital, ejecutar preprocesamiento e inferencia con el modelo sagital configurado, devolver mascaras/overlays/mediciones como artefactos tecnicos y `human_review_required=true`.

### POST /inference/axial

Estado: pendiente.

Propuesta: recibir una referencia controlada a imagen/serie axial, ejecutar inferencia axial si el modulo complementario esta habilitado, devolver artefactos tecnicos y `human_review_required=true`.

### POST /pipeline/run

Estado: pendiente.

Propuesta: orquestar un pipeline reproducible para un caso de trabajo, registrando entradas, version de modelo, outputs y limitaciones.

### GET /agent/report/{run_id}

Estado: pendiente.

Propuesta: recuperar un reporte especifico por `run_id` en lugar de leer siempre el reporte global actual.

### GET /agent/regression-test

Estado: pendiente.

Propuesta: ejecutar o consultar una prueba de regresion tecnica con datos sinteticos o fixtures no sensibles. No debe requerir datasets privados ni imagenes medicas pesadas.

## Consideraciones metodologicas

- No emitir diagnostico clinico.
- No recomendar tratamientos.
- No presentar resultados como decision medica.
- Las mediciones deben describirse como valores geometricos derivados de mascaras.
- La salida debe ser revisable y editable.
- El backend Spring Boot debe encargarse de persistencia, permisos y trazabilidad de producto.

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

### POST /inference/sagittal

Estado: implementado como contrato tecnico.

Recibe payload compatible con camelCase o snake_case. Ejecuta el pipeline tecnico en modo sagital y devuelve respuesta estructurada para backend. La inferencia real queda pendiente de conectar con modelos/pesos externos.

### POST /inference/axial

Estado: implementado como contrato tecnico.

Recibe payload compatible con camelCase o snake_case. Ejecuta el pipeline tecnico en modo axial y devuelve respuesta estructurada para backend. El plano axial se mantiene como modulo complementario.

### POST /pipeline/run

Estado: implementado como contrato tecnico.

Orquesta una corrida tecnica reproducible para un caso. Actualmente devuelve una respuesta smoke/contrato sin procesar imagenes medicas ni cargar modelos pesados.

### GET /agent/report/{run_id}

Estado: implementado.

Recupera un reporte JSON por `run_id` desde `PFI_OUTPUT_DIR/agent_reports/{run_id}.json`. Si no existe, responde `404`.

### GET /agent/regression-test

Estado: implementado.

Devuelve un chequeo tecnico liviano de politica asistiva, sin modelos pesados ni imagenes medicas.

## Consideraciones metodologicas

- No emitir diagnostico clinico.
- No recomendar tratamientos.
- No presentar resultados como decision medica.
- Las mediciones deben describirse como valores geometricos derivados de mascaras.
- La salida debe ser revisable y editable.
- El backend Spring Boot debe encargarse de persistencia, permisos y trazabilidad de producto.

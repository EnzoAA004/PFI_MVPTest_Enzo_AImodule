# PFI RM Lumbar — AI Module

Servicio Python/FastAPI del prototipo académico de análisis asistido de resonancias magnéticas lumbares. Registra inputs de-identificados, ejecuta preprocessing e inferencia, genera mediciones y assets técnicos, y devuelve resultados estructurados al backend Spring Boot.

```text
Frontend React → Backend Spring Boot → AI Module FastAPI → Model artifacts
```

El frontend nunca consume este servicio directamente. El backend aplica autenticación, permisos, persistencia y el contrato público del producto. El sistema no emite diagnósticos ni reemplaza la revisión profesional.

La arquitectura completa está documentada en el [repositorio backend](https://github.com/EnzoAA004/PFI_MVPTest_Enzo_Backend/blob/main/docs/architecture.md).

## Requisitos

- Python 3.12 recomendado y usado por CI/Docker (`pyproject.toml` admite Python 3.10 o superior);
- entorno virtual;
- Docker, opcional pero recomendado para reproducir el runtime publicado.

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r ai_service/requirements-ai-service.txt
PYTHONPATH=ai_service uvicorn pfi_ai_service.api:app --host 0.0.0.0 --port 8000
```

En PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ai_service\requirements-ai-service.txt
$env:PYTHONPATH = "ai_service"
python -m uvicorn pfi_ai_service.api:app --host 0.0.0.0 --port 8000
```

Comprobar el servicio:

```bash
curl http://localhost:8000/health
```

FastAPI publica su documentación técnica interna en <http://localhost:8000/docs>. Los consumidores del producto deben usar el OpenAPI del backend, no esta API interna.

## Docker

Construcción y ejecución aislada:

```bash
docker build -t pfi-ai-module .
docker run --rm -p 8000:8000 \
  -v pfi-ai-outputs:/app/outputs \
  -v pfi-ai-uploads:/app/uploads \
  pfi-ai-module
```

Para levantar el producto completo, usar `compose.yml` (registry) o `compose.local.yml` (source) del [backend](https://github.com/EnzoAA004/PFI_MVPTest_Enzo_Backend).

La imagen publicada es `ghcr.io/enzoaa004/pfi-ai-module`, con tags `latest` y `sha-<commit>`.

## Tests

```bash
python -m compileall ai_service/pfi_ai_service
python -m pytest -q
```

Para reproducir la medición de cobertura de CI:

```bash
python -m pytest -q --cov --cov-report=xml --cov-report=html --cov-report=term
```

`AI Module CI` también ejecuta Ruff sobre errores, un smoke del contrato FastAPI y el build Docker. El reporte HTML se publica como artifact `pytest-cov-report`.

## Configuración

El listado operativo está en [.env.example](.env.example). Las variables principales son:

| Variable | Uso |
|---|---|
| `PFI_MODEL_DIR` | Directorio de modelos finales y manifests. |
| `PFI_OUTPUT_DIR` | Assets y reportes generados. |
| `PFI_UPLOAD_DIR` | Inputs registrados; debe persistirse para conservar los `inputId`. |
| `PFI_INFERENCE_DEVICE` | `auto`, `cpu` o `cuda`. |
| `PFI_SUBARTICULAR_CHECKPOINT_PATH` | Checkpoint externo del clasificador subarticular. |
| `PFI_P10_7_CHECKPOINT_PATH` | Checkpoint externo del clasificador degenerativo multitarea. |
| `PORT` | Puerto HTTP, `8000` por defecto. |

No versionar secretos, datasets, imágenes médicas ni outputs pesados.

## Modelos y artifacts

La imagen incluye los artifacts de segmentación disponibles en `models/final/`:

- `sagittal_spider_multiclass_final_best.pt`;
- `axial_t2_alkafri_final_v2_candidate.pt`.

Cada artifact se acompaña de manifest y model card. El axial conserva explícitamente el estado `candidate`; no debe presentarse como un modelo que superó un quality gate no demostrado.

No se redistribuyen:

- `frozen_subarticular_checkpoint.pt`;
- `frozen_p10_7_spider_degenerative_multitask.pt`.

Sin esos archivos el servicio arranca normalmente, `/health` informa `not_configured` y sólo los endpoints dependientes responden 503. La procedencia autorizada y los términos/licencia de redistribución de esos checkpoints externos siguen pendientes de verificación documental.

## Flujo de inferencia

```text
Input → validación → preprocessing → modelo → postprocessing
      → mediciones geométricas → assets/resultado estructurado
```

Las respuestas conservan `humanReviewRequired`/`human_review_required` y `notClinicalDiagnosis` cuando corresponde. La disponibilidad de un endpoint o un HTTP 200 no debe confundirse con inferencia real: el backend expone los campos canónicos `degradedMode`, `aiModuleAvailable` y `effectiveInferenceMode` al consumidor final.

## Health y disponibilidad

`GET /health` informa:

- estado del servicio;
- resumen y verificación de artifacts;
- contrato de inferencia por defecto;
- disponibilidad de clasificadores degenerativos;
- obligatoriedad de revisión humana.

`GET /readiness` distingue disponibilidad general de preparación para inferencia real. No carga checkpoints externos ausentes de manera implícita.

## Estructura

```text
ai_service/pfi_ai_service/  API y runtime desplegable
src/lumbar_mri/             procesamiento, mediciones y utilidades reutilizables
tests/                      tests de componentes con datos sintéticos
ai_service/tests/           tests del servicio y contratos
models/final/               artifacts de segmentación distribuidos
notebooks/ y docs/          experimentación y evidencia histórica
```

Los documentos `P9_*`, `P10_*`, `*_EVIDENCE.md`, notebooks y notas de entrenamiento describen iteraciones históricas o evidencia experimental. Para operación actual usar este README, `.env.example`, `/health`, `/readiness` y el OpenAPI del servicio en ejecución.

## Limitaciones

- Prototipo académico, no dispositivo médico.
- Los resultados son técnicos y requieren revisión profesional.
- Los dos clasificadores externos indicados arriba no funcionan sin sus checkpoints autorizados.
- No se distribuyen datasets ni datos identificables.

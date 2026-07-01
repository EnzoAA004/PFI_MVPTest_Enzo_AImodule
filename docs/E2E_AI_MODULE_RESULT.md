# Resultado E2E local del AI Module

Fecha de validacion: 2026-07-01

## Estado Git inicial

```text
git status --short
<sin cambios>
```

```text
git log --oneline -5
2ec8cbf Harden AI module contract for backend integration
54e6a99 Fixes v1.0
665e413 Merge branch 'main' of https://github.com/EnzoAA004/PFI_MVPTest_Enzo_AImodule
5dd9404 Prepare AI module repository for multi-repo architecture
15e3ce9 Add repository split note
```

## Archivos verificados

- `ai_service/pfi_ai_service/api.py`
- `ai_service/pfi_ai_service/pipeline.py`
- `ai_service/pfi_ai_service/agent_policy.py`
- `ai_service/Dockerfile`
- `ai_service/requirements-ai-service.txt`
- `.env.example`
- `docs/AI_BACKEND_CONTRACT_ALIGNMENT.md`
- `docs/AI_MODULE_API_CONTRACT.md`
- `scripts/run_local.sh`
- `scripts/smoke_contract.sh`
- `scripts/smoke_api_contract.py`

## Comandos ejecutados

```bash
python -m compileall ai_service/pfi_ai_service
```

Resultado:

```text
Listing 'ai_service\\pfi_ai_service'...
```

Smoke con FastAPI TestClient:

```bash
python scripts/smoke_api_contract.py
```

Resultado:

```text
AI Module contract smoke test passed.
```

Nota: aparecio una advertencia no bloqueante de `StarletteDeprecationWarning` relacionada con `fastapi.testclient` y `httpx`.

Servidor local:

```bash
cd ai_service
uvicorn pfi_ai_service.api:app --host 0.0.0.0 --port 8000
```

En Windows se ejecuto con el Python del `.venv` local desde la raiz del repositorio y `WorkingDirectory=ai_service`.

## Respuesta de /health

Comando:

```bash
curl http://127.0.0.1:8000/health
```

Respuesta:

```json
{
  "status": "ok",
  "pfi_root": "\\content\\drive\\MyDrive\\PFI_MVP",
  "human_review_required": true
}
```

## Respuesta de /models

Comando:

```bash
curl http://127.0.0.1:8000/models
```

Respuesta resumida:

```json
{
  "models": {
    "sagittal_spider": {
      "plane": "sagittal",
      "num_classes": 4,
      "human_review_required": true
    },
    "axial_t2_alkafri": {
      "plane": "axial",
      "num_classes": 6,
      "human_review_required": true
    }
  },
  "paths": {
    "sagittal_model_path": "\\content\\drive\\MyDrive\\PFI_MVP\\models\\E12_sagittal_multiclass_final_best.pt",
    "axial_model_path": "\\content\\drive\\MyDrive\\PFI_MVP\\models\\E10_axial_t2_final_training_clean_best.pt"
  }
}
```

## Respuesta de /pipeline/run

Payload probado:

```json
{
  "caseId": "case-demo-001",
  "plane": "sagittal",
  "modelKey": "sagittal_spider",
  "inputPath": "demo/case-demo-001",
  "metadata": {
    "source": "e2e-local"
  }
}
```

Respuesta resumida:

```json
{
  "run_id": "a63014c107adef94",
  "runId": "a63014c107adef94",
  "case_id": "case-demo-001",
  "caseId": "case-demo-001",
  "plane": "sagittal",
  "model_key": "sagittal_spider",
  "modelKey": "sagittal_spider",
  "measurements": {
    "status": "pending_real_inference",
    "values": [],
    "source": "contract_smoke_pipeline"
  },
  "overlay_path": null,
  "overlayPath": null,
  "agent_decision": {
    "agent_status": "requires_professional_review",
    "review_priority": "standard",
    "human_review_required": true,
    "not_clinical_diagnosis": true
  },
  "agentDecision": {
    "agent_status": "requires_professional_review",
    "review_priority": "standard",
    "human_review_required": true,
    "not_clinical_diagnosis": true
  },
  "human_review_required": true,
  "humanReviewRequired": true,
  "not_clinical_diagnosis": true,
  "notClinicalDiagnosis": true
}
```

## Problemas detectados

- El API acepta correctamente `camelCase` y `snake_case`.
- Las banderas `human_review_required`/`humanReviewRequired` y `not_clinical_diagnosis`/`notClinicalDiagnosis` se preservan en la respuesta del pipeline.
- En PowerShell, el JSON inline con `curl.exe -d` puede perder comillas internas si no se escapa cuidadosamente. La prueba HTTP real del pipeline se valido con `Invoke-RestMethod` generando JSON desde PowerShell.
- El pipeline actual es un contrato tecnico/smoke: no carga modelos pesados ni procesa imagenes medicas reales. La inferencia real queda pendiente de conectar a artefactos externos controlados.

## Estado final

`ready` para una prueba E2E local con el backend Spring Boot usando el contrato HTTP actual.

El modulo sigue siendo asistivo: no emite diagnostico clinico, no recomienda tratamientos y no toma decisiones medicas automaticas.

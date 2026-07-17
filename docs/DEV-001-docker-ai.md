# DEV-001 - Docker AI Module

## Objetivo

Construir y arrancar el AI Module como servicio FastAPI en Docker con Python 3.12, PyTorch CPU, pydicom y SimpleITK. Los checkpoints `.pt/.pth` no se copian a la imagen: se montan por volumen en runtime usando `PFI_MODEL_DIR`.

## Archivos

- `Dockerfile`: imagen principal del repo.
- `ai_service/Dockerfile`: copia equivalente para compatibilidad con flujos que apunten al subdirectorio.
- `.dockerignore`: excluye entornos locales, datasets, outputs, uploads, notebooks, DICOM/MHA/MHD y checkpoints `.pt/.pth`.

## Base y dependencias clave

- Base: `python:3.12-slim`.
- Runtime API: `uvicorn pfi_ai_service.api:app --host 0.0.0.0 --port ${PORT:-8000}`.
- Variables principales:
  - `PYTHONPATH=/app/ai_service`
  - `PORT=8000`
  - `PFI_MODEL_DIR=/models/final`
  - `PFI_OUTPUT_DIR=/app/outputs`
- Dependencias verificadas dentro de la imagen:
  - `torch=2.13.0+cpu`
  - `pydicom=3.0.2`
  - `SimpleITK=2.5.5`

## Build

```powershell
docker build -t pfi-ai-module:dev-001 .
```

Evidencia local DEV-001:

```text
FINAL_NOCACHE_BUILD_SECONDS=126.0
FINAL_NOCACHE_IMAGE_SIZE_BYTES=411387417
```

## Run local con modelos por volumen

Los `.pt` viven fuera de la imagen. Montar `models/final` como volumen read-only:

```powershell
$modelsPath = (Resolve-Path -LiteralPath 'models\final').Path
docker run --rm --name pfi-ai-dev-001 `
  -p 18080:8000 `
  -e PFI_MODEL_DIR=/models/final `
  -v "${modelsPath}:/models/final:ro" `
  pfi-ai-module:dev-001
```

Health:

```powershell
Invoke-WebRequest -Uri 'http://127.0.0.1:18080/health' -UseBasicParsing
```

Evidencia local DEV-001:

```text
Docker status: Up 8 seconds (healthy), 0.0.0.0:18080->8000/tcp
HEALTH_STATUS=200
"status":"ok","service":"pfi-ai-module","modelsRoot":"/models/final"
```

## Verificacion de artifacts no embebidos

Comando usado:

```powershell
docker run --rm pfi-ai-module:dev-001 sh -c "find /app /models -type f \( -name '*.pt' -o -name '*.pth' \) -print"
```

Resultado: salida vacia. La imagen final no contiene `.pt` ni `.pth` embebidos. En el build final, el contexto Docker fue 12.39 kB, confirmando que los checkpoints pesados quedaron fuera del contexto.

## Notas de seguridad y reproducibilidad

- No incluir `.env`, secretos, datasets, DICOM/MHA/MHD ni outputs en la imagen.
- `models/final/*.pt` se materializa por volumen o por mecanismo de deploy externo.
- El endpoint `/health` puede indicar artifacts disponibles si se monta un volumen con checkpoints locales; eso no implica que los checkpoints esten dentro de la imagen.
- El servicio conserva `humanReviewRequired=true` y `notClinicalDiagnosis=true`.
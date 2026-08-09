# DEV-001 - Docker AI Module

## Objetivo

Construir y arrancar el AI Module como servicio FastAPI en Docker con Python 3.12, PyTorch CPU, pydicom y SimpleITK.

## Archivos

- `Dockerfile`: **el unico Dockerfile del repo**, en la raiz. Es el que usan `render.yaml`
  (`dockerfilePath: ./Dockerfile`), `scripts/docker_smoke.sh` y el compose del stack.
- `.dockerignore`: excluye entornos locales, datasets, outputs, uploads, notebooks y
  DICOM/MHA/MHD. Excluye tambien todos los `.pt/.pth` **salvo** los dos modelos finales, que se
  re-incluyen explicitamente al final del archivo.

Hasta INFRA-005 existia ademas `ai_service/Dockerfile`, declarado aca como "copia equivalente
para compatibilidad con flujos que apunten al subdirectorio". No lo referenciaba ningun deploy,
script ni compose, y habia quedado atras: no copiaba `scripts/`, `infra/` ni `docs/`, asi que la
suite no corria contra esa imagen. Se elimino. Un segundo Dockerfile que nadie construye no da
compatibilidad, da una version de la verdad que nadie verifica.

## Que lleva la imagen

Los dos checkpoints finales **si viajan dentro de la imagen** (`COPY models/final`), porque pesan
1,9 MB cada uno y sin ellos el servicio no puede hacer inferencia real:

```
/app/models/final/sagittal_spider_multiclass_final_best.pt
/app/models/final/axial_t2_alkafri_final_v2_candidate.pt
```

El checkpoint subarticular de P10.6 **no** esta en la imagen ni en Git: se monta por volumen con
`PFI_SUBARTICULAR_CHECKPOINT_PATH`. Sin el, `/health` reporta
`degenerativeFindingModels.subarticular.status = not_configured` y solo ese endpoint responde
503; el resto del servicio funciona.

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

## Verificacion de que artifacts viajan en la imagen

```powershell
docker run --rm --entrypoint sh pfi-ai-module:infra-005 -c "find /app /models -type f \( -name '*.pt' -o -name '*.pth' \) -print"
```

Medido el 2026-08-09 sobre la imagen construida desde `main`:

```text
/app/models/final/sagittal_spider_multiclass_final_best.pt
/app/models/final/axial_t2_alkafri_final_v2_candidate.pt
IMAGE_SIZE=414.3 MB
```

Este bloque decia antes que la salida era vacia y que la imagen no contenia checkpoints. Dejo de
ser cierto cuando `.dockerignore` empezo a re-incluir los dos modelos finales; la verificacion
quedo escrita pero nadie la volvio a correr. Los que **no** estan son los `.pt` de entrenamiento,
los datasets y el checkpoint subarticular.

## Suite dentro del contenedor

El Dockerfile copia `scripts/`, `infra/` y `docs/` justamente para que la suite pueda correr
contra la imagen y no solo contra el repo:

```powershell
docker exec <container> sh -c "pip install --quiet pytest && cd /app && python -m pytest ai_service/tests -q"
```

Resultado el 2026-08-09: `447 passed, 4 skipped`.

## Notas de seguridad y reproducibilidad

- No incluir `.env`, secretos, datasets, DICOM/MHA/MHD ni outputs en la imagen.
- `models/final/*.pt` se materializa por volumen o por mecanismo de deploy externo.
- El endpoint `/health` puede indicar artifacts disponibles si se monta un volumen con checkpoints locales; eso no implica que los checkpoints esten dentro de la imagen.
- El servicio conserva `humanReviewRequired=true` y `notClinicalDiagnosis=true`.
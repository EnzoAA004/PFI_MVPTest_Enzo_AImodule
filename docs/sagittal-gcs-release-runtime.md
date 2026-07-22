# Release sagital GCS en runtime

Este documento describe como el AI Module materializa la release sagital verificada desde Google Cloud Storage y como debe consumirla el backend por HTTP. Es un flujo academico de inferencia asistida; no emite diagnosticos clinicos y siempre requiere revision humana.

## Variables de entorno

```bash
PFI_MODEL_DIR=models/final
PFI_OUTPUT_DIR=outputs
PFI_GCP_PROJECT_ID=pfi-asplanatti-fabrello-v1
PFI_SAGITTAL_RELEASE_URI=gs://pfi-rm-lumbar-artifacts-2026-ef/models/releases/sagittal_spider_final_v1/
PFI_SAGITTAL_RELEASE_CONTENT_SHA256=7420ad4271fe634c970b2a543d1ef8fb1437888c99ca8bd5733a06e5f63e3e7e
PFI_SAGITTAL_RELEASE_MANIFEST_SHA256=d36d0c4fe183ba9a98f0a3471486be5dee1cf1fa820dc32b3a50177ce322be21
PFI_SAGITTAL_MODEL_SHA256=cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944
```

`PFI_SAGITTAL_RELEASE_URI` tiene prioridad sobre `PFI_SAGITTAL_MODEL_URI` para `sagittal_spider`. El axial conserva el flujo anterior de `PFI_AXIAL_MODEL_URI`/rutas locales.

## Autenticacion ADC

Localmente, usar:

```bash
gcloud auth application-default login
```

En Compute Engine, Cloud Run u otro runtime cloud, usar la identidad del workload. No guardar claves privadas, tokens ni service account JSON en el repositorio o la imagen Docker.

## POST /models/sync

Sin body. Query opcional:

```text
POST /models/sync?force=false
```

Respuesta esperada para sagital cuando la release se descarga y verifica:

```json
{
  "modelKey": "sagittal_spider",
  "source": "gcs_verified_release",
  "releaseId": "sagittal_spider_final_v1",
  "releaseContentSha256": "7420ad4271fe634c970b2a543d1ef8fb1437888c99ca8bd5733a06e5f63e3e7e",
  "releaseManifestSha256": "d36d0c4fe183ba9a98f0a3471486be5dee1cf1fa820dc32b3a50177ce322be21",
  "modelSha256": "cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944",
  "status": "synced_verified",
  "artifactSynced": true,
  "manifestSynced": true,
  "modelCardSynced": true,
  "filesReplaced": 3,
  "gcsReadOnly": true
}
```

Si los tres archivos locales ya existen y validan por SHA/manifest, retorna `existing_release_verified` y `filesReplaced=0`. Si hay una version local incompatible y `force=false`, retorna `local_release_mismatch_requires_force` sin reemplazar archivos. Con errores remotos o hashes invalidos retorna `sync_failed` y `readyForRealInference=false`.

## Validaciones

El materializador valida URI `gs://`, bucket esperado, `_SUCCESS.json`, `publish_receipt.json`, `release_manifest.json`, cantidad de artifacts, ausencia de `sourcePath`, tamano y SHA-256 de cada artifact, SHA del checkpoint, manifest runtime con `baselineReady=true`, `sha256Status=MATCH`, `modelKey=sagittal_spider`, `version=sagittal-spider-final-v1`, 4 clases, `baseChannels=16` y `targetSize=[256,256]`.

La descarga es atomica: primero baja a `models/.staging/sagittal_spider_final_v1-*`; solo si toda la release valida reemplaza en `PFI_MODEL_DIR`:

- `sagittal_spider_multiclass_final_best.pt`
- `sagittal_spider_multiclass_final_best.pt.manifest.json`
- `sagittal_spider_multiclass_final_best.pt.modelcard.md`

Tambien guarda provenance privada en `PFI_MODEL_DIR/.releases/sagittal_spider_final_v1/`:

- `_SUCCESS.json`
- `publish_receipt.json`
- `release_manifest.json`

Una corrida idempotente solo retorna `existing_release_verified` si esos tres documentos existen, sus SHA coinciden, el receipt corresponde a `_SUCCESS`, el release manifest no expone `sourcePath`, y el checkpoint, runtime manifest y model card locales coinciden en tamano y SHA contra el release manifest. Si falta o discrepa cualquier provenance, `force=false` retorna `local_release_mismatch_requires_force`; con `force=true` se descarga y reemplaza de forma transaccional.

## POST /pipeline/run

Request sagital estricto:

```json
{
  "caseId": "CASE-DEIDENTIFIED",
  "plane": "sagittal",
  "modelKey": "sagittal_spider",
  "inputPath": "/ruta/local/controlada/input.mha",
  "metadata": {
    "inferenceMode": "real_baseline",
    "allowContractFallback": false,
    "traceId": "backend-trace-id",
    "sliceIndex": 9
  }
}
```

Con `allowContractFallback=false`, un error real se propaga y no se devuelve modo contrato. La respuesta real incluye `runId`, `traceId`, `modelKey`, `modelVersion`, `artifactHash`, `inferenceMode`, `requestedInferenceMode`, `allowContractFallback`, `series`, `masks`, `measurements`, `assets`, `humanReviewRequired=true` y `notClinicalDiagnosis=true`.

## Orientacion SPIDER

Para volumentes sagitales 3D leidos por SimpleITK como `[17,512,512]`, el runtime aplica solo en plano sagital:

```python
np.moveaxis(array, 0, -1)
```

El resultado canonico es `[512,512,17]`, con `selectedAxis=2` y `sliceCount=17`. Arrays ya canonicos como `[512,512,17]` no se vuelven a transformar. El axial no aplica esta transformacion.

La metadata observable incluye:

- `inputShapeNative`
- `inputShapeCanonical`
- `inputOrientationTransform`
- `spacingXyz`
- `arrayAxisSpacingNative`
- `arrayAxisSpacingCanonical`
- `inPlaneSpacing`
- `inPlaneSpacingUnit`
- `sagittalAxis`
- `selectedAxis`
- `selectedSlice`
- `sliceCount`

Para SimpleITK, `spacingXyz=[sx,sy,sz]` y el array nativo llega como `[z,y,x]`, por lo que `arrayAxisSpacingNative=[sz,sy,sx]`. Si se aplica `move_axis_0_to_last`, el array canonico `[y,x,z]` usa `arrayAxisSpacingCanonical=[sy,sx,sz]`; con `selectedAxis=2`, `inPlaneSpacing=[sy,sx]`.

## Assets

El backend debe consumir los assets por los endpoints seguros existentes:

```text
GET /assets/{runId}/{plane}/{assetName}
```

Assets publicos PNG: `input.png`, `overlay.png`, `mask-preview.png`. Arrays raw `mask.npy` y `confidence.npy` son internos y no descarga publica de browser. No se sirven `.pt` ni paths del filesystem.

## E2E opt-in

El script local no corre en CI por defecto:

```bash
set RUN_LIVE_GCS_E2E=1
set PFI_E2E_INPUT_PATH=C:\ruta\controlada\input.mha
set PYTHONPATH=ai_service
python scripts\run_sagittal_final_release_e2e.py
```

Valida sync, cache limpia, `real_baseline`, ausencia de fallback, SHA del checkpoint, version, orientacion y existencia de mask/confidence/overlay. No calcula metricas clinicas ni requiere ground truth.

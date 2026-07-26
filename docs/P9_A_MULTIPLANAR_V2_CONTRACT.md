# P9-A Multiplanar V2 Contract

## Motivacion

P9-A introduce un contrato canonico para corridas multiplanares del AI Module. El objetivo es separar ejecucion, normalizacion, serializacion publica y compatibilidad legacy, evitando aliases v1, metadata libre, datos demo, rutas locales y estructuras duplicadas.

El endpoint legacy `POST /multiplanar/run` se conserva para el backend P8. El nuevo contrato publico esta en `POST /v2/multiplanar/run` y el descriptor de capacidades en `GET /v2/multiplanar/contract`.

Desde P9-A.1 el endpoint legacy tambien usa el core canonico:

`MultiplanarRunRequest v1 -> LegacyMultiplanarV1RequestMapper -> CanonicalMultiplanarExecutor -> LegacyMultiplanarV1Adapter -> response v1`

`run_multiplanar_pipeline()` queda fuera del flujo publico y se mantiene solo como compatibilidad interna/deprecated.

## Request V2

```json
{
  "caseId": "CASE-101",
  "traceId": "trace-optional",
  "inferenceMode": "real_baseline",
  "allowContractFallback": false,
  "planes": {
    "sagittal": {
      "inputId": "inp_...",
      "modelKey": "sagittal_spider"
    },
    "axial": null
  },
  "options": {
    "sliceIndex": null,
    "sliceAxis": null,
    "sliceWindowRadius": 3,
    "inputOrientationTransform": null
  }
}
```

`caseId` es obligatorio. Al menos un plano debe estar solicitado. `inputPath`, datos de sujeto, fechas de estudio, descripcion y metadata libre no forman parte del contrato v2. La convencion publica es camelCase.

`inferenceMode` acepta `contract`, `mock` y `real_baseline`; `real` se normaliza como alias de entrada a `real_baseline`. `allowContractFallback` vale `false` por defecto y no hay fallback silencioso.

Si `allowContractFallback=true`, el fallback P9-A.1 es global: ante un fallo de `real_baseline` o un modelo solicitado no listo, todos los planos solicitados se devuelven en `contract`, con `synthetic=true` y `fallbackReason` sanitizado. No se mezclan silenciosamente planos reales y sinteticos.

## Modos

| Planos solicitados | workspaceMode |
| --- | --- |
| sagittal | `sagittal_only` |
| axial | `axial_only` |
| sagittal + axial | `dual_plane` |

La politica inicial es `strict_all_requested`: si se solicitan ambos planos, ambos se validan antes de ejecutar cualquier inferencia. Si axial no esta listo, no se ejecuta sagital ni se persiste una corrida parcial.

## Modelos

Sagital:

- `modelKey`: `sagittal_spider`
- version: `sagittal-spider-final-v1`
- artifact: `sagittal_spider_multiclass_final_best.pt`
- SHA-256: `cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944`
- estado: habilitado para `real_baseline`

Axial:

- `modelKey`: `axial_t2_alkafri`
- artifact: `axial_t2_alkafri_final_v2_candidate.pt`
- SHA-256: `a48cbddd858b5615010fd809412f3d17dae6871fbe12a38f4720e6f6bc70f739`
- `trainingStatus`: `candidate_below_quality_gate`
- validation Dice macro foreground: `0.7283182698789201`
- test Dice macro foreground: `0.679348283374592`
- quality gate: `0.70`

Axial sigue experimental. P9-A no cambia manifests, hashes, metricas ni readiness para promoverlo.

## Response V2

La respuesta principal es `pfi.multiplanar-run.v2`:

```json
{
  "status": "completed",
  "schemaVersion": "pfi.multiplanar-run.v2",
  "runId": "multi-...",
  "traceId": "trace-...",
  "caseId": "CASE-101",
  "workspaceMode": "sagittal_only",
  "requestedInferenceMode": "real_baseline",
  "effectiveInferenceMode": "real_baseline",
  "requestedPlanes": ["sagittal"],
  "completedPlanes": ["sagittal"],
  "planes": {
    "sagittal": {},
    "axial": null
  },
  "synthetic": false,
  "fallbackReason": null
}
```

No se devuelven aliases v1 como `run_id`, `case_id`, `model_key`, `inputPath`, `overlayPath`, `measurementValues`, `human_review_required` ni `not_clinical_diagnosis`.

## Plane Result

Cada plano contiene secciones tipadas:

- `model`: version, readiness, trainingStatus, hash y estado de manifest.
- `input`: datos tecnicos seguros como `inputId`, formato, shape, spacing y slice seleccionado.
- `coordinateSpace`: dimensiones reales del runtime, unidades pixel, origen y direcciones.
- `series`: solo la serie procesada.
- `assets`: assets generados existentes con `relativePath`.
- `masks`: clases realmente producidas, sin geometria editable inventada.
- `landmarks`: puntos derivados de mascara cuando aplica.
- `measurements`: lista unica sin `reviewerValue`, sin niveles inventados y con `measurementBasis`.
- `quality`: metricas tecnicas de la corrida, no metricas de entrenamiento.
- `synthetic`: `false` para inferencia real exitosa; `true` para contract/mock o fallback.
- `fallbackReason`: `null` para real exitoso; razon sanitizada para contract/mock/fallback.

`effectiveInferenceMode` se calcula desde los planos completados. Si todos son `real_baseline`, el workspace es `real_baseline`; si todos son `contract`, es `contract`; si todos son `mock`, es `mock`; cualquier mezcla explicita seria `mixed`.

## Contract, Mock y Fallback

En v2, `contract` y `mock` explicitos no reutilizan los fixtures clinicos legacy. La respuesta contiene:

- `series=[]`
- `assets=[]`
- `masks=[]`
- `landmarks=[]`
- `measurements=[]`
- quality con contadores `0`
- `synthetic=true`

No se devuelven series ficticias como `Sagittal T1`, `Axial T1` o `Axial T2 L4-L5`, ni datos demo como `PAT-DEMO` o fechas clinicas. El adapter v1 puede conservar estructura P8 minima, pero queda marcada con `synthetic` y `fallbackReason`.

## Coordinate Space

`coordinateSpace` explicita `width`, `height`, `units`, origen, direcciones y slice fuente. Las mediciones usan `measurementBasis=physical_spacing` solo si existe spacing real; si no, usan `pixel_space`.

Para `real_baseline`, `width` y `height` deben ser mayores que cero y derivarse del runtime. Si faltan `processedShape`/`canonicalShape` validos, la normalizacion falla con `INVALID_MULTIPLANAR_RESPONSE`; no se fabrica `256`.

## Assets

Los assets publicos se expresan como rutas relativas:

```json
{
  "assetName": "overlay.png",
  "role": "overlay",
  "contentType": "image/png",
  "generated": true,
  "relativePath": "/assets/{planeRunId}/{plane}/overlay.png"
}
```

No se exponen URLs locales, hosts externos ni paths absolutos. `mask-preview.png` solo debe aparecer si existe realmente.

## 3D

P9-A no genera reconstruccion 3D paciente-especifica. `threeD.enabled` siempre es `false`:

- sagital-only: `blocked_missing_axial`
- axial-only: `blocked_missing_sagittal`
- dual: `pending_registered_reconstruction`

## Errores

Los errores v2 usan `pfi.error.v2`:

```json
{
  "status": "error",
  "schemaVersion": "pfi.error.v2",
  "code": "MODEL_NOT_READY",
  "message": "Modelo no habilitado para real_baseline: axial_t2_alkafri",
  "traceId": "trace-...",
  "caseId": "CASE-101",
  "requestedPlanes": ["axial"],
  "details": {
    "plane": "axial",
    "modelKey": "axial_t2_alkafri",
    "readiness": "candidate_below_quality_gate"
  },
  "governance": {
    "humanReviewRequired": true,
    "notClinicalDiagnosis": true,
    "deidentified": true,
    "diagnosisGenerated": false
  }
}
```

No se incluyen stack traces, paths internos, secretos ni contenido de archivos.

Errores adicionales P9-A.1:

- `INVALID_MULTIPLANAR_REQUEST`: JSON invalido, body no objeto o shape de request invalida.
- `CONTRACT_FALLBACK_DISABLED`: reservado para fallos donde se solicito real sin fallback permitido.
- `INVALID_MULTIPLANAR_RESPONSE`: normalizacion interna invalida, por ejemplo serie real ausente o `coordinateSpace` 0x0.

## Governance

Toda respuesta v2 mantiene:

- `humanReviewRequired=true`
- `notClinicalDiagnosis=true`
- `deidentified=true`
- `diagnosisGenerated=false`

El AI Module no genera diagnostico clinico.

## Compatibilidad V1

`LegacyMultiplanarV1Adapter` queda marcado como deprecated para sostener P8 hasta P9-B/P9-C. La equivalencia principal es:

| V1 | V2 |
| --- | --- |
| `schemaVersion=multiplanar-run-v1` | `schemaVersion=pfi.multiplanar-run.v2` |
| `sagittalRunReady` | `planes.sagittal != null` |
| `axialRunReady` | `planes.axial != null` |
| `dualRunReady` | ambos planos completados |
| `metadata.traceId` | `traceId` top-level |
| aliases snake_case | removidos del contrato canonico |

Campos legacy conservados temporalmente:

- `planes.{plane}.measurements.values`
- `planes.{plane}.measurementValues`
- `planes.{plane}.modelArtifact`
- `planes.{plane}.aiOutput`
- `sagittalRunReady`, `axialRunReady`, `dualRunReady`
- `humanReviewRequired`, `notClinicalDiagnosis`

No deben agregarse nuevos consumidores v1.

## Persistencia

Los reportes v2 se guardan en:

`outputs/multiplanar_reports_v2/{runId}.json`

El contenido es exactamente la respuesta v2 serializada, sin aliases, datos clinicos demo, rutas internas ni secretos.

## Migracion

P9-B debe migrar el backend desplegado desde `POST /multiplanar/run` hacia `POST /v2/multiplanar/run`. P9-C puede retirar el contrato legacy una vez que no queden consumidores v1.

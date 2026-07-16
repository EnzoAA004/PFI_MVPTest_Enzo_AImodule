# AI-013 - inputId server-side contract

## Objetivo
Introducir una indirección opaca `inputId` para que backend/frontend no envíen rutas internas al pipeline. Este ticket no implementa upload multipart; registra recursos ya presentes server-side mediante `sourceKey`.

## POST /inputs

Request:
```json
{
  "caseId": "CASE-AI013-PIPELINE",
  "plane": "sagittal",
  "sourceKey": "fixture:sagittal_sample"
}
```

`sourceKey` disponibles en este ticket:
- `fixture:sagittal_sample` -> fixture sagital server-side.
- `fixture:axial_sample` -> fixture axial server-side.

Response:
```json
{
  "inputId": "inp_<uuid-opaco>",
  "caseId": "CASE-AI013-PIPELINE",
  "plane": "sagittal",
  "format": "npy",
  "size": 262272
}
```

La respuesta no incluye rutas internas, nombres de directorios del repo, ni paths de outputs/modelos.

## Registro server-side

Implementación: `ai_service/pfi_ai_service/input_registry.py`.

- Mantiene un registro en memoria `inputId -> InputRecord`.
- `InputRecord` conserva `caseId`, `plane`, `format`, `size`, `sourceKey` y la ruta real server-side.
- La ruta real se usa solo internamente para ejecutar PyTorch.
- `inputId` es opaco (`inp_` + UUID hex) y no deriva del path.
- Si el `inputId` no existe, no coincide con `caseId` o no coincide con `plane`, se devuelve error controlado.

## Pipeline con inputId

`/pipeline/run` acepta ahora `inputId` además de `inputPath`:
```json
{
  "caseId": "CASE-AI013-PIPELINE",
  "plane": "sagittal",
  "modelKey": "sagittal_spider",
  "inputId": "inp_<uuid-opaco>",
  "metadata": {
    "inferenceMode": "real_baseline",
    "allowContractFallback": false,
    "traceId": "trace-ai013-pipeline"
  }
}
```

Cuando llega `inputId`:
- Se resuelve internamente a ruta real antes de llamar al runtime.
- La respuesta reemplaza `inputPath/sourcePath` por `inputId`.
- Los `outputFiles` se exponen como `{ generated, fileName }`, sin paths internos.
- `modelArtifact` conserva estado/hash/manifest, pero remueve campos `path`.

## Multiplanar con inputId

`/multiplanar/run` acepta:
- `sagittalInputId`
- `axialInputId`

También conserva compatibilidad con `sagittalInputPath`/`axialInputPath` para tests previos y transición, pero el camino preferido es `inputId`.

## Validación

Comando ejecutado:
```text
$env:PYTHONPATH="ai_service"
.venv\Scripts\python.exe -m pytest ai_service\tests\test_input_id_pipeline.py -q
```

Resultado real:
```text
....                                                                     [100%]
4 passed, 1 warning in 9.35s
```

Casos cubiertos:
- `POST /inputs` devuelve `inputId` + metadata y no expone path.
- `/pipeline/run` ejecuta `real_baseline` con `inputId` y no expone paths internos.
- `inputId` inexistente devuelve error claro.
- `/multiplanar/run` ejecuta `real_baseline` con `sagittalInputId` y `axialInputId` sin exponer paths internos.

## Fuera de alcance

- Upload multipart seguro queda para AI-014.
- Persistencia durable del registro queda para tickets posteriores si se requiere multi-proceso o restart-safe.


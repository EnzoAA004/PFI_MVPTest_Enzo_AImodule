# AI-015 - Asset registry by runId

## Objetivo
Registrar los assets generados por una corrida real_baseline para que `(runId, plane, assetName)` resuelva internamente a archivos generados por el runtime, sin aceptar paths arbitrarios ni exponer rutas internas.

## Diseño

Implementación: `ai_service/pfi_ai_service/asset_registry.py`.

Clave interna:
```text
(runId, plane, assetName) -> AssetRecord
```

`AssetRecord` contiene:
- `run_id`
- `plane` (`sagittal` o `axial`)
- `asset_name`
- `path` interno
- `size`

La ruta real queda solo en memoria del servidor. La metadata pública expuesta en respuestas contiene únicamente:
```json
{
  "runId": "...",
  "plane": "sagittal",
  "assetName": "overlay.png",
  "size": 12345
}
```

## Registro

`save_outputs()` en `real_inference_runtime.py` genera:
- `input.png`
- `mask.npy`
- `confidence.npy`
- `overlay.png`

Luego llama a `register_run_assets(runId, plane, outputs)`. La respuesta real_baseline agrega `assets` y `metadata.assets` con metadata pública, no paths.

## Allowlist

Asset names permitidos:
- `input.png`
- `mask.npy`
- `confidence.npy`
- `overlay.png`
- `mask-preview.png` (reservado si se agrega preview más adelante)

Cualquier otro nombre no resuelve.

## Seguridad

- La resolución solo acepta `runId`, `plane`, `assetName`.
- `assetName` debe ser basename exacto, sin `/`, `\\`, ni `..`.
- No se aceptan paths ni rutas relativas.
- `runId` inexistente devuelve error controlado.
- No se sirven assets todavía; AI-016 definirá si se exponen endpoints de descarga/preview.

## Validación

Comando ejecutado:
```text
$env:PYTHONPATH="ai_service"
.venv\Scripts\python.exe -m pytest ai_service\tests\test_asset_registry.py -q
```

Resultado real:
```text
.......                                                                  [100%]
7 passed, 1 warning in 4.98s
```

Casos cubiertos:
- Registro de assets en una corrida real con fixture sagital.
- Resolución OK de asset allowlisted.
- Rechazo de asset fuera de allowlist.
- Rechazo de traversal (`../overlay.png`, `sagittal/overlay.png`, etc.).
- Rechazo de runId inexistente.

## Git hygiene

- `outputs/` sigue ignorado.
- No se commitean outputs, uploads ni checkpoints `.pt`.


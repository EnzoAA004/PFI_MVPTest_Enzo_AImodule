# AI-016 - Servir assets seguros desde AI Module

## Objetivo

Exponer assets generados por corridas usando solo la clave logica `(runId, plane, assetName)`, sin aceptar rutas de filesystem ni exponer paths internos en el endpoint publico.

## Endpoint publico

`GET /assets/{runId}/{plane}/{assetName}`

- `runId`: identificador opaco de corrida ya registrada en `asset_registry`.
- `plane`: `sagittal` o `axial`.
- `assetName`: nombre permitido por allowlist.

El endpoint resuelve internamente el archivo real mediante `asset_registry.resolve_run_asset(...)`. No acepta paths ni parametros de ruta alternativos.

## Allowlist y tipos servibles

Allowlist total registrada por corrida:

- `input.png`
- `overlay.png`
- `mask-preview.png` (reservado)
- `mask.npy`
- `confidence.npy`

Assets publicos servibles al browser:

- `input.png` -> `image/png`
- `overlay.png` -> `image/png`
- `mask-preview.png` -> `image/png`

Assets internos no servidos por el endpoint publico:

- `mask.npy`
- `confidence.npy`

Los arrays raw quedan registrados para uso interno/controlado, pero el endpoint publico responde `403` y no los descarga como `application/octet-stream`.

## Artifacts bloqueados

`*.pt` y `*.pth` no pertenecen a la allowlist de assets y no son servibles por `GET /assets/...`. Cualquier intento como `model.pt` o `model.pth` responde `403`.

## Errores

- `403`: `assetName` invalido, intento de traversal, asset fuera de allowlist, asset raw interno (`.npy`) o artifact no permitido (`.pt/.pth`).
- `404`: `runId`, `plane` o asset permitido pero no registrado/no disponible.

Las respuestas de error no incluyen paths internos.

## Evidencia pytest

Comando ejecutado:

```powershell
$env:PYTHONPATH="ai_service"
.\.venv\Scripts\python.exe -m pytest ai_service\tests\test_asset_serving.py -q
```

Resultado:

```text
....                                                                     [100%]
4 passed, 1 warning in 3.95s
```

Cobertura del test enfocado:

- `200` + `content-type: image/png` para `overlay.png` generado por una corrida real con fixture sagital.
- `403` para traversal y asset fuera de allowlist.
- `404` para `runId` inexistente y `mask-preview.png` no registrado.
- `403` para `mask.npy`, `confidence.npy`, `model.pt` y `model.pth`.

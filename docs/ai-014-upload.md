# AI-014 - Secure upload contract

## Objetivo
Permitir que el AI Module reciba un archivo demo/deidentificado por multipart, lo valide, lo guarde con nombre server-side y devuelva un `inputId` opaco compatible con `/pipeline/run` y `/multiplanar/run`.

## Endpoint

`POST /inputs` acepta dos modos:

1. JSON server-side existente de AI-013:
```json
{
  "caseId": "CASE-AI014",
  "plane": "sagittal",
  "sourceKey": "fixture:sagittal_sample"
}
```

2. Multipart upload:
```text
file=<archivo>
caseId=CASE-AI014-UPLOAD
plane=sagittal|axial
```

Response:
```json
{
  "inputId": "inp_<uuid-opaco>",
  "caseId": "CASE-AI014-UPLOAD",
  "plane": "sagittal",
  "format": "npy",
  "size": 262272
}
```

La respuesta no expone rutas internas ni el filename original del cliente.

## Validaciones

- Extensiones permitidas: `.npy`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.mha`, `.mhd`, `.dcm`.
- Límite de tamaño configurable por `PFI_MAX_UPLOAD_BYTES`.
- Default: `25 MiB` (`26214400` bytes).
- Directorio configurable por `PFI_UPLOAD_DIR`.
- Default: `uploads/inputs`.
- `plane` debe ser `sagittal` o `axial`.
- Archivo vacío: rechazado.
- Si excede el límite: HTTP 413.
- Extensión inválida: HTTP 400.

## Anti path-traversal

- El filename del cliente se usa solo para leer la extensión.
- El path de guardado se genera como `uploads/inputs/<plane>/<inputId>.<ext>`.
- `inputId` es `inp_` + UUID hex.
- Se valida que el destino resuelto permanezca dentro del upload root.
- Ejemplo `../../evil.npy` se guarda como `<inputId>.npy` dentro del directorio server-side, nunca fuera.

## Pipeline

El `inputId` subido se registra en memoria y puede usarse en `/pipeline/run`:
```json
{
  "caseId": "CASE-AI014-PIPELINE",
  "plane": "sagittal",
  "modelKey": "sagittal_spider",
  "inputId": "inp_<uuid-opaco>",
  "metadata": {
    "inferenceMode": "real_baseline",
    "allowContractFallback": false,
    "traceId": "trace-ai014-upload-pipeline"
  }
}
```

La respuesta por flujo `inputId` no expone `inputPath`, `sourcePath`, rutas de outputs ni rutas de modelos. `outputFiles` se publica como `{ generated, fileName }`.

## Validación

Comando ejecutado:
```text
$env:PYTHONPATH="ai_service"
.venv\Scripts\python.exe -m pytest ai_service\tests\test_input_upload.py -q
```

Resultado real:
```text
.....                                                                    [100%]
5 passed, 1 warning in 3.61s
```

Casos cubiertos:
- Upload válido devuelve `inputId` + metadata sin path.
- Extensión inválida rechaza con error claro.
- Tamaño excedido rechaza con HTTP 413 y no conserva archivo parcial.
- Filename con path traversal se neutraliza y guarda con nombre server-side.
- `/pipeline/run` ejecuta `real_baseline` con el `inputId` subido sin exponer paths internos.

## Git hygiene

- `uploads/` está agregado a `.gitignore`.
- No se deben commitear uploads, outputs ni checkpoints `.pt`.


# P10.5-F E2E AI - Integracion de ingesta ZIP con catalogo volumetrico

## Alcance

Esta fase trabaja solamente en `EnzoAA004/PFI_MVPTest_Enzo_AImodule`.
No se trabajo en Backend ni Frontend, no se modifico `main`, no se abrio PR y no se versionaron ZIP, DICOM, MHA, modelos, outputs ni datasets.

## Base y commits

- Base aprobada: `2d15373abb1ac6be718a68ece98369f38555863a` (`P10.5-B test and document volume catalog`).
- Commit de Francisco integrado: `9a049599408ff6016eb0e04e6c772736a9d27b4d`.
- Rama de integracion: `enzo/p10-5-f-e2e-ai`.
- `origin/main` verificado: `40027cc40122b4ce9f907cfef4683ba3785cc8c9`.
- `origin/feat/study-zip-ingestion` verificado: `9a049599408ff6016eb0e04e6c772736a9d27b4d`.
- `merge-base`: `593bffe5afd043c053ece4449b5736da5e8b9307`.

## Estrategia de integracion

Se creo la rama desde la base aprobada y se aplico exclusivamente:

```bash
git cherry-pick 9a049599408ff6016eb0e04e6c772736a9d27b4d
```

No se uso merge, no se rebaseo la rama de Francisco y no se modifico `origin/feat/study-zip-ingestion`.

El cherry-pick no tuvo conflictos textuales. Aun asi se revisaron manualmente los archivos de alto riesgo:

- `ai_service/pfi_ai_service/api.py`
- `ai_service/pfi_ai_service/real_inference_runtime.py`

## Resolucion tecnica

### api.py

Se preservo la estructura P10-AI:

- middleware `X-Trace-Id` con sanitizacion y limite de longitud;
- `clean_for_json()` con `sanitize_public_payload()`;
- handlers registrados por `register_error_handlers(app)`;
- endpoints existentes de health, readiness, pipeline, assets y agent reports.

La integracion agrega `POST /inputs/study`, que recibe `file` y `caseId` por multipart y delega en `register_study_zip()`. La respuesta pasa por `clean_for_json()`, por lo que no publica `Path`, rutas internas ni payloads no serializables.

### real_inference_runtime.py

Se preservo P10.5-B:

- `slice_asset_name()`;
- `save_slice_catalog_assets()`;
- `build_volume_slice_catalog()`;
- preview PNG por cada slice;
- `slices[]` con indices 0-based y `displayIndex` 1-based;
- `hasResults=true` solo en el slice inferido;
- `overlayAsset` solo en el slice inferido;
- metadata geometrica honesta (`geometryComplete`, `geometryMetadataSource`, spacing/origin/direction opcionales);
- ejecucion del modelo en un unico slice seleccionado, no en todos los cortes.

La integracion de Francisco agrega lectura DICOM por contenido para directorios y soporte de series materializadas desde ZIP. Se mantuvo la secuencia esperada:

ZIP de estudio -> extraccion segura -> clasificacion de series -> inputId sagital/axial -> lectura de stack DICOM 3D -> canonicalizacion -> seleccion del slice de inferencia -> inferencia unica -> catalogo completo -> preview por slice -> overlay/resultados solo en slice inferido.

Como reconciliacion propia se agrego soporte de `.ima` como DICOM individual en `load_input()`, ademas del soporte ya existente por directorio/serie.

### input_registry.py

Se preservo:

- `safe_extract_zip()`;
- prevencion Zip Slip;
- limite de bytes comprimidos via `PFI_MAX_UPLOAD_BYTES`;
- limite de bytes descomprimidos via `PFI_MAX_SERIES_UNCOMPRESSED_BYTES`;
- materializacion de series como directorios con archivos `.dcm`;
- `inputId` opaco con validacion de `caseId` y plano.

Como reconciliacion propia se agrego `PFI_MAX_SERIES_FILES`, para poder validar automatizadamente el limite de cantidad de archivos sin crear miles de entradas. Tambien se rechazan explicitamente miembros ZIP con ruta absoluta o drive antes de resolver el destino.

### study_ingestion.py

Se preservo la ingesta aportada:

- `register_study_zip()`;
- deteccion DICOM por contenido con SimpleITK/GDCM;
- soporte de `.ima` y archivos sin extension;
- clasificacion por `ImageOrientationPatient`;
- inferencia de ponderacion por `SeriesDescription` y `EchoTime`;
- seleccion sagital T2, fallback sagital T1, y axial T2;
- limpieza del directorio temporal del estudio al finalizar.

## Pruebas sinteticas

Se agrego `ai_service/tests/test_p10_5_f_study_zip_ingestion.py` con fixtures DICOM sinteticos en `tmp_path`.

Cobertura incluida:

- ZIP valido con sagital y axial;
- endpoint `POST /inputs/study`;
- DICOM `.ima`;
- DICOM sin extension;
- seleccion sagital T2;
- fallback sagital T1;
- seleccion axial T2;
- ausencia de axial;
- multiples candidatas;
- ZIP vacio;
- ZIP corrupto;
- Zip Slip con `../`;
- ruta absoluta dentro del ZIP;
- limite de cantidad de archivos;
- limite de tamano comprimido;
- limite de tamano descomprimido;
- limpieza de extraccion parcial;
- `inputId` asociado al `caseId`;
- `inputId` asociado al plano;
- lectura de serie DICOM como volumen 3D;
- catalogo `slices[]`;
- `slices.length == sliceCount`;
- indices 0-based;
- `displayIndex` 1-based;
- preview por slice;
- `hasResults` solo en el slice inferido;
- overlay solo en el slice inferido;
- ausencia de paths internos en respuesta;
- regresion MHA sagital;
- regresion MHA axial;
- seguridad P10-AI en errores.

## Prueba real opt-in

La prueba real queda en el mismo archivo y solo corre cuando:

```powershell
$env:RUN_PFI_REAL_STUDY_E2E='1'
$env:PFI_E2E_STUDY_ZIP='<ruta-local>'
pytest ai_service/tests/test_p10_5_f_study_zip_ingestion.py -k real_study_zip_opt_in -v
```

El test registra solo:

- cantidad de series;
- plano;
- ponderacion inferida;
- cantidad de slices;
- tiempo;
- resultado general.

No imprime tags DICOM completos, UIDs, paths absolutos, nombres internos, headers medicos ni metadata identificable.

En esta ejecucion local `PFI_E2E_STUDY_ZIP` no estaba configurada, por lo que el caso real quedo saltado.

## Evidencia de ejecucion

Comandos ejecutados:

```powershell
python -m compileall ai_service/pfi_ai_service
$env:PYTHONPATH='ai_service;src'; pytest ai_service/tests/test_p10_ai_security_hardening.py ai_service/tests/test_p10_5_b_volume_catalog.py -q
$env:PYTHONPATH='ai_service;src'; pytest ai_service/tests/test_p10_5_f_study_zip_ingestion.py -q
$env:PYTHONPATH='ai_service;src'; pytest ai_service/tests
```

Resultados:

- `compileall`: OK.
- P10-AI + P10.5-B: `10 passed` en 4.18 s.
- Nuevos tests P10.5-F: `12 passed, 1 skipped` en 19.78 s.
- Suite completa `ai_service/tests`: `211 passed, 5 skipped` en 72.37 s.

Warnings observados:

- `StarletteDeprecationWarning` por `httpx`/`TestClient`.
- `pydicom` depreca `write_like_original` en fixtures sinteticos.

No son fallos funcionales de P10.5-F.

## Seguridad

- No se publican paths de extraccion ni rutas temporales.
- Los errores pasan por la forma estandar con `traceId`.
- `safe_extract_zip()` elimina extracciones parciales ante errores.
- Los limites de upload, bytes descomprimidos y cantidad de archivos son configurables.
- Los DICOM sinteticos no contienen datos reales.
- El fixture real es opt-in y no se versiona.

## Impacto para Backend

El AI Module expone `POST /inputs/study` y devuelve `inputId` opacos por plano. Backend puede proxificar esa operacion mediante multipart sin exponer rutas locales y luego ejecutar `POST /api/ai/multiplanar/run` con `sagittalInputId` y `axialInputId`.

El contrato de corrida multiplanar no cambia: la ingesta de estudio produce inputs compatibles con el pipeline existente.

## Limitaciones y riesgos

- La clasificacion depende de metadata DICOM disponible (`ImageOrientationPatient`, `SeriesDescription`, `EchoTime`).
- Si un estudio real no trae orientacion o GDCM no detecta series, la ingesta rechaza o reporta warnings.
- La prueba real local no se ejecuto porque `PFI_E2E_STUDY_ZIP` no estaba configurada.
- La siguiente fase debe validar proxy, persistencia y serving durable desde Backend con AI activo y apagado.

## Datos no versionados

No se agregaron ZIP, DICOM, MHA, modelos, outputs ni datasets al repositorio. Los tests usan DICOM sinteticos generados en `tmp_path`; el fixture real se referencia solo por variable de entorno.

# Notebook de entrenamiento axial final

## Objetivo

`notebooks/48_axial_final_training_patient_split.ipynb` prepara el entrenamiento final reproducible del modelo axial `axial_t2_alkafri` con split por paciente. Esta tarea no ejecuta entrenamiento ni reemplaza artifacts.

## Estructura esperada de Drive

Default Colab configurable:

```python
PFI_USE_GOOGLE_DRIVE=1
PFI_DRIVE_ROOT=/content/drive/MyDrive/PFI_MVP
AXIAL_E9_CURATED_SPLIT_CSV=/content/drive/MyDrive/PFI_MVP/results/E9_alkafri_axial_t2_final_labels_baseline/E9_t2_final_labels_curated_split.csv
AXIAL_IMAGES_DIR=/content/drive/MyDrive/PFI_MVP/data/AXIAL_ALKAFRI
AXIAL_MASKS_DIR=/content/drive/MyDrive/PFI_MVP/data/AXIAL_ALKAFRI
```

El notebook monta Drive solo en Colab, solo si `PFI_USE_GOOGLE_DRIVE=True`, y solo si `/content/drive/MyDrive` no esta disponible.

El codigo de `ai_service` queda fijado en `AI_SERVICE_COMMIT_SHA=285159982832abb604a176b4302ac83a837ff1c9`, commit validado para `AxialUNet2D` y los helpers de carga de checkpoints.

## Variables de entorno principales

- `RUN_MODE`: `preflight`, `smoke` o `full`. Default seguro: `preflight`.
- `PFI_RUN_ID`: identificador del run.
- `PFI_OUTPUT_ROOT`: carpeta externa para outputs.
- `RESUME_MODE`: `auto`, `off` o `required`.
- `AXIAL_RAW0_WEIGHT_BOOST`: boost configurable para `raw_0`.
- `ENABLE_SPATIAL_AUGMENTATION`: default `false`; no se aplican rotaciones espaciales por defecto.
- `ENABLE_HORIZONTAL_FLIP`: default `false`; solo habilitar tras revisar lateralidad axial.

## Orden recomendado en Colab

1. Ejecutar con `RUN_MODE=preflight`.
2. Revisar `reports/preflight_report.json` y `figures/preflight_ground_truth_overlay.png`.
3. Ejecutar `RUN_MODE=smoke` para validar entrenamiento corto, checkpoint, resume y round-trip.
4. Ejecutar `RUN_MODE=full` con GPU cuando preflight y smoke esten correctos.

## Mapeo raw

`250->background_250`, `0->raw_0`, `50->raw_50`, `100->raw_100`, `150->raw_150`, `200->raw_200`. Las clases `raw_*` no tienen traduccion anatomica validada.

## Preflight

Valida columnas E9, nulos antes de convertir a string, splits, pacientes, paths, labels permitidos, dimensiones imagen/mascara, presencia de clases en train, warnings para clases ausentes en val/test, forward pass y mosaico visual. Tambien guarda `manifests/split_snapshot.csv` como evidencia exacta del split usado.

El manifest esperado conserva los 610 registros del split E9: 427 train, 81 val y 102 test. El notebook no genera splits nuevos, no filtra registros, no excluye `raw_0` en bordes y no convierte labels automaticamente.

Hay seis pares conocidos del paciente `56` con imagen `(384, 384)` y mascara `(320, 320)`, misma relacion de aspecto y split `train`. El notebook los acepta como patron conocido, registra `sourceShapeMismatchCount`/`sourceShapeMismatches`/`sourceShapeWarnings`, y normaliza imagen y mascara a `(256, 256)` como el resto del pipeline. Cualquier otro mismatch de shape aborta.

La deteccion de duplicados no usa solo tamano de archivo: agrupa por `kind + fileSize`, calcula SHA-256 cuando hay tamanos repetidos y conserva `sliceId` en el reporte. Las reglas son:

- misma imagen SHA-256 en pacientes distintos o splits distintos: error;
- mismo par exacto imagen SHA-256 + mascara SHA-256 en pacientes distintos o splits distintos: error;
- misma mascara SHA-256 con imagenes diferentes: warning y registro como `repeatedMasks`/`maskOnlyWarnings`, no error;
- mismo tamano con SHA-256 distinto: no es duplicado.

## Outputs

El run escribe fuera del repo: `models`, `manifests`, `metrics`, `figures`, `predictions`, `logs`, `reports` y `RESUME_DIR`. No reemplaza `models/final`.

## Resume

El checkpoint `.last_checkpoint.pt` se escribe en `RESUME_DIR = PFI_RESUME_ROOT / PFI_RUN_ID`, separado de `PFI_OUTPUT_ROOT`. Guarda modelo, optimizer, scheduler, scaler, historia, patience, seed, modelKey, arquitectura, mapping, hash de split, preprocessing y run ID. En `auto`, si no existe o es incompatible se empieza desde cero con warning. En `required`, aborta si falta o no coincide.

Para reanudar una sesion distinta de Colab se debe reutilizar el mismo `PFI_RUN_ID` y mantener estable `PFI_RESUME_ROOT`, el split manifest y el preprocessing config. Si cambia el run ID o cualquiera de esas piezas no coincide, el notebook lo trata como otra corrida o aborta segun `RESUME_MODE`.

## Artifact final y candidato

El manifest y la model card se nombran desde el artifact real:

- `axial_t2_alkafri_final_best.pt.manifest.json` / `.modelcard.md` si el gate aprueba;
- `axial_t2_alkafri_final_candidate.pt.manifest.json` / `.modelcard.md` si el gate no aprueba.

El campo `artifactFile` coincide exactamente con `artifact_path.name`.

## Criterios de aceptacion

El gate tecnico principal es Dice macro foreground >= 0.70. Tambien se reportan Dice/IoU por clase, macro incluyendo y excluyendo `raw_0`, matriz de confusion y presencia/ausencia por clase.

## Limitaciones

No es diagnostico clinico, no valida semantica anatomica de `raw_*`, no publica artifacts y no debe usar test para ajustar hiperparametros.

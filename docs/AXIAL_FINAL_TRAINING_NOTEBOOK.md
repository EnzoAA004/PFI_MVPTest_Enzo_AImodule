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

## Variables de entorno principales

- `RUN_MODE`: `preflight`, `smoke` o `full`. Default seguro: `preflight`.
- `PFI_RUN_ID`: identificador del run.
- `PFI_OUTPUT_ROOT`: carpeta externa para outputs.
- `RESUME_MODE`: `auto`, `off` o `required`.
- `AXIAL_RAW0_WEIGHT_BOOST`: boost configurable para `raw_0`.

## Orden recomendado en Colab

1. Ejecutar con `RUN_MODE=preflight`.
2. Revisar `reports/preflight_report.json` y `figures/preflight_ground_truth_overlay.png`.
3. Ejecutar `RUN_MODE=smoke` para validar entrenamiento corto, checkpoint, resume y round-trip.
4. Ejecutar `RUN_MODE=full` con GPU cuando preflight y smoke esten correctos.

## Mapeo raw

`250->background_250`, `0->raw_0`, `50->raw_50`, `100->raw_100`, `150->raw_150`, `200->raw_200`. Las clases `raw_*` no tienen traduccion anatomica validada.

## Preflight

Valida columnas E9, nulos, splits, pacientes, duplicados, paths, labels permitidos, dimensiones imagen/mascara, presencia de clases en train, warnings para clases ausentes en val/test, forward pass y mosaico visual.

## Outputs

El run escribe fuera del repo: `models`, `resume`, `manifests`, `metrics`, `figures`, `predictions`, `logs`, `reports`. No reemplaza `models/final`.

## Resume

El checkpoint `.last_checkpoint.pt` guarda modelo, optimizer, scheduler, scaler, historia, patience, seed, modelKey, arquitectura, mapping, hash de split, preprocessing y run ID. En `auto`, un checkpoint incompatible se ignora con warning; en `required`, aborta.

## Criterios de aceptacion

El gate tecnico principal es Dice macro foreground >= 0.70. Tambien se reportan Dice/IoU por clase, macro incluyendo y excluyendo `raw_0`, matriz de confusion y presencia/ausencia por clase.

## Limitaciones

No es diagnostico clinico, no valida semantica anatomica de `raw_*`, no publica artifacts y no debe usar test para ajustar hiperparametros.

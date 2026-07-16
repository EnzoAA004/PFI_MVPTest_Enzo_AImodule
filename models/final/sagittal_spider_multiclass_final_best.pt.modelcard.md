# Model Card - sagittal_spider_multiclass_final_best.pt

## Identificacion

- Artifact: `sagittal_spider_multiclass_final_best.pt`
- Model key: `sagittal_spider`
- Version: `real-baseline-v1`
- Fecha de model card: 2026-07-16
- SHA-256: `7dd393cc750311c98003516d8110136310c31e8b6f0f00b6815f949fd61ef15b`
- Tamano local inspeccionado: 1,975,947 bytes
- Manifest: `models/final/sagittal_spider_multiclass_final_best.pt.manifest.json`

## Lineage / Origen

- Origen: notebook de Colab `20_E12_sagittal_final_training_clean.ipynb` / lineage E12 sagital final.
- Dataset: SPIDER sagittal lumbar MRI.
- Split de entrenamiento/validacion/test final: PENDIENTE de manifest versionado por paciente.
- Evaluacion preliminar indicada para QUAL-006: `held-out preliminary_gt_not_confirmed_unseen`.
- Riesgo conocido: el held-out preliminar NO esta confirmado como test limpio final por paciente.

## Arquitectura e hiperparametros

- Arquitectura runtime: `SagittalUNet2D`.
- `num_classes`: 4.
- `base_channels`: 16.
- `target_size`: 256 x 256.
- State dict: bajo key top-level `model_state_dict`.
- Metadata embebida confirmada en AI-001: `base_channels`, `num_classes`, `target_size`, `label_group_mapping`, `sagittal_axis`, `slice_strategy`, `val_dice_macro_no_bg`.

## Clases

- 0: `background`
- 1: `vertebra_group`
- 2: `canal`
- 3: `disc_group`

El checkpoint incluye `label_group_mapping` para remapear labels crudos SPIDER al espacio agrupado del modelo.

## Calidad actual documentada

Estado: PRELIMINAR, no final.

- Evaluador: QUAL-003b official evaluator.
- Split: `held-out preliminary_gt_not_confirmed_unseen`.
- Dice macro foreground preliminar: 0.625.
- IoU macro foreground preliminar: 0.511.
- `reliable`: true.
- Umbral objetivo: Dice macro foreground >= 0.70.
- Estado frente al umbral: por debajo del umbral.

Aclaracion obligatoria: estas metricas NO son test limpio final. El conjunto held-out aun no esta confirmado por paciente, por lo que no debe presentarse como resultado final de calidad.

## Metricas legacy en manifest

El manifest conserva metricas historicas internas bajo `metrics` con status `legacy_not_final_quality_gate`. No reemplazan la evaluacion final requerida por QUAL-004/QUAL-005.

## Uso previsto y restricciones

- Uso: prototipo academico para segmentacion asistida de RM lumbar sagital y generacion de mascaras revisables.
- No emite diagnosticos.
- No recomienda tratamientos.
- Requiere revision profesional humana.
- No debe usarse como evidencia clinica final sin quality gate sobre test held-out limpio.

## Pendientes

- Congelar y versionar test held-out limpio por paciente.
- Ejecutar QUAL-003b sobre el test limpio final.
- Alcanzar o justificar brecha frente a Dice macro foreground >= 0.70.
- Documentar split train/val/test definitivo con IDs no sensibles.
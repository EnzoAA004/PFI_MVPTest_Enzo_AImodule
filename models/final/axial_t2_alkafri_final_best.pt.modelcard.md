# Model Card - axial_t2_alkafri_final_best.pt

## Identificacion

- Artifact: `axial_t2_alkafri_final_best.pt`
- Model key: `axial_t2_alkafri`
- Version: `real-baseline-v1`
- Fecha de model card: 2026-07-16
- SHA-256: `a8b91f563e101c9c4a3cf2c0ec84af29cbcfabd76eb76b2d06dbd13f7d7c78d6`
- Tamano local inspeccionado: 1,973,243 bytes
- Manifest: `models/final/axial_t2_alkafri_final_best.pt.manifest.json`

## Lineage / Origen

- Origen: notebook de Colab axial E10 / `AxialUNet2D` final segun runtime y backlog.
- Dataset: ALKAFRI/Sudirman axial T2 lumbar MRI.
- Split de entrenamiento/validacion/test final: PENDIENTE de manifest versionado por paciente.
- Evaluacion preliminar indicada para QUAL-006: `held-out preliminary_gt_not_confirmed_unseen`.
- Riesgo conocido: el held-out preliminar NO esta confirmado como test limpio final por paciente.

## Arquitectura e hiperparametros

- Arquitectura runtime: `AxialUNet2D`.
- `num_classes`: 6.
- `base_channels`: 16.
- `target_size`: 256 x 256, confirmado por dato externo QUAL-006.
- State dict: checkpoint crudo, sin wrapper `model_state_dict`.
- Metadata ausente en checkpoint: `target_size`, `label_group_mapping`, config de entrenamiento completa.

## Clases

- 0: `background_250`
- 1: `raw_0`
- 2: `raw_50`
- 3: `raw_100`
- 4: `raw_150`
- 5: `raw_200`

PENDIENTE: confirmar definicion anatomica/semantica de las clases `raw_*`, en especial resolver `raw_0` antes de usar macro como indicador confiable.

## Calidad actual documentada

Estado: PRELIMINAR, no final.

- Evaluador: QUAL-003b official evaluator.
- Split: `held-out preliminary_gt_not_confirmed_unseen`.
- Dice macro foreground preliminar: 0.344.
- IoU macro foreground preliminar: PENDIENTE/no provisto.
- `reliable`: false.
- Motivo de no confiabilidad: `raw_0 gt_present_cases=0`.
- Dice macro util excluyendo `raw_0`: aproximadamente 0.43.
- Umbral objetivo: Dice macro foreground >= 0.70.
- Estado frente al umbral: por debajo del umbral y pendiente de resolver mapping/clase `raw_0`.

Aclaracion obligatoria: estas metricas NO son test limpio final. El conjunto held-out aun no esta confirmado por paciente, por lo que no debe presentarse como resultado final de calidad.

## Metricas legacy en manifest

El manifest conserva metricas historicas bajo `metrics` con status `legacy_not_final_quality_gate`. No reemplazan la evaluacion final requerida por QUAL-004/QUAL-005.

## Uso previsto y restricciones

- Uso: prototipo academico para segmentacion asistida de RM lumbar axial y generacion de mascaras revisables.
- Modulo axial tratado como complementario/spike tecnico dentro del MVP.
- No emite diagnosticos.
- No recomienda tratamientos.
- Requiere revision profesional humana.
- No debe usarse como evidencia clinica final sin quality gate sobre test held-out limpio.

## Pendientes

- Resolver `raw_0` y confirmar mapping/semantica de clases axiales.
- Congelar y versionar test held-out limpio por paciente.
- Ejecutar QUAL-003b sobre el test limpio final.
- Alcanzar o justificar brecha frente a Dice macro foreground >= 0.70.
- Documentar split train/val/test definitivo con IDs no sensibles.
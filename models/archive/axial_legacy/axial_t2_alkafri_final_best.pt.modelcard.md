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
- Evaluaciones preliminares documentadas: E10 sobre test curado y QUAL-004b sobre `pairing_v1`.
- Riesgo conocido: el numero de gate se tomara del split curado E9 por paciente; las metricas actuales son contexto preliminar.

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

Definicion actual: clases discretas del dataset axial (`background_250`, `raw_0`, `raw_50`, `raw_100`, `raw_150`, `raw_200`) sin traduccion anatomica cerrada. NO se inventa equivalencia anatomica para `raw_*`; queda PENDIENTE. En el test curado E10, `raw_0` esta presente aproximadamente en 16-21% de los casos, por lo que no debe tratarse como clase ausente global.

## Calidad actual documentada

Estado: PRELIMINAR, no final.

### E10 sobre test curado

- Evaluador: QUAL-003b official evaluator / contexto E10 axial.
- Split: test curado E10, con identidad de labels preservada.
- Dice macro foreground (`dice_macro_no_bg`): 0.659.
- Dice macro foreground excluyendo `raw_0` (`dice_macro_excluding_raw0`): 0.817.
- Dice de `raw_0`: 0.026.
- Presencia de `raw_0`: aproximadamente 16-21% de los casos.
- Umbral objetivo: Dice macro foreground >= 0.70.
- Estado frente al umbral: 0.659 queda por debajo del umbral global; excluyendo `raw_0` supera el umbral, pero `raw_0` requiere resolucion especifica.

### QUAL-004b preliminar sobre pairing_v1

- Dice macro foreground preliminar: 0.344.
- `reliable`: false.
- Motivo de no confiabilidad: `raw_0 gt_present_cases=0` en pairing_v1.
- Interpretacion: ese 0.344 no debe usarse como calidad axial final; evidencia un problema de split/mapping/presencia de `raw_0` en pairing_v1.

Aclaracion obligatoria: el numero de gate se tomara del split curado E9 por paciente, con identidad de labels. Hasta congelar ese split, estas metricas son preliminares y NO son test limpio final.

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

- Resolver desempeno de `raw_0` y confirmar semantica anatomica de clases axiales `raw_*` sin inventar traducciones.
- Congelar y versionar test held-out limpio por paciente.
- Ejecutar QUAL-003b sobre el test limpio final.
- Alcanzar o justificar brecha frente a Dice macro foreground >= 0.70.
- Documentar split train/val/test definitivo con IDs no sensibles.
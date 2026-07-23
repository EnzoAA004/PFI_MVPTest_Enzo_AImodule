# axial-full-v1 summary

## Estado

Este documento preserva la evidencia resumida del baseline axial `axial-full-v1`. No es diagnostico clinico y no reemplaza revision profesional. El artifact generado fue candidate, no validated baseline.

## Configuracion

- Run ID: `axial-full-v1`
- Max epochs configurado: 80
- Epoca final: 38
- Mejor checkpoint: epoca 26
- Early stopping patience: 12
- Monitor de seleccion: `val_dice_macro_excluding_raw0`
- AXIAL_RAW0_WEIGHT_BOOST: 3.0
- Peso final raw_0 aproximado: 3.5526
- Peso background aproximado: 0.1474
- Ratio max/min aproximado: 24.10
- Arquitectura: `AxialUNet2D`, 6 clases, base_channels=16, target_size=(256, 256)

## Dataset y split

Split E9 por paciente para ALKAFRI/Sudirman axial T2. El test held-out ya fue evaluado en este baseline, por lo que futuras comparaciones v2 no deben presentarse como evaluacion externa completamente no observada.

## Validacion

- bestValidationMetric: 0.88268009952954
- Metrica monitorizada: `val_dice_macro_excluding_raw0`

## Test held-out

- dice_macro_foreground: 0.6827304611400741
- iou_macro_foreground: 0.5750633478983105
- dice_macro_excluding_raw0: 0.8149698000695298
- iou_macro_excluding_raw0: 0.6980065674207926
- raw0Dice: 0.15377310542225076
- raw0Iou: 0.08329046980838194

## Quality gate

- Threshold Dice macro foreground: 0.70
- qualityGatePassed: false
- Razon: `dice_macro_foreground_below_threshold`

## Artifact

- `axial_t2_alkafri_final_candidate.pt`
- Runtime shape validado: `[1, 6, 256, 256]`
- Estado: candidate, no validated baseline.

## Analisis raw_0

`raw_0` aparece en ground truth de 21/102 casos de test, pero fue predicho en 102/102. Matriz aproximada:

- true positives raw_0: 11.497
- false positives raw_0: 122.371
- false negatives raw_0: 4.167
- precision raw_0 aproximada: 0.0859
- recall raw_0 aproximado: 0.7340

Hipotesis para v2: reducir `AXIAL_RAW0_WEIGHT_BOOST` de 3.0 a 1.0 y seleccionar por `dice_macro_foreground`, alineando monitor, scheduler, early stopping y quality gate.

## Limitaciones

La semantica anatomica definitiva de `raw_*`, especialmente `raw_0`, sigue pendiente. No inventar traducciones anatomicas. No usar este resultado como diagnostico clinico.

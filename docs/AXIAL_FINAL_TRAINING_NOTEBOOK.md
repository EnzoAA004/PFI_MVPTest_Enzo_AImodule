# Notebook de entrenamiento axial final

## Objetivo

`notebooks/48_axial_final_training_patient_split.ipynb` prepara el entrenamiento final reproducible del modelo axial `axial_t2_alkafri` con split por paciente. Esta tarea no ejecuta entrenamiento ni reemplaza artifacts.

## Baseline historico

El artifact historico `axial_t2_alkafri_final_best.pt` queda intacto. Sus metricas son preliminares y sirven solo como contexto, no como resultado final nuevo.

## Fuentes de datos

Se requiere el dataset ALKAFRI/Sudirman axial T2 y el manifest E9 `E9_t2_final_labels_curated_split.csv` con columnas `image_file_path`, `final_label_file_path`, `case_id_norm` y `split`.

## Mapeo raw

El notebook usa `250->background_250`, `0->raw_0`, `50->raw_50`, `100->raw_100`, `150->raw_150`, `200->raw_200`. No inventa traduccion anatomica para `raw_*`.

## Modos

- `preflight`: default; valida entorno, rutas, dataset, labels, split, duplicados y forward.
- `smoke`: maximo dos epochs con subconjunto deterministico y artifacts `smoke_only=true`.
- `full`: entrenamiento completo, reanudable, selection por validation, test held-out una sola vez y quality gate.

## Uso en Colab

Abrir el notebook, configurar Drive si las rutas difieren, ejecutar primero `RUN_MODE=preflight`, luego `RUN_MODE=smoke` y recien despues `RUN_MODE=full` con GPU.

## Resume

`RESUME_MODE=auto` reanuda desde `resume/axial_t2_alkafri_final.last_checkpoint.pt` si el checkpoint coincide con modelKey, arquitectura y label mapping.

## Outputs

Los outputs se escriben fuera del repositorio bajo `OUTPUT_ROOT/RUN_ID/`: `models`, `resume`, `manifests`, `metrics`, `figures`, `predictions`, `logs` y `reports`.

## Quality gate

El umbral objetivo es Dice macro foreground >= 0.70, con IoU disponible, reporte explicito de `raw_0`, split held-out por paciente, cero leakage, revision humana y flag de no diagnostico clinico.

## GCP y publicacion

El notebook no usa GCP en esta etapa inicial y no publica artifacts. La publicacion controlada queda para otro ticket.

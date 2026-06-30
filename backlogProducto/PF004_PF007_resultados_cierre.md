# Cierre PF-004 a PF-007 - Modelos finales, registry y métricas

Estado: cerrado.

## Objetivo

Consolidar los artefactos finales de modelos para el producto final, crear una configuración única de modelos y registrar las métricas principales que deberán utilizar los próximos módulos productivos de inferencia.

## Resultado principal

El bloque cerró correctamente con la decisión:

```text
PF004_PF007_final_models_ready_for_product_inference
```

## Modelos consolidados

### axial_t2_alkafri

- plano: axial
- etapa fuente: E10
- dataset: alkafri_sudirman_axial
- estado: final_checkpoint_consolidated
- checkpoint final: `/content/drive/MyDrive/PFI_MVP/models/final/axial_t2_alkafri_final_best.pt`
- existe: true
- tamaño: 1.973.243 bytes
- sha256 registrado: true

### sagittal_spider

- plano: sagittal
- etapa fuente: E12
- dataset: spider_sagittal
- estado: final_checkpoint_consolidated
- checkpoint final: `/content/drive/MyDrive/PFI_MVP/models/final/sagittal_spider_multiclass_final_best.pt`
- existe: true
- tamaño: 1.975.947 bytes
- sha256 registrado: true

## Métricas finales principales

### Axial T2 Al-Kafri/Sudirman

- validación macro no background incluyendo raw_0: Dice 0.7054, IoU 0.6364
- validación clases útiles excluyendo raw_0: Dice 0.8817, IoU 0.7955
- test macro no background incluyendo raw_0: Dice 0.6587, IoU 0.5628
- test clases útiles excluyendo raw_0: Dice 0.8167, IoU 0.7001

La métrica recomendada para comunicación del modelo axial es la de clases útiles excluyendo raw_0, dado que raw_0 fue documentada previamente como clase minoritaria/intermitente en E11.

### Sagital SPIDER

- validación macro no background: Dice 0.8392, IoU 0.7466
- holdout/test macro no background: Dice 0.8311, IoU 0.7309

## Métricas por clase destacadas

### Axial test

- background: Dice 0.99296
- raw_0: Dice 0.02640
- raw_50: Dice 0.93480
- raw_100: Dice 0.84850
- raw_150: Dice 0.79970
- raw_200: Dice 0.68400

### Sagital holdout/test

- background: Dice 0.97450
- vertebra_group: Dice 0.85160
- canal: Dice 0.86450
- disc_group: Dice 0.79780

## Checks

Todos los checks dieron OK:

- data_freeze_config_exists: true
- axial_final_checkpoint_exists: true
- sagittal_final_checkpoint_exists: true
- artifacts_csv_written: true
- metrics_csv_written: true
- class_metrics_csv_written: true
- registry_results_written: true
- registry_repo_written: true
- docs_repo_written: true
- all_artifacts_have_sha256: true
- only_final_scope_datasets_in_metrics: true

## Archivos generados

- `models/final/axial_t2_alkafri_final_best.pt`
- `models/final/sagittal_spider_multiclass_final_best.pt`
- `config/model_registry_final.json`
- `docs/PF004_PF007_final_models_metrics.md`
- `results/PF004_PF007_final_models/PF004_PF007_model_artifacts.csv`
- `results/PF004_PF007_final_models/PF006_model_registry_final.json`
- `results/PF004_PF007_final_models/PF007_final_model_metrics.csv`
- `results/PF004_PF007_final_models/PF007_final_model_metrics_by_class.csv`
- `results/PF004_PF007_final_models/PF004_PF007_checks.csv`
- `results/PF004_PF007_final_models/PF004_PF007_report.json`

## Política metodológica

- human_review_required: true
- not_clinical_diagnosis: true
- not_real_3d_reconstruction: true
- optional_datasets_excluded_from_final_metrics: true

## Próximo bloque

PF-008 a PF-011:

- convertir la inferencia E13 en módulos productivos,
- soportar entrada de archivo,
- generar overlays finales,
- calcular mediciones geométricas mínimas.

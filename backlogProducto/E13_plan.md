# E13 - Pipeline comun de inferencia multiplanar

Estado: En progreso.

## Objetivo

Construir el primer pipeline comun de inferencia para utilizar los dos modelos finales del proyecto:

- Modelo sagital SPIDER consolidado en E12.
- Modelo axial T2 Al-Kafri/Sudirman entrenado en E10.

## Alcance

E13 no entrena modelos. Estandariza inferencia, preprocesamiento, overlays, metricas por clase cuando hay ground truth y quality flags.

## Entradas esperadas

- models/E12_sagittal_multiclass_final_best.pt
- models/E10_axial_t2_final_training_clean_best.pt
- results/E9_alkafri_axial_t2_final_labels_baseline/E9_t2_final_labels_curated_split.csv
- results/E5_multiclase_holdout/E5_multiclass_holdout_selected_cases.csv

## Salidas esperadas

- results/E13_multiplanar_inference_pipeline/E13_multiplanar_pipeline_report.json
- results/E13_multiplanar_inference_pipeline/E13_axial_examples_quality.csv
- results/E13_multiplanar_inference_pipeline/E13_axial_examples_metrics_by_class.csv
- results/E13_multiplanar_inference_pipeline/E13_sagittal_examples_quality.csv
- results/E13_multiplanar_inference_pipeline/E13_sagittal_examples_metrics_by_class.csv
- docs/E13_multiplanar_inference_pipeline_conclusion.md
- figures/E13_axial_t2_example_*.png
- figures/E13_sagittal_example_*.png

## Criterios de aceptacion

- Carga correcta de ambos checkpoints.
- Inferencia axial funcional.
- Inferencia sagital funcional o reporte claro si algun path no puede resolverse.
- Salida comun con plane, model_key, pred, confidence y quality flags.
- Reporte final E13 generado.

## Proximo paso posterior

E14: agente/orquestador IA que detecte el plano, seleccione modelo y genere reporte automatico.

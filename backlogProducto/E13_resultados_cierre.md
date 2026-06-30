# Cierre E13 - Pipeline comun de inferencia multiplanar

Estado: Hecho experimentalmente.

## Resultado principal

E13 integro correctamente los dos modelos finales disponibles:

- Sagital SPIDER: models/E12_sagittal_multiclass_final_best.pt
- Axial T2 Al-Kafri/Sudirman: models/E10_axial_t2_final_training_clean_best.pt

El pipeline comun genera inferencia 2D, overlays, metricas por clase y quality flags para ambos planos.

## Axial T2

Se procesaron 6 ejemplos axiales del split test:

- foreground_ratio aproximado: 0.058 a 0.073
- mean_confidence aproximada: 0.989
- mean_fg_confidence aproximada: 0.943 a 0.968
- clases presentes: [1, 2, 3, 4, 5]

Solo un ejemplo marco flag muchos_componentes, pero visualmente los overlays son coherentes y alineados con la anatomia axial.

## Sagital SPIDER

Se procesaron 6 ejemplos sagitales del holdout con deteccion dinamica de eje:

- 101_t1: selected_axis 2, slice 8, shape (298, 320, 17)
- 116_t1: selected_axis 2, slice 5, shape (320, 320, 15)
- 117_t1: selected_axis 2, slice 13, shape (448, 448, 24)
- 12_t1: selected_axis 2, slice 8, shape (320, 320, 15)
- 131_t1: selected_axis 2, slice 12, shape (463, 448, 24)
- 136_t1: selected_axis 0, slice 8, shape (17, 512, 512)

La deteccion dinamica corrigio el problema observado previamente con el caso 136_t1, donde axis=2 fijo generaba una imagen deformada.

## Metricas sagitales por clase en ejemplos

Los ejemplos sagitales muestran desempeno coherente:

- 101_t1: vertebra 0.9035, canal 0.9026, disc 0.8296
- 116_t1: vertebra 0.8590, canal 0.8212, disc 0.7077
- 117_t1: vertebra 0.8243, canal 0.7588, disc 0.5993
- 12_t1: vertebra 0.9410, canal 0.9109, disc 0.9179
- 131_t1: vertebra 0.9003, canal 0.9237, disc 0.8834
- 136_t1: vertebra 0.6925, canal 0.7514, disc 0.6934

## Decision metodologica

El pipeline E13 queda cerrado como prototipo de inferencia multiplanar. No es reconstruccion 3D todavia; integra ambos modelos 2D en una salida comun y deja lista la base para E14, el agente/orquestador IA.

## Salidas generadas

- results/E13_multiplanar_inference_pipeline/E13_multiplanar_pipeline_report.json
- results/E13_multiplanar_inference_pipeline/E13_axial_examples_quality.csv
- results/E13_multiplanar_inference_pipeline/E13_axial_examples_metrics_by_class.csv
- results/E13_multiplanar_inference_pipeline/E13_sagittal_examples_quality.csv
- results/E13_multiplanar_inference_pipeline/E13_sagittal_examples_metrics_by_class.csv
- docs/E13_multiplanar_inference_pipeline_conclusion.md
- figures/E13_axial_t2_example_*.png
- figures/E13_sagittal_dynamic_example_*.png

## Proximo paso recomendado

E14 - Agente/orquestador IA: detectar plano, elegir modelo, ejecutar pipeline E13 y generar reporte automatico con quality flags para revision profesional.

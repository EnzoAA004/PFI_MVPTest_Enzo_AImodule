# Cierre E11 - Mapeo de clases axiales y reporte final

Estado: Hecho a nivel experimental en Colab.

## Resultado principal

El notebook E11 genero la documentacion tecnica del modulo axial T2 a partir de los resultados E8, E9 y E10.

## Dataset usado

- Pares T2 curados: 610
- Train: 427 imagenes / 128 casos
- Val: 81 imagenes / 27 casos
- Test: 102 imagenes / 29 casos

## Distribucion de clases

- background_250: mean_ratio 0.937816, presencia 610/610
- raw_0: mean_ratio 0.002401, mediana 0.000000, presencia 109/610
- raw_50: mean_ratio 0.035394, presencia 610/610
- raw_100: mean_ratio 0.015679, presencia 610/610
- raw_150: mean_ratio 0.003173, presencia 610/610
- raw_200: mean_ratio 0.005536, presencia 610/610

## Interpretacion raw_0

raw_0 queda confirmado como clase minoritaria e intermitente. Esta presente solo en 109 de 610 imagenes, con mediana 0 y proporcion media de 0.24 por ciento por imagen. El problema no parece ser un desbalance entre splits, ya que la presencia es similar en train, val y test.

## Metricas E10 documentadas

- VAL Dice macro sin fondo: 0.705371
- TEST Dice macro sin fondo: 0.658684
- VAL Dice macro excluyendo raw_0: 0.881714
- TEST Dice macro excluyendo raw_0: 0.816746

## Decision metodologica

Para el informe se deben reportar ambas lecturas: con raw_0 y excluyendo raw_0. raw_0 no se elimina de la trazabilidad, pero se interpreta como clase minoritaria problematica. Las clases principales para lectura de desempeno son raw_50, raw_100, raw_150 y raw_200.

## Salidas generadas en Drive

- results/E11_axial_class_mapping_final_report/E11_axial_class_mapping.csv
- results/E11_axial_class_mapping_final_report/E11_axial_class_distribution_long.csv
- results/E11_axial_class_mapping_final_report/E11_axial_class_distribution_summary.csv
- results/E11_axial_class_mapping_final_report/E11_axial_class_distribution_by_split.csv
- results/E11_axial_class_mapping_final_report/E11_axial_metrics_summary.csv
- results/E11_axial_class_mapping_final_report/E11_axial_metrics_by_class.csv
- results/E11_axial_class_mapping_final_report/E11_axial_final_report.json
- docs/E11_axial_class_mapping_final_report.md
- figures/E11_axial_class_pixel_distribution.png
- figures/E11_axial_class_presence_by_split.png
- figures/E11_axial_dice_by_class_val_test.png
- figures/E11_axial_iou_by_class_val_test.png

## Proximo paso recomendado

Actualizar el notebook 19 en GitHub con la version ejecutada desde Colab y luego avanzar a E12: entrenamiento sagital final limpio.

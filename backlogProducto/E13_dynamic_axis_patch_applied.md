# E13 - Dynamic axis patch aplicado al notebook local

Estado: notebook corregido generado fuera del repo por bloqueo de escritura directa de .ipynb.

## Cambio aplicado

Se reemplazo la logica sagital fija axis=2 por inferencia sagital con deteccion dinamica de eje por caso.

## Razon

El caso 136_t1 tiene shape (17, 512, 512). En ese caso, axis=2 genera cortes 17x512 deformados. El eje correcto para stack sagital es axis=0. En cambio, otros casos como (298, 320, 17), (320, 320, 15) o (448, 448, 24) siguen usando axis=2.

## Nueva regla

- Leer volumen en formato z,y,x sin transponer.
- Probar ejes 0, 1 y 2.
- Descartar ejes cuyo corte 2D tenga una dimension menor a 128 px.
- Elegir el eje valido con menor numero de slices.
- Seleccionar el slice por score de foreground del modelo, sin usar ground truth.

## Archivos generados por el notebook corregido

- figures/E13_sagittal_dynamic_example_*.png
- results/E13_multiplanar_inference_pipeline/E13_sagittal_examples_quality.csv
- results/E13_multiplanar_inference_pipeline/E13_sagittal_examples_metrics_by_class.csv
- results/E13_multiplanar_inference_pipeline/E13_multiplanar_pipeline_report.json
- docs/E13_multiplanar_inference_pipeline_conclusion.md

## Nota operativa

La herramienta bloqueo el update directo del archivo .ipynb en GitHub, por lo que el notebook corregido debe subirse manualmente desde Colab o reemplazarse en la ruta:

notebooks/21_E13_multiplanar_inference_pipeline.ipynb

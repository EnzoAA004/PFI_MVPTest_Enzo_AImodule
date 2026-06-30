# E12 plan

Estado: En progreso.

Objetivo: cerrar el modulo sagital con un reporte final limpio y comparable contra el modulo axial.

Resultados previos sagitales:
- Binario mejorado: Val Dice 0.8875; Holdout Dice 0.8816.
- Multiclase agrupado: Val Dice macro sin fondo 0.8392; Holdout Dice macro sin fondo 0.8311.

Tareas:
- Cargar insumos SPIDER ya preprocesados.
- Consolidar la estrategia multiclase agrupada.
- Cargar o reentrenar el checkpoint final sagital.
- Evaluar metricas por clase.
- Generar figuras cualitativas.
- Generar reporte JSON y Markdown.

Salidas esperadas:
- results/E12_sagittal_final_training_clean/E12_final_report.json
- docs/E12_sagittal_final_training_clean_conclusion.md
- figures/E12_sagittal_prediction_*.png
- models/E12_sagittal_multiclass_final_best.pt

Decision: si el checkpoint E5 ya es el mejor validado, E12 puede actuar como consolidacion final del modelo sagital.

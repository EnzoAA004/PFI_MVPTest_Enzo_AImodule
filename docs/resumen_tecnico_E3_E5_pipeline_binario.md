# Resumen técnico E3-E5: pipeline binario de segmentación sagital

## Objetivo del documento

Este documento consolida los avances técnicos realizados entre las etapas E3 y E5 del proyecto, desde la exploración inicial del dataset SPIDER hasta la validación holdout interna y la evaluación de estrategias de selección de slice sin máscara.

El alcance corresponde a una formulación simplificada y experimental del problema: imágenes T1 del dataset SPIDER, segmentación binaria 2D, plano sagital, un slice representativo por volumen y evaluación interna. Esta evidencia permite documentar un baseline binario reproducible antes de avanzar a segmentación multiclase, evaluación axial, inferencia 3D, nnU-Net o integración backend.

## Dataset y alcance

El trabajo se desarrolló sobre el dataset SPIDER. En la exploración inicial se detectaron 447 pares imagen/máscara en formato `.mha`. Luego del análisis de consistencia y la exclusión inicial de casos `t2_SPACE`, se consolidaron 406 candidatos válidos para el baseline.

En esta fase se utilizó la modalidad T1 para estabilizar el pipeline experimental. La segmentación se formuló como binaria, usando `mask > 0` como foreground anatómico y `mask == 0` como fondo. El eje sagital utilizado fue el eje 2 y los slices se redimensionaron a `256x256`.

El alcance no incluye segmentación multiclase, evaluación axial, nnU-Net, inferencia 3D, validación clínica externa ni integración con backend.

## Resumen por notebook

### `01_E3_E4_exploracion_dataset_visualizacion.ipynb`

**Objetivo:** realizar la primera exploración del dataset SPIDER, validar lectura de imágenes y máscaras, inspeccionar metadatos y generar evidencias visuales.

**Principales salidas:**

- Detección de 447 imágenes y 447 máscaras.
- Lectura correcta de archivos `.mha` con SimpleITK.
- Inspección de shape, spacing, dtype y labels.
- Generación de overlay imagen/máscara.
- Exportación de PNG de evidencia, CSV resumen y conclusión técnica.
- Identificación preliminar del eje 2 como el más representativo para visualización sagital en el caso evaluado.

**Archivos generados:** figuras de overlay, CSV resumen del dataset y conclusión técnica Markdown.

**Conclusión técnica:** el dataset quedó accesible y técnicamente legible desde Colab/Drive. La exploración confirmó compatibilidad básica entre imagen y máscara en el caso inspeccionado y permitió fijar una primera hipótesis de trabajo sobre el eje sagital.

### `02_E4_preprocesamiento_normalizacion.ipynb`

**Objetivo:** preparar el dataset SPIDER para una etapa posterior de baseline sagital, cerrando consistencia, normalización y preparación de casos.

**Principales salidas:**

- Detección de 447 pares imagen/máscara.
- Verificación de lectura sin errores.
- Confirmación de que los 447 pares tienen mismo shape, spacing y origin.
- Identificación de 14 casos con `same_direction = False`, asociados a secuencias `t2_SPACE`.
- Generación de candidatos para baseline inicial excluyendo casos `SPACE`.
- Exportación de un ejemplo `.npz` con imagen, máscara, slice sagital y metadata.
- Confirmación visual del eje 2 como vista sagital representativa en el caso de ejemplo.

**Archivos generados:**

- `/content/drive/MyDrive/PFI_MVP/results/E4_preprocesamiento/E4_spider_consistency_summary.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E4_preprocesamiento/E4_baseline_candidates_no_space.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E4_preprocesamiento/E4_preprocessed_example_100_t1.npz`

**Conclusión técnica:** el preprocesamiento permitió consolidar casos válidos, detectar inconsistencias relevantes y definir una normalización robusta por percentiles p1-p99. La exclusión inicial de `t2_SPACE` quedó justificada para estabilizar la primera fase del baseline.

### `03_E5_baseline_segmentacion_sagital.ipynb`

**Objetivo:** implementar un primer baseline binario 2D sobre slices sagitales de SPIDER para validar el flujo entrenamiento-inferencia-métricas-visualización.

**Principales salidas:**

- Uso de 20 casos T1.
- Split: 16 casos train y 4 casos validation.
- Tensores de entrada `[B, 1, 256, 256]`.
- Máscaras binarias con `mask > 0`.
- Modelo U-Net 2D simple en PyTorch.
- Entrenamiento de 5 epochs en CPU.
- Exportación de métricas, curva de loss, predicción PNG, modelo `.pt` y conclusión Markdown.

**Archivos generados:**

- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_training_history.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_baseline_metrics.csv`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_training_loss_curve.png`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_baseline_prediction_88_t1.png`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_simple_unet2d_baseline.pt`
- `/content/drive/MyDrive/PFI_MVP/docs/E5_baseline_segmentacion_sagital_conclusion.md`

**Conclusión técnica:** el pipeline completo funcionó, pero el rendimiento fue bajo. El Dice de validación con threshold 0.5 quedó en 0.2364 y el IoU en 0.1344, lo que motivó un diagnóstico específico antes de modificar el modelo.

### `04_E5_diagnostico_baseline_binario.ipynb`

**Objetivo:** diagnosticar el comportamiento del baseline inicial sin reentrenar, analizando logits, probabilidades, thresholds y proporción de foreground.

**Principales salidas:**

- Validación reconstruida de 4 casos.
- Dice promedio con threshold 0.5: 0.2364.
- IoU promedio con threshold 0.5: 0.1344.
- Mejor threshold por Dice e IoU: 0.7.
- Dice promedio con threshold 0.7: aproximadamente 0.3148.
- IoU promedio con threshold 0.7: aproximadamente 0.1885.
- Foreground promedio en ground truth: 0.1344.
- Foreground predicho con threshold 0.5: 1.0000.
- Diagnóstico automático: posible colapso a foreground, sobreajuste probable, Dice de validación constante e influencia relevante del threshold.

**Archivos generados:**

- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_diagnostic_probability_stats.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_threshold_diagnostics.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_threshold_summary.csv`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_threshold_diagnostic_example.png`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_probability_histogram.png`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_diagnostic_report.json`
- `/content/drive/MyDrive/PFI_MVP/docs/E5_diagnostico_baseline_binario_conclusion.md`

**Conclusión técnica:** el baseline inicial no fallaba por el pipeline de datos, sino por un comportamiento de predicción mal calibrado y colapsado hacia foreground. El diagnóstico justificó aumentar datos, épocas y usar una pérdida con manejo explícito del desbalance.

### `05_E5_baseline_mejorado_binario.ipynb`

**Objetivo:** entrenar un baseline binario 2D mejorado, manteniendo la formulación sagital simplificada pero corrigiendo las limitaciones del baseline inicial.

**Principales salidas:**

- Uso de 100 casos T1.
- Split: 80 train y 20 validation.
- Entrenamiento de 20 epochs en CPU.
- Foreground ratio train: 0.1810.
- `pos_weight`: 4.5256.
- Pérdida: `BCEWithLogitsLoss(pos_weight)` + Dice loss.
- Mejor threshold: 0.5.
- Dice validation threshold 0.5: 0.8875.
- IoU validation threshold 0.5: 0.7988.
- Pred foreground ratio th 0.5: 0.2161.
- GT foreground ratio validation: 0.2001.
- Diagnóstico: sin colapso foreground evidente en threshold 0.5.

**Archivos generados:**

- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_selected_cases.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_split.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_training_history.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_threshold_diagnostics.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_threshold_summary.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_validation_report.json`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_unet2d_binary_best.pt`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_improved_prediction_example.png`
- `/content/drive/MyDrive/PFI_MVP/docs/E5_baseline_mejorado_binario_conclusion.md`

**Conclusión técnica:** el baseline mejorado corrigió el colapso a foreground y logró un desempeño alto en validación interna para la formulación binaria 2D. La mejora se asocia al uso de más casos, más épocas y pérdida ponderada por desbalance.

### `06_E5_validacion_holdout_y_sanity_checks.ipynb`

**Objetivo:** validar el modelo mejorado sobre un holdout interno de casos T1 no utilizados en el entrenamiento ni en la validación interna del notebook 05.

**Principales salidas:**

- Candidatos T1: 196.
- Casos usados en notebook 05: 100.
- Casos disponibles para holdout: 96.
- Casos evaluados en holdout: 40.
- Mejor threshold holdout: 0.5.
- Dice holdout threshold 0.5: 0.8816.
- IoU holdout threshold 0.5: 0.7904.
- Foreground GT promedio threshold 0.5: 0.1764.
- Foreground predicho promedio threshold 0.5: 0.1910.
- Sin predicciones vacías ni llenas.
- 0% de casos con `pred_foreground_ratio > 0.8`.
- 0% de casos con `pred_foreground_ratio < 0.01`.

**Archivos generados:**

- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_selected_cases.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_metrics_by_case.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_threshold_summary.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_sanity_checks.json`
- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_vs_internal_validation.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_validation_report.json`
- `/content/drive/MyDrive/PFI_MVP/docs/E5_holdout_validacion_conclusion.md`

**Conclusión técnica:** la validación holdout interna mostró un desempeño cercano a la validación interna. Esto sugiere que el baseline binario mejorado generaliza razonablemente dentro de SPIDER para la formulación T1, 2D, binaria y sagital.

### `07_E5_evaluacion_slices_sin_mascara.ipynb`

**Objetivo:** evaluar el impacto de reemplazar la selección de slice basada en máscara por estrategias que no dependan de la máscara, manteniendo el modelo mejorado sin reentrenar.

**Principales salidas:**

- Holdout evaluado: 40 casos T1.
- Threshold: 0.5.
- Estrategias evaluadas:
  - `mask_max_area`, referencia con máscara.
  - `central_slice`, sin máscara.
  - `center_window_best_prediction`, sin máscara.
  - `all_slices_best_prediction`, sin máscara.
  - `top_k_prediction_mean`, sin máscara.
- Mejor estrategia sin máscara: `center_window_best_prediction`.
- Dice `mask_max_area`: 0.8816.
- IoU `mask_max_area`: 0.7904.
- Dice `center_window_best_prediction`: 0.8575.
- IoU `center_window_best_prediction`: 0.7631.
- Caída absoluta de Dice: -0.0241.
- Caída relativa de Dice: -2.73%.
- Caída absoluta de IoU: -0.0273.

**Archivos generados:**

- `/content/drive/MyDrive/PFI_MVP/results/E5_slice_selection_eval/E5_slice_selection_metrics_by_case.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_slice_selection_eval/E5_slice_selection_summary.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_slice_selection_eval/E5_slice_selection_vs_mask_reference.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_slice_selection_eval/E5_slice_selection_report.json`
- `/content/drive/MyDrive/PFI_MVP/docs/E5_slice_selection_eval_conclusion.md`

**Conclusión técnica:** la selección de slice sin máscara es viable en esta formulación binaria 2D. La estrategia `center_window_best_prediction` mantiene un desempeño cercano a la referencia con máscara y evita depender de la máscara real, por lo que se recomienda como estrategia operativa para la siguiente iteración.

## Tabla de resultados principales

| Experimento | Configuración | Threshold | Dice | IoU | Observación |
|---|---:|---:|---:|---:|---|
| Baseline inicial | 20 casos T1, validación interna | 0.5 | 0.2364 | 0.1344 | Colapso a foreground |
| Baseline inicial | 20 casos T1, mejor threshold | 0.7 | 0.3148 | 0.1885 | Mejora parcial por threshold |
| Baseline mejorado | 100 casos T1, validación interna | 0.5 | 0.8875 | 0.7988 | Sin colapso evidente |
| Baseline mejorado | Holdout interno T1 | 0.5 | 0.8816 | 0.7904 | Generalización interna razonable |
| Referencia con máscara | `mask_max_area`, holdout | 0.5 | 0.8816 | 0.7904 | Selección idealizada con máscara |
| Selección sin máscara | `center_window_best_prediction`, holdout | 0.5 | 0.8575 | 0.7631 | Mejor estrategia sin máscara |

## Interpretación de resultados

El baseline inicial permitió validar el flujo completo de entrenamiento, inferencia, métricas y visualización, pero presentó un comportamiento deficiente. El diagnóstico posterior mostró que el modelo tendía a predecir foreground de forma excesiva, con `pred_foreground_ratio` igual a 1.0000 en threshold 0.5. Esta evidencia permitió identificar un problema de calibración, desbalance y sobreajuste antes de introducir cambios más ambiciosos.

El baseline mejorado corrigió el colapso a foreground mediante una combinación de más datos, más épocas y una función de pérdida con manejo explícito del desbalance (`BCEWithLogitsLoss` con `pos_weight` más Dice loss). El desempeño en validación interna aumentó sustancialmente, alcanzando Dice 0.8875 e IoU 0.7988 con threshold 0.5.

La validación holdout interna mostró resultados cercanos a la validación interna: Dice 0.8816 e IoU 0.7904. Además, los sanity checks no evidenciaron predicciones vacías ni llenas, ni casos con proporciones extremas de foreground predicho. Esto respalda el baseline binario como una validación técnica sólida dentro de SPIDER para la formulación simplificada.

Finalmente, la evaluación de selección de slice sin máscara abordó una limitación importante: hasta ese punto, el slice representativo se seleccionaba usando la máscara real. La estrategia `center_window_best_prediction`, que no usa máscara, obtuvo Dice 0.8575 e IoU 0.7631, con una caída absoluta de Dice de 0.0241 frente a la referencia con máscara. Esta caída es pequeña para una validación preliminar y sugiere que es viable avanzar con estrategias de selección sin máscara o inferencia multi-slice.

## Decisiones técnicas justificadas

- **Excluir `t2_SPACE` en la primera fase:** se detectaron inconsistencias de direction en 14 casos asociados a secuencias `t2_SPACE`. Excluirlos permitió estabilizar el pipeline inicial.
- **Usar T1 como modalidad inicial:** T1 ofreció un subconjunto homogéneo para validar el flujo binario antes de incorporar variabilidad adicional.
- **Formular segmentación binaria antes de multiclase:** `mask > 0` permitió validar lectura, normalización, entrenamiento, inferencia y métricas sin introducir todavía complejidad por estructura anatómica individual.
- **Mantener eje sagital 2:** las exploraciones multi-eje identificaron el eje 2 como representativo para la inspección sagital en los casos evaluados.
- **Usar threshold 0.5 luego del baseline mejorado:** el diagnóstico y la validación holdout mostraron que threshold 0.5 era el mejor o equivalente al mejor threshold tras corregir el entrenamiento.
- **Adoptar `center_window_best_prediction` como estrategia sin máscara:** obtuvo la mejor relación entre desempeño y realismo operativo entre las estrategias sin máscara evaluadas.
- **Postergar multiclase, axial y nnU-Net:** el pipeline binario todavía es una evidencia preliminar y debe consolidarse antes de escalar complejidad.

## Limitaciones

- La segmentación evaluada es binaria y no diferencia estructuras anatómicas individuales.
- Se utiliza un único slice sagital por volumen.
- La evaluación se realizó únicamente dentro del dataset SPIDER.
- No existe validación clínica externa.
- No se implementó inferencia 3D.
- No hay interfaz ni backend integrados.
- No se evaluó desempeño por estructura anatómica individual.
- Las estrategias multi-slice fueron evaluadas como selección de slice, no como segmentación volumétrica.
- El entrenamiento se realizó en CPU en las ejecuciones reportadas, lo que limita exploración de hiperparámetros.

## Recomendaciones próximas

- Consolidar figuras y tablas para el capítulo de implementación/resultados de la tesis.
- Mantener este pipeline como baseline comparativo para experimentos posteriores.
- Avanzar a evaluación por estructura o multiclase 2D cuando el objetivo sea diferenciar anatomías individuales.
- Evaluar T2 luego de cerrar la línea T1.
- Considerar inferencia multi-slice para reducir dependencia de una única selección de corte.
- Evaluar augmentations y mayor cantidad de casos antes de saltar a arquitecturas más complejas.
- Recién después comparar contra nnU-Net o explorar axial/3D.

## Archivos de evidencia

### Preprocesamiento y consistencia

- `/content/drive/MyDrive/PFI_MVP/results/E4_preprocesamiento/E4_spider_consistency_summary.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E4_preprocesamiento/E4_baseline_candidates_no_space.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E4_preprocesamiento/E4_preprocessed_example_100_t1.npz`

### Baseline inicial y diagnóstico

- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_training_history.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_baseline_metrics.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_threshold_summary.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_diagnostic_report.json`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_sagital/E5_simple_unet2d_baseline.pt`

### Baseline mejorado

- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_selected_cases.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_split.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_training_history.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_threshold_summary.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_validation_report.json`
- `/content/drive/MyDrive/PFI_MVP/results/E5_baseline_mejorado_binario/E5_improved_unet2d_binary_best.pt`

### Holdout y sanity checks

- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_selected_cases.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_metrics_by_case.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_threshold_summary.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_sanity_checks.json`
- `/content/drive/MyDrive/PFI_MVP/results/E5_holdout_validacion/E5_holdout_validation_report.json`

### Selección de slice sin máscara

- `/content/drive/MyDrive/PFI_MVP/results/E5_slice_selection_eval/E5_slice_selection_metrics_by_case.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_slice_selection_eval/E5_slice_selection_summary.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_slice_selection_eval/E5_slice_selection_vs_mask_reference.csv`
- `/content/drive/MyDrive/PFI_MVP/results/E5_slice_selection_eval/E5_slice_selection_report.json`

### Figuras y conclusiones

- `/content/drive/MyDrive/PFI_MVP/figures/E5_improved_prediction_example.png`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_holdout_prediction_example_01.png`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_holdout_prediction_example_02.png`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_holdout_prediction_example_03.png`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_slice_strategy_example_01.png`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_slice_strategy_example_02.png`
- `/content/drive/MyDrive/PFI_MVP/figures/E5_slice_strategy_example_03.png`
- `/content/drive/MyDrive/PFI_MVP/docs/E5_baseline_mejorado_binario_conclusion.md`
- `/content/drive/MyDrive/PFI_MVP/docs/E5_holdout_validacion_conclusion.md`
- `/content/drive/MyDrive/PFI_MVP/docs/E5_slice_selection_eval_conclusion.md`

## Cierre

Entre E3 y E5 se construyó y validó un pipeline experimental de segmentación sagital binaria sobre SPIDER. La evidencia muestra que el baseline inicial fue útil para detectar fallas, el diagnóstico permitió corregir el colapso a foreground y el baseline mejorado alcanzó un desempeño consistente en validación interna y holdout interno. La evaluación de selección de slice sin máscara redujo una limitación importante del flujo y dejó una estrategia viable para continuar.

Estos resultados deben interpretarse como validación técnica preliminar de una formulación simplificada, no como un modelo clínico final.

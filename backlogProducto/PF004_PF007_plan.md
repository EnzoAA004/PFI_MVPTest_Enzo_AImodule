# PF-004 a PF-007 - Modelos finales, configuración y métricas

Estado: En progreso.

## Objetivo

Cerrar los artefactos finales de modelos para el producto final, tomando como base la estructura de datos congelada en PF-001 a PF-003.

## Tickets cubiertos

- PF-004: consolidar o reentrenar axial final reproducible.
- PF-005: consolidar modelo sagital final.
- PF-006: generar configuración única de modelos.
- PF-007: generar reporte final de métricas por modelo.

## Criterio operativo

No se reentrena automáticamente si ya existen checkpoints validados. El bloque prioriza consolidar artefactos finales, registrar hashes, rutas, métricas y decisión metodológica. El reentrenamiento axial queda como opción explícita si se decide ejecutar una corrida final adicional.

## Entradas esperadas

- `config/data_freeze_config.json`
- `models/E10_axial_t2_final_training_clean_best.pt`
- `models/E12_sagittal_multiclass_final_best.pt`
- reportes y métricas previas E10, E11 y E12

## Salidas esperadas

- `models/final/axial_t2_alkafri_final_best.pt`
- `models/final/sagittal_spider_multiclass_final_best.pt`
- `config/model_registry_final.json`
- `results/PF004_PF007_final_models/PF004_PF007_model_artifacts.csv`
- `results/PF004_PF007_final_models/PF006_model_registry_final.json`
- `results/PF004_PF007_final_models/PF007_final_model_metrics.csv`
- `results/PF004_PF007_final_models/PF007_final_model_metrics_by_class.csv`
- `results/PF004_PF007_final_models/PF004_PF007_report.json`
- `docs/PF004_PF007_final_models_metrics.md`

## Decisión esperada

Dejar los modelos finales listos para ser consumidos por el servicio Python y por los próximos endpoints de inferencia real.

## Próximo bloque

PF-008 a PF-011: convertir la inferencia E13 en módulos productivos reutilizables.

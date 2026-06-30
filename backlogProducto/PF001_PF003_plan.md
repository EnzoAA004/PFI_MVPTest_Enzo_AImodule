# PF-001 a PF-003 - Base final de datos, splits e inventario

Estado: En progreso.

## Objetivo

Cerrar la base reproducible para el producto final antes de avanzar con entrenamiento definitivo, endpoints reales y frontend final.

## Tickets cubiertos

- PF-001: estandarizar estructura de datos final.
- PF-002: congelar splits finales.
- PF-003: inventario de datasets y licencias.

## Decisiones

- SPIDER queda como dataset principal sagital.
- Al-Kafri/Sudirman queda como dataset complementario axial.
- Datasets opcionales no entran a metricas finales salvo decision explicita y licencia verificada.
- No se usan datos privados identificables.
- Axial y sagital se documentan como modulos 2D separados si no pertenecen al mismo paciente.

## Evidencia esperada

- config/data_freeze_config.json
- results/PF001_PF003_dataset_freeze/PF001_canonical_data_paths.json
- results/PF001_PF003_dataset_freeze/PF001_physical_file_inventory.csv
- results/PF001_PF003_dataset_freeze/PF002_final_splits_manifest.csv
- results/PF001_PF003_dataset_freeze/PF002_split_summary.csv
- results/PF001_PF003_dataset_freeze/PF003_dataset_inventory.csv
- results/PF001_PF003_dataset_freeze/PF003_dataset_licenses.md
- results/PF001_PF003_dataset_freeze/PF001_PF003_report.json

## Proximo paso posterior

PF-004 a PF-007: modelos finales, configuracion unica de modelos y reporte definitivo de metricas.

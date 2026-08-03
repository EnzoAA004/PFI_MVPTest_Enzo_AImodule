# P10.6-AI - Alcance RSNA para hallazgos degenerativos candidatos

## 1. Origen del requerimiento en user research

La linea P10.6-AI nace de la necesidad de ampliar el AI Module desde mediciones y segmentaciones hacia una clasificacion asistida de hallazgos degenerativos frecuentes en resonancia lumbar. El resultado previsto es un hallazgo candidato que requiere revision profesional, no una conclusion clinica autonoma.

## 2. Motivo de priorizar hallazgos frecuentes

Los hallazgos degenerativos lumbares son frecuentes en los flujos de revision profesional y aparecen en niveles anatomicos repetibles. Priorizarlos permite construir una salida estructurada, auditable y util para revision, manteniendo `humanReviewRequired=true` y `notClinicalDiagnosis=true`.

## 3. Alcance cubierto por RSNA

El dataset RSNA 2024 Lumbar Spine Degenerative Classification, usado aqui como RSNA LumbarDISC, contiene estudios lumbares DICOM y etiquetas por estudio, condicion, nivel y severidad para un conjunto acotado de hallazgos degenerativos. Su uso en este PFI queda restringido al ambito academico y no comercial.

## 4. Condiciones entrenables

Las condiciones entrenables con RSNA en esta linea son:

- `spinal_canal_stenosis`
- `neural_foraminal_narrowing_left`
- `neural_foraminal_narrowing_right`
- `subarticular_stenosis_left`
- `subarticular_stenosis_right`

Cada salida debe presentarse como clasificacion asistida de un hallazgo candidato y requiere revision profesional.

## 5. Niveles

Los niveles normalizados son:

- `L1-L2`
- `L2-L3`
- `L3-L4`
- `L4-L5`
- `L5-S1`

La unidad de split futura sera `study_id` para reducir riesgo de leakage entre train, validation y holdout interno.

## 6. Severidad

Las clases de severidad normalizadas son:

- `normal_mild`
- `moderate`
- `severe`

El problema no debe colapsarse inicialmente a binario. En particular, `moderate` y `severe` se conservan como clases independientes.

## 7. Secuencias

El mapeo previsto por condicion es:

| Condicion | Secuencia preferida | Nota |
|---|---|---|
| `spinal_canal_stenosis` | Sagittal T2/STIR | Puede recibir apoyo axial en una version fusionada futura. |
| `neural_foraminal_narrowing_left` | Sagittal T1 | Lateralidad izquierda. |
| `neural_foraminal_narrowing_right` | Sagittal T1 | Lateralidad derecha. |
| `subarticular_stenosis_left` | Axial T2 | Lateralidad izquierda. |
| `subarticular_stenosis_right` | Axial T2 | Lateralidad derecha. |

La normalizacion de `series_description` debe ser explicita y auditable. No se clasifica una serie por contener una letra aislada.

## 8. Esquema de etiquetas

Se mantiene un mapping reversible entre nombres originales de RSNA y nombres normalizados. El Notebook 53 debe auditar labels no reconocidos, niveles no reconocidos, severidades no reconocidas, duplicados, nulos y consistencia entre `train.csv`, `train_label_coordinates.csv`, `train_series_descriptions.csv` y `train_images/`.

## 9. Que no cubre RSNA

RSNA no contiene etiquetas directas y confiables para:

- `disc_bulge`
- `disc_protrusion`
- `disc_extrusion`
- `disc_sequestration`

Tampoco cubre en esta linea tumor, infeccion ni fractura. Estas categorias quedan excluidas de entrenamiento RSNA.

## 10. Plan independiente para protrusion/extrusion

La protrusion y extrusion discal requieren un protocolo de anotacion especifico, independiente de RSNA. No deben inferirse desde etiquetas de estenosis ni incorporarse como targets derivados en P10.6-AI.

## 11. Privacidad

El notebook no debe imprimir `PatientName`, `PatientID` original, fecha de nacimiento, institucion asociada a un estudio individual, accession number, UIDs completos, headers DICOM completos ni rutas absolutas del usuario. Los `study_id` y `series_id` se tratan como identificadores pseudonimos; los reportes versionables deben usar conteos, hashes o identificadores truncados.

## 12. Licencia y restriccion no comercial

El uso academico/no comercial del dataset RSNA fue autorizado para este PFI. El repositorio no debe versionar DICOM, dataset oficial, checkpoints grandes ni outputs pesados.

## 13. Politica de test

El conjunto oficial de test, si existe en la raiz local del dataset, queda prohibido para esta fase:

- no abrir;
- no inventariar;
- no visualizar;
- no utilizar;
- no extraer estadisticas.

El reporte solo puede registrar `officialTestPresent=true` y `officialTestAccessed=false`.

## 14. Riesgos de desbalance

Se espera desbalance por condicion, nivel, severidad y secuencia. El Notebook 53 debe reportar clases minoritarias, ratios maximo/minimo, prevalencias, estudios con etiquetas incompletas y distribuciones cruzadas.

## 15. Metricas

Las metricas futuras deben incluir recall/sensibilidad, precision, especificidad, F1 macro, balanced accuracy, ROC-AUC one-vs-rest, matriz de confusion, log loss ponderada, calibracion, falsos negativos `severe`, metricas por nivel y metricas por institucion solo si existe metadata autorizada y segura. Accuracy sola no es suficiente.

## 16. Notebooks 53 a 60

Plan preliminar:

| Notebook | Objetivo |
|---|---|
| 53 | Preflight seguro de dataset RSNA train-only. |
| 54 | Split interno por paciente y plan de modelado. |
| 55 | Baseline de clasificacion asistida por condicion/nivel. |
| 56 | Auditoria de coordenadas y seleccion de slices. |
| 57 | Entrenamiento controlado y registro de experimentos. |
| 58 | Calibracion y errores, con foco en falsos negativos severe. |
| 59 | Empaquetado candidato para AI Module. |
| 60 | Evaluacion final autorizada segun protocolo aprobado. |

No se inicia Notebook 54 sin aprobacion explicita.

## 17. Impacto futuro en AI Module

El AI Module debera incorporar un `modelKey` futuro, por ejemplo `lumbar_findings_rsna`, capaz de devolver hallazgos candidatos con probabilidades por severidad, evidencia de serie/slice, calidad y flags de revision.

## 18. Impacto futuro en Backend

El Backend debera persistir hallazgos candidatos editables, trazabilidad de modelo, flags de calidad y estado de revision profesional. No debe tratarlos como diagnosticos confirmados.

## 19. Impacto futuro en Frontend

El Frontend debera mostrar clasificacion asistida con probabilidades, nivel, lateralidad, secuencia fuente y estado `requires_professional_review`, permitiendo edicion o descarte por un profesional.

## 20. Contrato preliminar

```json
{
  "findingId": "finding_<opaque>",
  "findingType": "spinal_canal_stenosis",
  "level": "L4-L5",
  "laterality": "central",
  "severity": "moderate",
  "probabilities": {
    "normal_mild": 0.08,
    "moderate": 0.74,
    "severe": 0.18
  },
  "confidence": 0.74,
  "sourcePlane": "sagittal",
  "sourceSequence": "T2_STIR",
  "sourceSeriesId": "opaque",
  "sourceSliceIndices": [7, 8, 9],
  "sourceInstanceNumbers": [8, 9, 10],
  "levelLocalizationSource": "rsna_coordinate_or_model",
  "modelKey": "lumbar_findings_rsna",
  "modelVersion": "candidate",
  "status": "requires_professional_review",
  "qualityFlags": [],
  "humanReviewRequired": true,
  "notClinicalDiagnosis": true
}
```

Los campos `diagnosis`, `confirmedPathology`, `treatment`, `medicalConclusion` y `clinicalRecommendation` no forman parte del contrato.

## 21. Definicion de exito del Notebook 53

Notebook 53 se considera exitoso cuando:

- valida dependencias y entorno;
- monta Drive solo cuando corresponde;
- valida estructura de train sin exigir test;
- ignora test oficial si esta presente;
- construye inventario train-only;
- calcula hashes SHA-256 de los CSV autorizados;
- audita consistencia de estudios, series, coordenadas, labels, niveles, severidad y secuencias;
- genera reportes externos sanitizados bajo `PFI_P10_6_OUTPUT_ROOT`;
- mantiene `officialTestAccessed=false`;
- no entrena, no descarga dataset y no accede al conjunto oficial de test.

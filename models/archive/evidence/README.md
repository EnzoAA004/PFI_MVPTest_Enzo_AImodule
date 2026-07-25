# Runtime manifest evidence

Esta carpeta conserva evidencia original de manifests antes de adaptaciones de compatibilidad runtime.

- `axial_t2_alkafri_final_v2_candidate.original.manifest.json` conserva el manifest axial candidate antes de agregar los campos resumen `metrics.dice` y `metrics.iou`.
- El manifest runtime en `models/final/axial_t2_alkafri_final_v2_candidate.pt.manifest.json` mantiene sin cambios `artifactFile`, `artifactSha256`, `trainingStatus`, `humanReviewRequired`, `notClinicalDiagnosis`, clases raw_*, metricas de validacion/test y `qualityGate`.
- Los campos runtime `metrics.dice` y `metrics.iou` fueron copiados desde `metrics.test.dice_macro_foreground` y `metrics.test.iou_macro_foreground`; no fueron recalculados.

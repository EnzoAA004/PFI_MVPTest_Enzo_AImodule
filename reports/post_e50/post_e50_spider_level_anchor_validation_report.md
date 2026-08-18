# Post-E50 SPIDER Level Anchor Validation

## Objective

Validate the Notebook 67 sagittal instance-localization pipeline against real SPIDER data. SPIDER provides RELATIVE instance identity (bottom-up numbering), not absolute anatomical levels -- this notebook never claims L1-L2...L5-S1 ground truth from SPIDER.

## CIERRE FORMAL -- Full SPIDER validation run en Google Colab (2026-08-18)

**Este es el estado final y autoritativo del experimento**, registrado a partir de una corrida real manual en Colab con `RUN_FULL_VALIDATION=True` sobre el cohort completo T2-evaluable. Estos números NO se reproducen dinámicamente al regenerar este reporte localmente (SPIDER no está disponible en este worktree) -- son el resultado real registrado, documentado como evidencia estática de cierre.

**Validation cohort accounting:**
`validation_patients_total=39`, `validation_patients_t2_evaluable=38`, `cases_attempted=38`, `cases_executed=38`, `cases_failed=0`, `cases_excluded_no_t2=1`, `test_patients_used=0`.

**Final full-validation summary:**
`mean_precision=0.9506996658312448`, `mean_recall=1.0`, `mean_f1=0.9721906576395739`, `median_f1=1.0`, `mean_centroid_error_mm=1.7523566724391513`, `median_centroid_error_mm=1.5368929128025899`, `relative_order_monotonic_rate=1.0`, `relative_order_direction_resolved_rate=0.0`, `mean_spearman_rank_correlation=0.05263157894736842`.

Todos los GT discs fueron recuperados en los 38 casos evaluables (`recall=1.0`); los errores restantes son predominantemente falsos positivos/over-segmentation, no falta de sensibilidad. El ordering relativo fue monotónico en 38/38 casos, pero con signo del eje PCA no resuelto de forma independiente: 20 casos con Spearman +1 y 18 con Spearman -1 (`relative_order_direction_resolved_rate=0.0`) -- confirma empíricamente la ambigüedad de signo investigada en la sección de causa raíz más abajo.

**FOV / GT primary-T2 disc-count distribution (38 pacientes T2-evaluables):** 3 discs: 1 · 6 discs: 14 · 7 discs: 9 · 8 discs: 9 · 9 discs: 5. No se asume que una serie lumbar válida deba producir exactamente 5 disc instances.

**Final gates:** GATE_A=PASS, GATE_B=PASS, GATE_C=PASS, GATE_D=PASS, GATE_E=PASS, GATE_F=PASS, GATE_G=PASS, GATE_H=UNAVAILABLE_FROM_DATASET_REFERENCE, GATE_I=UNAVAILABLE, GATE_J=PASS, RELATIVE_ORDERING_VALIDATION=PASS, QUALITY_GATE_OVERALL=PARTIAL. **Decision:** SPIDER_RELATIVE_LOCALIZATION_BASELINE_ESTABLISHED. **Readiness:** ready_for_relative_instance_validation_batch=YES, ready_for_axial_cluster_pairing=NO, ready_for_absolute_anchor_dataset_research=YES.

`AUTOMATIC_DISC_LOCALIZATION_VALIDATED` permanece `False` -- no se modifica ese flag de producto/runtime. 67A valida detección, localización física y ordering relativo; no valida naming anatómico absoluto (L1-L2...L5-S1).

### What 67A proves

On the public SPIDER validation cohort, the frozen sagittal baseline established reliable relative disc-instance localization: all 38 T2-evaluable studies executed successfully, mean per-case F1 was 0.9722, median F1 was 1.0, mean physical centroid error was 1.75 mm, and the matched disc sequence was monotonic in 100% of evaluated cases.

### What 67A does NOT prove

- no absolute L1-L2...L5-S1 anatomical anchor
- no clinically validated diagnosis
- no pathology classification
- no cross-frame axial level pairing
- no test-set result
- no automatic product promotion

**Aclaración explícita:** `RELATIVE_ORDERING_VALIDATION=PASS` significa "correct relative sequence up to global inversion" -- **no** significa "cranial/caudal direction independently resolved". Esa dirección absoluta sigue sin resolverse (`relative_order_direction_resolved_rate=0.0`), motivando el siguiente experimento (67B, absolute lumbar level anchor).

## Execution status (this run)

Stage A / local validation: SPIDER data was not accessed in this run (expected outside Colab).

## Embedded runtime provenance

- EMBEDDED_RUNTIME_SOURCE_COMMIT: `0e97083d443225226beb1f705fd794578b1b17f9`
- EMBEDDED_RUNTIME_SCOPE: `minimal frozen sagittal inference helpers required by Notebook 67A`
- This notebook embeds a minimal frozen copy of `build_checkpoint_model`, `resize_image`, `robust_percentile_normalize`, `connected_instances`, and `MODEL_REGISTRY` from the immutable Notebook-67 commit above -- it is NOT a new implementation, and it is NOT imported from `ai_service.pfi_ai_service.*` at runtime (no `PFI_AI_REPO_ROOT`, no `<PFI_DRIVE_ROOT>/repo` dependency).

## Checkpoint resolver

- Expected SHA-256: `cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944`
- Resolved source: `local_repo_checkpoint`, sha256: `cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944`
- GATE A: `PASS`
- The resolver never trusts a candidate by path/filename -- it always verifies by computed SHA-256. The LOCAL git checkout's models/final/sagittal_spider_multiclass_final_best.pt matches the expected SHA and is accepted outside Colab; in Colab, PFI_POST_E50_SAGITTAL_CHECKPOINT is preferred and accepted only on a SHA match (a previously observed Drive-only copy with a mismatched SHA is rejected by design, not the local git checkout).

## Preprocessing parity (4 independent sub-audits -- do not overclaim from axis parity alone)

- AXIS_CANONICALIZATION_PARITY: `PASS` (code-identity with notebooks/45_gcs_spider_final_training.ipynb: same shape heuristic, same fixed axis=2)
- RESIZE_PARITY: `PASS` (source: ai_service.pfi_ai_service.real_inference_runtime.resize_image, embedded verbatim in STEP 1B -- commit 0e97083d443225226beb1f705fd794578b1b17f9, not imported)
- INTENSITY_PREPROCESSING_PARITY: `PASS` (source: ai_service.pfi_ai_service.real_inference_runtime.robust_percentile_normalize, embedded verbatim in STEP 1B -- commit 0e97083d443225226beb1f705fd794578b1b17f9, not imported)
- MODEL_INPUT_SHAPE_PARITY: `PASS` (sagittal_runtime_meta['targetSize'] vs training TARGET_SIZE=(256,256))

## Prediction -> original MHA coordinate roundtrip (critical, gates the smoke test)

- PREDICTION_TO_MHA_COORDINATE_ROUNDTRIP: `PASS`
- Validated via synthetic self-tests: (1) canonical index <-> native SimpleITK index mapping round-trip for both the swap and no-swap cases of canonicalize_spider_array; (2) SimpleITK TransformContinuousIndexToPhysicalPoint <-> TransformPhysicalPointToContinuousIndex round-trip on a synthetic image; (3) native-pixel <-> model-grid resize-inverse round-trip. At runtime, each predicted centroid is additionally self-checked to round-trip back to its native canonical index via image_sitk (the ORIGINAL, pre-preprocessing image) -- both predicted and GT centroids end in the same physical space by construction (both call canonical_index_to_physical_xyz against their own image_sitk / mask_sitk objects). If this gate is not PASS, run_v67_frozen_pipeline_on_case() returns status=coordinate_roundtrip_not_validated and no centroid_error_mm is produced for that case.

## Dataset verification / splits / leakage

- GATE B (dataset structure): `NOT_RUN`
- GATE C (train/validation leakage): `NOT_RUN`
- GATE E (image-mask exact pairing): `NOT_RUN`
- public_test_availability: `HIDDEN_EXTERNAL_NOT_AVAILABLE`

_Not computed in this run._

## Validation cohort accounting

- validation_patients_total: `0`, validation_patients_t2_evaluable: `0`, validation_patients_not_evaluable: `0`
- not_evaluable_reasons: `{}`
- A full validation batch attempts exactly the T2-evaluable count, never the total validation patient count.

## Smoke test (real pipeline execution)

- GATE D (execution): `NOT_RUN` -- PASS means 3/3 cases ran without runtime failure, NOT high accuracy.
- GATE F (instance GT extraction + metrics): `NOT_RUN`

_Not run in this session (no SPIDER data)._

## Relative disc ordering validation (separate from GATE F detection)

- RELATIVE_ORDERING_VALIDATION: `UNRESOLVED`
- Computed per matched pair (Hungarian output) as predicted_rank vs gt_relative_instance_label. relative_order_direction is NEVER derived from GT -- it stays `ORDER_DIRECTION_UNRESOLVED` unless independent physical-space/anatomical evidence resolves it (not available for SPIDER MHA in this notebook). See the sign-ambiguity investigation above (spine_axis_from_points self-test) for the confirmed root cause of pred_index-to-gt_index direction differing across cases.

## Mask/overview/grading audits

- overview-mask disc-count consistency rate: `NOT_RUN`
- GRADING_MASK_LABEL_MAPPING_STATUS: `UNRESOLVED` (never converted to an absolute anatomical level)

## FOV analysis

- GATE G: `NOT_RUN`. Notebook 67's real-DICOM result of 7 candidates is NOT concluded to be an error by this analysis.
- `validation_series_num_discs_distribution` (ALL validation series, may include >1 per patient) is now named explicitly to avoid implying it is per-patient. `validation_primary_t2_num_discs_distribution` (exactly one primary T2 per T2-evaluable patient) is the primary comparison distribution against this pipeline.

## Absolute anatomical anchor

- GATE H: `UNAVAILABLE_FROM_DATASET_REFERENCE` -- a valid methodological outcome, not a failure.
- GATE I (absolute level naming metrics): `UNAVAILABLE` -- no L1-L2...L5-S1 confusion matrix generated (no ground truth exists).
| method                         | anchor_available   |   anchor_confidence | anchor_reason                                       | predicted_target_window   | warnings   |
|:-------------------------------|:-------------------|--------------------:|:----------------------------------------------------|:--------------------------|:-----------|
| A_inferior_disc_sacral_context | False              |                   0 | no_dedicated_sacral_class_in_checkpoint             |                           | []         |
| B_vertebral_sequence_context   | False              |                   0 | vertebral_instances_not_separable                   |                           | []         |
| C_disc_spacing_pattern         | False              |                   0 | insufficient_spacing_samples                        |                           | []         |
| D_combined                     | False              |                   0 | no_component_anchor_resolved_absolute_target_window |                           | []         |

## Quality gates

| gate                                   | status                             |
|:---------------------------------------|:-----------------------------------|
| GATE_A_checkpoint_identity             | PASS                               |
| GATE_B_spider_structure                | NOT_RUN                            |
| GATE_C_public_train_validation_leakage | NOT_RUN                            |
| GATE_D_smoke_test_execution            | NOT_RUN                            |
| GATE_E_image_mask_exact_pairing        | NOT_RUN                            |
| GATE_F_instance_gt_extraction_metrics  | NOT_RUN                            |
| GATE_G_fov_analysis                    | NOT_RUN                            |
| GATE_H_absolute_anatomical_anchor      | UNAVAILABLE_FROM_DATASET_REFERENCE |
| GATE_I_absolute_level_naming_metrics   | UNAVAILABLE                        |
| GATE_J_privacy                         | PASS                               |

## Results

- Overall quality gate: `PARTIAL`
- Decision: `PARTIAL`
- ready_for_axial_cluster_pairing: `False`
- ready_for_relative_instance_validation_batch: `False`
- ready_for_absolute_anchor_dataset_research: `True`

## Test lock

- TEST_SPLIT_LOCKED: `True` (structurally enforced) · test_patients_used: `0` · public_test_availability: `HIDDEN_EXTERNAL_NOT_AVAILABLE`

## Limitations

- SPIDER instance labels (vertebrae 1,2,3...; canal 100; discs 201,202,203...) are RELATIVE, bottom-up identity, not absolute anatomical levels -- L1-L2...L5-S1 naming is UNAVAILABLE_FROM_SPIDER_REFERENCE.
- The public SPIDER split has no test subset locally; the hidden challenge test is out of scope and never accessed.
- The checkpoint resolver never trusts a candidate by path/filename alone -- it always verifies by computed SHA-256 (expected: cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944). The LOCAL git checkout's models/final/sagittal_spider_multiclass_final_best.pt DOES match this expected SHA (confirmed: cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944) and is accepted when resolved outside Colab. In Colab, PFI_POST_E50_SAGITTAL_CHECKPOINT is preferred and accepted only on a SHA match; a stale Drive-only copy with a different SHA (7dd393cc750311c98003516d8110136310c31e8b6f0f00b6815f949fd61ef15b) was observed in one prior Colab session and is rejected by design, but that staleness was specific to that Drive clone, not the local git history.
- MHA slice-axis handling reproduces notebooks/45_gcs_spider_final_training.ipynb's shape heuristic + fixed axis=2 exactly (code-identity parity), not an independently re-derived or spacing-verified convention.
- GRADING_MASK_LABEL_MAPPING_STATUS, if VALIDATED, is scoped to the cases actually processed in this run (smoke test / full validation), not the full 218-patient cohort unless RUN_FULL_VALIDATION=True was executed.
- GATE D (smoke test) PASS means 3/3 cases executed without runtime failure -- it does NOT imply high detection accuracy; no performance threshold is defined.
- No training was performed; no checkpoint was modified; AUTOMATIC_DISC_LOCALIZATION_VALIDATED was not touched; Notebook 67 was not modified.
- TEST_SPLIT_LOCKED=True structurally prevents any test execution in this notebook.
- overview.csv parsing skipped: SPIDER_AVAILABLE=False in this run.
- Split inventory not built: GATE B did not PASS in this run.
- spider_dataset_inventory not built: GATE B did not PASS in this run.
- validation_t2_series_inventory not built: dataset inventory or split_inventory unavailable.
- validation_cohort_accounting not computed: validation_t2_series_inventory unavailable in this run.
- mask_overview_consistency requires opening real mask .mha files -- populated per-case during the smoke test / full validation loop below, not as a separate full-cohort pass in this cell (avoids opening all 447 masks twice).
- radiological_gradings.csv audit skipped: SPIDER_AVAILABLE=False in this run.
- Smoke test cohort not selected: validation_t2_series_inventory unavailable in this run.
- Smoke test NOT executed: SPIDER unavailable, checkpoint invalid, image/mask pairing invalid, or empty cohort in this run.
- FOV analysis NOT RUN: overview.csv/split_inventory unavailable in this run.
- Drive output directories not configured in this run (not in Colab / PFI_DRIVE_ROOT unset) -- only local repo-scoped artifacts were written.

## What 67A proves (this run)

> No real SPIDER execution evidence exists in this run (Stage A / local validation only, or the smoke test did not PASS). No performance or execution claim is made beyond what actually ran.

## What 67A does not prove (this run)

- no absolute anatomical level ground truth exists in SPIDER for this dataset/run -- L1-L2...L5-S1 is never claimed
- GATE D PASS is execution-only, not a performance/accuracy claim
- no full validation batch (would attempt exactly 0 T2-evaluable patients, not the 0 total validation patients) unless RUN_FULL_VALIDATION was explicitly set True and executed
- relative disc ordering is NOT claimed as validated beyond the RELATIVE_ORDERING_VALIDATION sub-audit's actual scope (smoke-test cohort only, unless full validation was run) -- ORDER_DIRECTION_UNRESOLVED unless independent physical evidence exists
- test split was never touched
- Notebook 67 was not modified

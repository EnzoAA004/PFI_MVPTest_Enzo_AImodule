# Post-E50 Absolute Lumbar Level Anchor (RSNA) -- Stage A + Stage B + Stage C smoke

## Roadmap

67A = SPIDER relative localization validation (closed). 67B (this notebook) = absolute lumbar level anchor. 67C = axial cluster-level pairing (future, blocked until 67B produces an anchor).

## Objective

Resolve, reproducibly and evaluably: predicted relative disc instances -> absolute lumbar levels (L1-L2...L5-S1), without using ground truth to decide the answer during inference. GT is used exclusively for training/evaluation.

## Execution status (this run)

- RSNA_AVAILABLE: `False`
- officialTestPresent: `False`, officialTestAccessed: `False` (never accessed, structurally)
- Stage A / local validation: RSNA data was not accessed in this run (expected outside Colab / without Drive mounted). This is a Colab-ready design, stopped before real execution.

## Stage A -- dataset structure and schema audit

- GATE A (dataset structure): `NOT_RUN`
- GATE B (absolute level reference availability): `NOT_RUN`
- GATE C (split leakage, study_id-level): `NOT_RUN`

_RSNA not available in this run -- no real schema to report._

## Stage B -- coordinate geometry design

- coordinate_mapping_formula_selftest (method-level, synthetic pydicom self-test): `PASS`
- simpleitk_physical_crosscheck (method-level, independent SimpleITK geometry engine): `PASS`
- **These two sub_audits confirm the METHOD is correct, not parity against real RSNA DICOM.** GATE_D and GATE_E never inherit a PASS from them -- they require real data.
- GATE D (series/annotation-to-DICOM mapping, real data): `NOT_RUN`
- GATE E (coordinate physical parity, real data): `NOT_RUN`
- `dicom_pixel_to_patient_xyz` reuses the same class of DICOM pixel-to-patient-physical formula as Notebook 66/67 -- no new geometric convention was invented.

## Stage C -- frozen 67A smoke (3 validation studies, never internal_test)

- Expected checkpoint SHA-256: `cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944`
- Checkpoint identity this run: `PASS` (source: `local_repo_checkpoint`)
- Smoke cohort selected (opaque study_id hashes): `[]`
- GATE F (frozen 67A execution on RSNA): `NOT_RUN`
- GATE G (predicted-to-absolute-GT matching): `NOT_RUN`

{
  "cases_attempted": 0,
  "cases_executed": 0,
  "cases_failed": 0,
  "reference_points_total": 0,
  "predicted_instances_total": 0,
  "mean_nearest_full_xyz_distance_mm": null,
  "median_nearest_full_xyz_distance_mm": null,
  "min_nearest_full_xyz_distance_mm": null,
  "max_nearest_full_xyz_distance_mm": null,
  "mean_nearest_axis_distance_mm": null,
  "median_nearest_axis_distance_mm": null,
  "min_nearest_axis_distance_mm": null,
  "max_nearest_axis_distance_mm": null,
  "axis_assignment_level_order_consistency_rate": null,
  "axis_hungarian_distance_count": 0,
  "mean_axis_hungarian_distance_mm": null,
  "median_axis_hungarian_distance_mm": null,
  "min_axis_hungarian_distance_mm": null,
  "max_axis_hungarian_distance_mm": null,
  "cases_monotonic_up_to_reversal": 0,
  "cases_same_direction": 0,
  "cases_reversed_direction": 0,
  "cases_mixed_direction": 0,
  "GATE_F_frozen_67A_execution_on_RSNA": "NOT_RUN",
  "GATE_G_pred_to_absolute_GT_matching": "NOT_RUN",
  "GATE_G_reason": null
}

**Diagnostic matching (this revision):** `MAX_MATCH_DISTANCE_MM` cutoff removed for diagnosis -- the RSNA Spinal Canal Stenosis points are `ABSOLUTE_LEVEL_REFERENCE_POINT`, not `disc_centroid_ground_truth`, and may sit anterior/posterior to the disc. Full 3D nearest/Hungarian distances and a longitudinal-axis projection (onto the frozen predicted axis, sign never chosen from GT) are reported side by side to separate AP offset from level-position error. `GATE_G` stays `UNRESOLVED` -- never `FAIL` -- until this diagnostic determines the semantically correct evaluation criterion (full 3D centroid proximity, longitudinal proximity, sequence matching, or a combination).

**Question 1** (this smoke addresses): does 67A detect a disc instance close to each `ABSOLUTE_LEVEL_REFERENCE_POINT`? **Question 2** (NOT resolved by this smoke): can we infer the absolute lumbar level without looking at GT? A good nearest-matching result does NOT by itself resolve Question 2 -- `relative_sequence_monotonic_up_to_reversal=True` can hold even when the predicted direction is reversed vs the absolute labels (see `sequence` per case); GT is used only for this post-hoc scoring, never to pick the PCA sign, orientation, which instance to keep, or any threshold.

## Stage D -- NOT executed in this run (documented only)

The absolute-anchor training-dataset design (Stage D) is documented and gate-scaffolded but intentionally not implemented/trained in this iteration.

## Decision

- stage_ab_decision: `BLOCKED` (blocker: `RSNA_DATASET_NOT_AVAILABLE_IN_THIS_RUN`)
- overall_decision_67b: `BLOCKED`
- Blocking gates (informational -- includes Stage C/D, not all are Stage A/B blockers): `['GATE_A_RSNA_dataset_structure', 'GATE_B_absolute_level_reference', 'GATE_C_split_leakage', 'GATE_D_series_annotation_mapping', 'GATE_E_coordinate_physical_parity', 'GATE_F_frozen_67A_execution_on_RSNA', 'GATE_G_pred_to_absolute_GT_matching', 'GATE_H_training_dataset_readiness']`
- ABSOLUTE_LEVEL_TRAINING_DATASET_READY: `NO` (never automatic YES in this iteration)

## Quality gates

| gate                                | status   |
|:------------------------------------|:---------|
| GATE_A_RSNA_dataset_structure       | NOT_RUN  |
| GATE_B_absolute_level_reference     | NOT_RUN  |
| GATE_C_split_leakage                | NOT_RUN  |
| GATE_D_series_annotation_mapping    | NOT_RUN  |
| GATE_E_coordinate_physical_parity   | NOT_RUN  |
| GATE_F_frozen_67A_execution_on_RSNA | NOT_RUN  |
| GATE_G_pred_to_absolute_GT_matching | NOT_RUN  |
| GATE_H_training_dataset_readiness   | NO       |
| GATE_I_privacy                      | PASS     |

## What this notebook does NOT do

- no training performed (segmenter, classifier, or any model)
- no checkpoint modified
- no Notebook 67 / 67A modification
- no axial cluster-level pairing
- no cross-frame registration
- no pathology grading
- no official RSNA test access (officialTestAccessed=False, structurally enforced)
- no internal_test access (RSNA_INTERNAL_TEST_LOCKED=True, only the 296-study validation split is eligible for smoke selection)
- no full validation batch (296 studies) -- only a deterministic 3-study smoke
- no SPIDER test access
- no assumption that a study must have exactly 5 visible levels
- no GT-informed inference decisions (PCA sign, orientation, instance selection, thresholds are never chosen from GT)

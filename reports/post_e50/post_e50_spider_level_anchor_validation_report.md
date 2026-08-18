# Post-E50 SPIDER Level Anchor Validation

## Objective

Validate the Notebook 67 sagittal level-localization pipeline against SPIDER held-out data and study how to reproducibly resolve disc instance detection, disc ordering, target lumbar window selection, absolute level anchoring, L1-L2...L5-S1 naming, and abstention.

## Execution model

- Stage A (local development, this run): `COLAB_EXECUTION_REQUIRED`
- Stage B (Google Colab, manual): mount Drive, locate SPIDER, verify dataset/splits, leakage audit, smoke test, validation batch -- to be run by the user.

## Checkpoint

- checkpoint_sha256: `cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944`
- Matches Notebook 67 checkpoint: `True`
- GATE A: `PASS`

## Colab/Drive/SPIDER configuration

- IN_COLAB: `False`
- PFI_AI_REPO_ROOT source: `auto_detected_local`, valid: `True`
- PFI_DRIVE_ROOT source: `not_applicable_outside_colab`
- SPIDER_AVAILABLE: `False`, selection status: `NOT_FOUND`

## Dataset verification / splits / leakage

- GATE B (dataset structure): `NOT_RUN`
- GATE C (split leakage audit): `NOT_RUN` (function verified with synthetic self-tests; not yet run against real SPIDER splits)

## Smoke test design

- Cohort selection: deterministic (seed=2026, size=3), verified order-independent via synthetic self-test.
- GATE D: `NOT_RUN`

## Reused algorithm (Notebook 67 baseline)

Geometry primitives, slice quality score, connected-component instance extraction, multi-slice consensus (union-find on centroid proximity), spine-axis PCA (patient-Z sign-corrected), and instance confidence were reproduced verbatim from Notebook 67 and verified with synthetic self-tests (all PASSED locally).

## Matching, FOV analysis, anchor research, level naming (functions defined and self-tested)

- Hungarian instance matching + detection metrics: synthetic self-test PASSED.
- FOV candidate analysis: schema defined; deferred to Colab (needs real ground truth).
- Anchor methods A-D: defined; Method A is structurally unavailable for this checkpoint (no sacral/S1 class).
- Absolute level naming + level metrics: synthetic self-test PASSED (perfect-window and full-abstention cases).

## Quality gates

| gate                                    | status   |
|:----------------------------------------|:---------|
| GATE_A_checkpoint_identity              | PASS     |
| GATE_B_dataset_structure_verified       | NOT_RUN  |
| GATE_C_split_leakage_audit              | NOT_RUN  |
| GATE_D_smoke_test                       | NOT_RUN  |
| GATE_E_frozen_baseline_batch            | NOT_RUN  |
| GATE_F_instance_localization_gt_metrics | NOT_RUN  |
| GATE_G_fov_analysis                     | NOT_RUN  |
| GATE_H_absolute_level_anchor            | NOT_RUN  |
| GATE_I_level_naming_metrics             | NOT_RUN  |
| GATE_J_privacy                          | PASS     |

## Results

- Overall quality gate: `PARTIAL`
- Decision: `BLOCKED`
- ready_for_axial_cluster_pairing: `False`

## Test lock

- TEST_SPLIT_LOCKED: `True` (structurally enforced, not just documented)
- test_patients_used: `0`

## Limitations

- This is Stage A (local development) only: no SPIDER data was processed in this run.
- All SPIDER-dependent gates (B-I) are NOT_RUN, not PASS/FAIL -- they require Stage B (Colab) execution.
- Anchor Method A (inferior disc/sacral context) is structurally unavailable for the sagittal_spider checkpoint: it has no dedicated sacral/S1 class.
- The pure functions in this notebook (consensus, spine axis, matching, metrics, level naming) were verified only against small synthetic self-tests, not real anatomy.
- No training was performed; no checkpoint was modified; AUTOMATIC_DISC_LOCALIZATION_VALIDATED was not touched.
- TEST_SPLIT_LOCKED=True structurally prevents test execution in this notebook; test promotion requires a separate future experiment.
- Dataset structure (folder layout, class IDs, manifest schema) for SPIDER was not assumed and was not verifiable locally -- Colab execution must confirm it before any parsing logic is trusted.
- License/attribution metadata for SPIDER is UNKNOWN_NOT_VERIFIED pending local/Colab inspection.
- Dataset structure verification skipped: SPIDER_AVAILABLE=False (Stage A local run).
- Split discovery skipped: SPIDER_AVAILABLE=False (Stage A local run).
- Leakage audit not run against real data: SPIDER split_inventory unavailable in this Stage-A run.
- spider_dataset_inventory and spider_ground_truth_discs are empty schemas only: populate in Colab once SPIDER structure is confirmed (GATE B) -- not fabricated here.
- Smoke test NOT RUN: SPIDER unavailable and/or checkpoint identity gate not satisfied in this Stage-A run.
- FOV analysis NOT RUN: no SPIDER ground truth available in this Stage-A run.
- Anchor methods A-D are defined and unit-testable, but were not run against real cases in this Stage-A run (no SPIDER data).
- No figures generated in this Stage-A run (no real SPIDER results). Required in Colab: ['smoke_test_overlay.png', 'predicted_disc_count_distribution.png', 'gt_disc_count_fov_distribution.png', 'predicted_vs_gt_instance_count.png', 'centroid_error_distribution.png', 'centroid_error_by_level.png', 'detection_precision_recall_summary.png', 'level_confusion_matrix.png', 'abstention_accuracy_tradeoff.png', 'example_best_case.png', 'example_median_case.png', 'example_worst_case.png']

## What 67A proves (this run)

> The Notebook 67 algorithm was extracted into reusable, documented functions and verified correct against small synthetic self-tests (consensus grouping, spine-axis sign, Hungarian matching, detection metrics, level-naming abstention logic). The frozen checkpoint was verified byte-identical to the one used in Notebook 67. Colab/Drive/SPIDER path resolution is implemented and was exercised in local mode.

## What 67A does not prove (this run)

- no SPIDER data was processed -- no real dataset/split/leakage verification occurred
- no smoke test or validation batch was run against real cases
- no FOV, anchor, or level-naming evidence exists yet against ground truth
- no claim of ABSOLUTE_LEVEL_ANCHOR_VALIDATED_ON_VALIDATION is made
- test split was never touched

## Ready for Notebook 67B?

`ready_for_axial_cluster_pairing = False`. Requires Stage B (Colab) execution with GATE H and GATE I both PASS before this can become true.

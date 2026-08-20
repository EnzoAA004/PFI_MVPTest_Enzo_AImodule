# Post-E50 -- 67B1 Absolute Level Orientation + Lumbar Window

- base_commit (67B closing): `351d9a5ba63b3cfffea91180c0075f778a858e20`
- frozen_checkpoint_sha256: `cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944`
- decision: **UNRESOLVED**
- training_performed: False | internal_test_locked: True | official_test_accessed: False
- AUTOMATIC_DISC_LOCALIZATION_VALIDATED: False (unchanged)

## Gates
```json
{
  "GATE_A_RSNA_dataset_structure": "NOT_RUN",
  "GATE_C_split_leakage": "NOT_RUN",
  "GATE_D_series_annotation_mapping": "NOT_RUN",
  "GATE_E_coordinate_physical_parity_real": "NOT_RUN",
  "GATE_checkpoint_identity_67b1": "PASS",
  "GATE_67B1_A_AXIS_DIRECTION_WITHOUT_GT": "NOT_RUN",
  "GATE_67B1_B_LUMBAR_WINDOW_WITHOUT_GT": "NOT_RUN",
  "GATE_67B1_C_ABSOLUTE_LEVEL_SEQUENCE": "NOT_RUN",
  "GATE_67B1_D_PRIVACY": "PASS"
}
```

## Q1 -- Is orientation (cranial/caudal) resolvable without GT?
NOT_RUN in this environment (RSNA data unavailable locally) -- see `warnings` for details.

## Q2 -- Is the lumbar 5-disc window selectable without GT via geometry alone?
NOT_RUN in this environment (RSNA data unavailable locally) -- see `warnings` for details.

## Q3 -- What is the combined direction+window match rate vs RSNA GT?
NOT_RUN in this environment (RSNA data unavailable locally) -- see `warnings` for details.

## Q4 -- Does the system abstain instead of forcing a wrong answer?
Yes -- abstention is a first-class outcome, never bypassed. Prediction status breakdown this run: {'ABSOLUTE_SEQUENCE_PREDICTED': 0, 'ABSTAIN_DIRECTION': 0, 'ABSTAIN_WINDOW': 0, 'ABSTAIN_BOTH': 0, 'INSUFFICIENT_PREDICTED_INSTANCES': 0}. Abstention reason breakdown: {'INSUFFICIENT_DISC_INSTANCES': 0, 'AXIS_DIRECTION_UNRESOLVED': 0, 'WINDOW_HEURISTICS_DISAGREE': 0, 'FOV_INCOMPLETE': 0, 'REQUIRED_DICOM_GEOMETRY_UNAVAILABLE': 0}.

## Q5 -- Is a supervised scorer actually necessary, or does geometry already suffice?
Cannot be answered yet in this environment -- no RSNA execution occurred this run. Requires a real Colab run on Stage 1+2 (and eventually Stage 3) before this question is answerable.

## Warnings
- STAGE A NOT_RUN: RSNA dataset not available in this run.
- GATE_C split leakage audit NOT_RUN: GATE_A did not PASS in this run.
- Sagittal T2/STIR + Spinal Canal Stenosis reference table NOT_RUN: GATE_A did not PASS or required columns missing.
- Primary Sagittal T2/STIR Spinal Canal Stenosis annotation-to-DICOM geometry mapping NOT_RUN.
- RSNA smoke cohort selection NOT_RUN: prerequisites (GATE_A/GATE_C/level schema) not met in this run.
- Development cohort selection NOT_RUN: prerequisites (GATE_A/GATE_C/level schema) not met in this run.
- End-to-end 67B1 execution NOT_RUN: RSNA unavailable, checkpoint identity did not PASS, or smoke cohort is empty in this run.

## Limitations
- Heuristic (E) vertebral/canal context was NOT implemented: not directly derivable from the frozen segmentation without a new model.
- GATE_67B1_A/B/C are capped at PARTIAL in this notebook -- PASS requires an explicitly-approved future locked-validation phase (Stage 3, RUN_LOCKED_VALIDATION=True), not executed here.
- Development cohort is 30 TRAIN-split studies (deterministic, seed=2026) -- not the full 1382-study TRAIN split, and never the validation/internal_test splits used for locked evaluation.
- No supervised scorer was trained (Sections 15/16 fallback intentionally not implemented) -- this notebook only measures whether pure geometry suffices.
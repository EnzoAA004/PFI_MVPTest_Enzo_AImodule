# P10.6 AI Runtime Integration Checkpoint

- Branch: `enzo/p10-6-ai-rsna-findings`
- Base HEAD at implementation start: `60d8b0ab71ab385651891733cdac1dbcfa4d48bb`
- Observed remote main: `06dc0c7b05ad70cff829e820bec1502ddd95da43`
- Model: `rsna_subarticular_axial_t2_2p5d`
- Frozen checkpoint file name: `frozen_subarticular_checkpoint.pt`
- Required SHA-256: `d41262d57b13c146a48ab15f5e183cc6a55fc92724b7d0c286cea1f2ce26e84a`
- Required epoch: `6`
- Contract: `pfi.degenerative-findings.v1`
- Finding type: `subarticular_stenosis`
- Review: `humanReviewRequired=true`, `notClinicalDiagnosis=true`, `autonomousDiagnosis=false`

## Integration Status

- API endpoint: `POST /degenerative-findings/subarticular/predict`
- Input resolution: registered `inputId` only; client filesystem paths are rejected.
- Runtime loading: lazy cache with reload on checkpoint path, device, mtime or size changes.
- Health/readiness/models: status block exposed without loading the checkpoint and without paths.
- Existing segmentation models: unchanged.
- Synthetic tests: pending final local execution in this working tree.
- REAL_CHECKPOINT_SMOKE: NOT_RUN unless `PFI_SUBARTICULAR_CHECKPOINT_PATH` is configured locally.

## Limitations

The endpoint requires external operator-provided ROI coordinates. It does not
implement or claim automatic anatomical localization, autonomous diagnosis,
patient-specific validated pathology detection or treatment recommendation.

## Merge Readiness

Status before final validation: READY_FOR_CHECKPOINT_REVIEW, not READY_FOR_MERGE.
The branch must not be merged until the full requested validation output is
reviewed and the real checkpoint smoke is run when the artifact is available.

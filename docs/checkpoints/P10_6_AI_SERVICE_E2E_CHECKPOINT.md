# P10.6 AI Service E2E Checkpoint

- Branch: `enzo/p10-6-ai-rsna-integration-clean`
- Base main SHA: `06dc0c7b05ad70cff829e820bec1502ddd95da43`
- Historical source branch: `origin/enzo/p10-6-ai-rsna-findings`
- Historical source SHA: `61663f9d4c07a16b1a8ddcc84331f292b65dcc3a`
- Model: `rsna_subarticular_axial_t2_2p5d`
- Frozen checkpoint file: `frozen_subarticular_checkpoint.pt`
- Checkpoint SHA-256: `d41262d57b13c146a48ab15f5e183cc6a55fc92724b7d0c286cea1f2ce26e84a`
- Contract: `pfi.degenerative-findings.v1`
- Finding type: `subarticular_stenosis`

## Final State

- FRONTEND_CONTRACT_READY = true
- AI_RUNTIME_TECHNICALLY_VALIDATED = true
- REAL_CHECKPOINT_SMOKE = true
- AI_SERVICE_E2E_VALIDATED = true
- READY_FOR_MERGE = true
- FULL_PRODUCT_E2E_VALIDATED = false
- CLOSED = false

## Evidence

- Clean branch created from `origin/main`.
- Python compileall passed with a temporary pycache outside OneDrive.
- Full AI service test suite passed: `296 passed, 5 skipped, 1 warning`.
- Docker build passed: `pfi-ai-module:p10-6-clean-e2e`.
- Docker image digest: `sha256:3bfe1f94116622432ac04d28f19bdc9500ee141d1219a29a8c69c51af453213a`.
- Docker image size: `430082835` bytes.
- Runtime versions: `torch=2.13.0+cpu`, `torchvision=0.28.0+cpu`, `timm=1.0.28`.
- Existing segmentation models remained visible: `sagittal_spider`, `axial_t2_alkafri`.
- Service status before inference: `configured=true`, `artifactPresent=true`, `loaded=false`, `status=available`.
- Direct checkpoint smoke passed in Docker.
- Real HTTP ingestion passed through `POST /inputs/study`.
- Real axial `inputId` was obtained from the public ingestion response.
- Real HTTP prediction passed through `POST /degenerative-findings/subarticular/predict`.
- Runtime status after inference: `configured=true`, `artifactPresent=true`, `loaded=true`, `status=loaded`.

## Sanitized E2E Sample

- Manifest split: `train`.
- Sample hash: `eb6095e6f6e9`.
- Uploaded axial series file count: `43`.
- Sanitized inputId prefix: `inp_4d85`.
- Request: `instanceNumber=3`, `side=left`, `level=L1-L2`, rounded coordinates only.
- Prediction: `findingType=subarticular_stenosis`, `label=normal_mild`, `probabilitySum=1.0`.
- Review status: `pending`.
- Localization: `external_coordinate`, `researchOnly=true`.

No full study ID, series UID, patient identifier, local path, mounted path or
DICOM UID was printed as public evidence.

## Negative Checks

- Missing inputId: 404.
- Invalid side: 400.
- Invalid level: 400.
- Out-of-range coordinate: 400.
- Extra `path` field: 422.
- Extra `checkpointPath` field: 422.
- Extra DICOM identifier field: 422.
- Service without checkpoint: health 200 and predict 503 with
  `SUBARTICULAR_CHECKPOINT_UNAVAILABLE`.

All negative responses were sanitized and did not expose stack traces, local
paths, mounted paths, patient identifiers or DICOM UIDs.

## Governance

- `humanReviewRequired=true`
- `notClinicalDiagnosis=true`
- `autonomousDiagnosis=false`
- No clinical diagnosis.
- No treatment recommendation.
- No automatic ROI localization.

## Scope And Remaining Work

This checkpoint validates the AI Module HTTP E2E path only. It does not validate
the full product E2E with Backend and Frontend.

The classifier still requires external anatomical coordinates. A validated
automatic ROI localizer remains outside this checkpoint.

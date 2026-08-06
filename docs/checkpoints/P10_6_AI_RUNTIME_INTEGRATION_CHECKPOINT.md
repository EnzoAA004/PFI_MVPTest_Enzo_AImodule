# P10.6 AI Runtime Integration Checkpoint

- Branch: `enzo/p10-6-ai-rsna-findings`
- Base HEAD at implementation start: `60d8b0ab71ab385651891733cdac1dbcfa4d48bb`
- Observed remote main: `06dc0c7b05ad70cff829e820bec1502ddd95da43`
- Validated HEAD: `c9b55f210c860b53220e6719c6f4ddebe33cf104`
- Model: `rsna_subarticular_axial_t2_2p5d`
- Frozen checkpoint file name: `frozen_subarticular_checkpoint.pt`
- Required SHA-256: `d41262d57b13c146a48ab15f5e183cc6a55fc92724b7d0c286cea1f2ce26e84a`
- Required epoch: `6`
- Contract: `pfi.degenerative-findings.v1`
- Finding type: `subarticular_stenosis`
- Review: `humanReviewRequired=true`, `notClinicalDiagnosis=true`, `autonomousDiagnosis=false`

## Final Technical Validation

- REAL_CHECKPOINT_HASH: PASSED
- REAL_CHECKPOINT_LOCAL_SMOKE: PASSED
- REAL_CHECKPOINT_DOCKER_SMOKE: PASSED
- REAL_CHECKPOINT_SMOKE: PASSED
- STRICT_STATE_DICT_LOAD: PASSED
- CONTRACT_VALIDATION: PASSED
- PROBABILITIES_FINITE: PASSED
- PROBABILITIES_NORMALIZED: PASSED
- ARGMAX_MATCHES_LABEL: PASSED
- API_IMPORT: PASSED
- DOCKER_BUILD: PASSED
- DOCKER_SERVICE_HEALTH: PASSED
- FULL_TEST_SUITE: PASSED

Full suite summary:

- 255 passed
- 6 skipped
- 87 warnings
- 0 failed
- Duration: 74.09s

The skipped tests do not invalidate the real checkpoint smoke because the
checkpoint was executed directly through
`scripts/smoke_test_subarticular_runtime.py` both locally and inside the Docker
container.

## Docker Validation

- Docker Client: 28.5.1
- Docker Engine: 28.5.1
- Docker Desktop: 4.49.0
- Docker context: `desktop-linux`
- Platform: `linux/amd64`
- Image: `pfi-ai-module:p10-6-checkpoint`
- Image digest: `sha256:40133726ab097311807ee77c67732be7260f0b10f2c75cd90b34d561cb97f780`
- torch: `2.13.0+cpu`
- torchvision: `0.28.0+cpu`
- timm: `1.0.28`

The frozen checkpoint was mounted read-only from outside the repository. It was
not copied into the repository and was not incorporated into the Docker image.

## Service State

- `/health`: 200
- `/readiness`: 200
- `/models`: 200
- `/models/runtime`: 200
- `configured=true`
- `artifactPresent=true`
- `loaded=false` before inference
- `status=available`
- No paths were exposed in service responses.
- `sagittal_spider` remains available.
- `axial_t2_alkafri` remains available.

Querying health/readiness/models endpoints does not load the checkpoint.

## Governance

- `humanReviewRequired=true`
- `notClinicalDiagnosis=true`
- `autonomousDiagnosis=false`
- `officialTestAccessed=false`
- No clinical diagnosis.
- No treatment recommendation.
- No automatic ROI localization.

## Pending Separate Evidence

- ENDPOINT_REAL_ROI_SMOKE: NOT_RUN_NO_REGISTERED_INPUT

A real POST to `/degenerative-findings/subarticular/predict` with a registered
axial series was not executed because there was no real axial `inputId`
registered inside the validation container.

This does not invalidate the technical compatibility of the frozen checkpoint,
but it remains pending for the functional end-to-end product test. It must not
be covered by fabricating an `inputId` or by accepting arbitrary filesystem
paths.

## Integration Status

- API endpoint: `POST /degenerative-findings/subarticular/predict`
- Input resolution: registered `inputId` only; client filesystem paths are rejected.
- Runtime loading: lazy cache with reload on checkpoint path, device, mtime or size changes.
- Health/readiness/models: status block exposed without loading the checkpoint and without paths.
- Existing segmentation models: unchanged.
- Synthetic tests: executed as part of the full suite.
- REAL_CHECKPOINT_SMOKE: PASSED.

## Limitations

The endpoint requires external operator-provided ROI coordinates. It does not
implement or claim automatic anatomical localization, autonomous diagnosis,
patient-specific validated pathology detection or treatment recommendation.

## Pending Risks

1. The model requires external anatomical coordinates.
2. There is no validated automatic ROI localizer.
3. The virtual environment under OneDrive produced `WinError 206`; validation
   was completed in a clean virtual environment under `C:\tmp`.
4. Docker currently installs torch, torchvision and timm without exact pins.
5. Before integrating to main, pin the validated versions and rebuild the image.
6. The branch is diverged and must not be merged directly into main.

## Merge Readiness

Status after final technical validation: READY_FOR_CHECKPOINT_COMMIT, not
READY_FOR_MERGE, not CLOSED and not E2E_VALIDATED.

The technical integration of the frozen checkpoint is validated and ready for a
checkpoint documentation commit. The HTTP E2E test with a real registered axial
series remains pending and will be executed after porting the integration to a
clean branch based on main.

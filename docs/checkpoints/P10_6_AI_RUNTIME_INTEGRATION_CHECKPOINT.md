# P10.6 AI Runtime Integration Checkpoint

## Identification

- Branch: `enzo/p10-6-ai-rsna-findings`
- Base HEAD at implementation start: `60d8b0ab71ab385651891733cdac1dbcfa4d48bb`
<<<<<<< HEAD
- Observed remote main: `06dc0c7b05ad70cff829e820bec1502ddd95da43`
- Validated HEAD: `c9b55f210c860b53220e6719c6f4ddebe33cf104`
=======
- Validated HEAD before this documentation commit: `c9b55f210c860b53220e6719c6f4ddebe33cf104`
- Observed remote main at implementation time: `06dc0c7b05ad70cff829e820bec1502ddd95da43`
>>>>>>> 61663f9d4c07a16b1a8ddcc84331f292b65dcc3a
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
<<<<<<< HEAD
- Health/readiness/models: status block exposed without loading the checkpoint and without paths.
- Existing segmentation models: unchanged.
- Synthetic tests: executed as part of the full suite.
- REAL_CHECKPOINT_SMOKE: PASSED.
=======
- Health, readiness and models endpoints expose status without loading the checkpoint and without exposing local paths.
- Existing segmentation models remain available and are not replaced by the degenerative-finding classifier.
- The frozen checkpoint is configured through `PFI_SUBARTICULAR_CHECKPOINT_PATH` and is not stored in Git or embedded in the Docker image.
>>>>>>> 61663f9d4c07a16b1a8ddcc84331f292b65dcc3a

## Final Technical Validation

- `REAL_CHECKPOINT_HASH = PASSED`
- `REAL_CHECKPOINT_LOCAL_SMOKE = PASSED`
- `REAL_CHECKPOINT_DOCKER_SMOKE = PASSED`
- `REAL_CHECKPOINT_SMOKE = PASSED`
- `STRICT_STATE_DICT_LOAD = PASSED`
- `CONTRACT_VALIDATION = PASSED`
- `PROBABILITIES_FINITE = PASSED`
- `PROBABILITIES_NORMALIZED = PASSED`
- `ARGMAX_MATCHES_LABEL = PASSED`
- `API_IMPORT = PASSED`
- `DOCKER_BUILD = PASSED`
- `DOCKER_SERVICE_HEALTH = PASSED`
- `FULL_TEST_SUITE = PASSED`

<<<<<<< HEAD
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
=======
The real frozen checkpoint was loaded locally and inside the Docker container. Both smoke executions returned:

```json
{
  "status": "SUBARTICULAR_REAL_CHECKPOINT_SMOKE_OK",
  "checkpointHashVerified": true,
  "modelLoaded": true,
  "device": "cpu",
  "probabilitiesFinite": true,
  "probabilitiesNormalized": true,
  "argmaxMatchesLabel": true,
  "contractValidated": true,
  "humanReviewRequired": true,
  "notClinicalDiagnosis": true,
  "autonomousDiagnosis": false
}
```

This smoke validation proves technical artifact compatibility with the runtime. It does not constitute clinical validation, a new quality evaluation or authorization for autonomous use.

## Test Suite

Final full AI service suite:

- Passed: `255`
- Skipped: `6`
- Warnings: `87`
- Failed: `0`
- Duration: `74.09s`

The skipped optional real-checkpoint pytest does not invalidate the evidence because the actual frozen checkpoint was executed directly with `scripts/smoke_test_subarticular_runtime.py` both locally and inside Docker.

Additional checks:

- `python -m compileall ai_service/pfi_ai_service`: passed.
- FastAPI application import: passed with title `PFI AI Service`.
- `git diff --check`: passed before the documentation update.

## Docker Validation

- Docker Client: `28.5.1`
- Docker Engine: `28.5.1`
- Docker Desktop: `4.49.0`
- Context: `desktop-linux`
- Platform: `linux/amd64`
- Image tag: `pfi-ai-module:p10-6-checkpoint`
- Image digest: `sha256:40133726ab097311807ee77c67732be7260f0b10f2c75cd90b34d561cb97f780`
- PyTorch: `2.13.0+cpu`
- torchvision: `0.28.0+cpu`
- timm: `1.0.28`

The checkpoint was mounted as a read-only file for the Docker smoke. It was not copied into the repository or image.

## Service Status with Mounted Checkpoint

The container started healthy and returned HTTP `200` for:

- `GET /health`
- `GET /readiness`
- `GET /models`
- `GET /models/runtime`

Reported subarticular runtime status before inference:

- `configured=true`
- `artifactPresent=true`
- `loaded=false`
- `status=available`
- `checkpointHashStatus=not_checked`
- `humanReviewRequired=true`
- `notClinicalDiagnosis=true`
- `autonomousDiagnosis=false`

Health and readiness checks did not load the checkpoint. No local checkpoint path or mounted container path was exposed in public responses.

Existing segmentation models remained available:

- `sagittal_spider`
- `axial_t2_alkafri`

## Governance

The integration preserves the following non-negotiable controls:

- `humanReviewRequired=true`
- `notClinicalDiagnosis=true`
- `autonomousDiagnosis=false`
- `officialTestAccessed=false`
- no clinical diagnosis claim
- no treatment recommendation
- no automatic anatomical ROI localization claim
- no threshold, hyperparameter or checkpoint reselection based on the internal test

The model produces research-only candidate findings that require professional review.

## Pending Separate E2E Evidence

- `ENDPOINT_REAL_ROI_SMOKE = NOT_RUN_NO_REGISTERED_INPUT`

A real HTTP `POST /degenerative-findings/subarticular/predict` was not executed with a registered axial series because no valid axial `inputId` was prepared inside that container session.

This does not invalidate technical compatibility of the frozen checkpoint, strict model loading or contract generation. It remains a separate product-level end-to-end validation step and must not be replaced by fabricating an `inputId` or accepting arbitrary filesystem paths.

## Limitations and Risks

1. The classifier requires external operator-provided anatomical ROI coordinates.
2. No validated automatic ROI localizer or segmentation-to-coordinate bridge is implemented.
3. The repository virtual environment under OneDrive encountered `WinError 206` while installing the validated PyTorch stack; final validation used a clean temporary environment under `C:\tmp`.
4. The Dockerfile currently installs `torch`, `torchvision` and `timm` without exact version pins.
5. Before integration into main, the validated dependency versions should be pinned and the image rebuilt and retested.
6. The source branch is historically diverged from main and must not be merged directly without a clean integration strategy and full regression validation.
7. Product-level E2E HTTP validation with a real registered axial series remains pending.

## Checkpoint State

- `READY_FOR_CHECKPOINT_COMMIT`
- `READY_FOR_MERGE = false`
- `E2E_VALIDATED = false`
- `CLOSED = false`

The frozen checkpoint integration is technically validated and ready for this documentation checkpoint commit. The real HTTP E2E test with a registered axial series remains pending and should be performed after the reviewed integration is ported to a clean branch based on the current main branch.
>>>>>>> 61663f9d4c07a16b1a8ddcc84331f292b65dcc3a

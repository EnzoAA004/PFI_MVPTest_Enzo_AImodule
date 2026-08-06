# P10.6-AI - Subarticular frozen classifier runtime

This document records the runtime integration for the frozen RSNA subarticular
stenosis classifier. The model output is a research-only degenerative finding
candidate and always requires professional review. It is not a clinical diagnosis,
not an autonomous diagnosis, and not a treatment recommendation.

## Frozen checkpoint

The runtime does not download model weights. Configure the checkpoint path with:

```text
PFI_SUBARTICULAR_CHECKPOINT_PATH=/path/to/frozen_subarticular_checkpoint.pt
```

If the variable is absent, development fallback resolution points to:

```text
PFI_MVP/models/P10_6_rsna_findings/subarticular_axial_t2_2p5d/final_internal_test_evaluation/frozen_subarticular_checkpoint.pt
```

The required SHA-256 is:

```text
d41262d57b13c146a48ab15f5e183cc6a55fc92724b7d0c286cea1f2ce26e84a
```

The runtime rejects missing files, hash mismatches, incompatible checkpoint
configuration, invalid state dicts, and unavailable CUDA requests.

## Model configuration

- `model_name`: `efficientnet_b0`
- `task`: `subarticular_stenosis_left_right`
- `sequence`: `Axial T2`
- `input_channels`: `3`
- `crop_size`: `256`
- `image_size`: `224`
- classes: `normal_mild`, `moderate`, `severe`
- `humanReviewRequired=true`
- `notClinicalDiagnosis=true`
- `autonomousDiagnosis=false`

Final internal-test metrics recorded by Notebook 64 are exposed as provenance only.
They must not be used to retrain, tune thresholds, reselect checkpoints or relax the
human-review requirement.

## Input modes

The integration supports two controlled input modes:

1. Prepared 2.5D runtime input: an array/tensor with three channels representing
   the same three-slice crop format used by Notebook 63/64. This is intended for
   tests and the first integration layer.
2. Explicit ROI input: an Axial T2 DICOM series plus `instance_number`, `x`, `y`,
   `side` and `level`. The runtime reuses the training crop/stack normalization.

The classifier was trained from dataset-provided anatomical coordinates. The AI
Module does not yet provide a validated bridge from generated segmentations to the
coordinates expected by this classifier. No automatic anatomical ROI localizer is
implemented or claimed here.

## Output contract

Predictions are mapped to the existing root contract:

```text
degenerativeFindings.schemaVersion = pfi.degenerative-findings.v1
```

The finding uses:

- `findingType=subarticular_stenosis`
- severity label by argmax over `normal_mild`, `moderate`, `severe`
- probabilities normalized by softmax and checked for finite values
- `sourceSeries.role=axial_t2`
- `localization.source=external_coordinate` for ROI inputs
- `localization.researchOnly=true`
- `review.required=true`
- `review.status=pending`
- `notClinicalDiagnosis=true`

No patient name, patient identifier, StudyInstanceUID, SeriesInstanceUID,
SOPInstanceUID or other original DICOM identifiers are included in the public
finding payload.

## Runtime limits

- CPU and CUDA are supported; CUDA must be available when requested.
- The checkpoint is loaded with strict state-dict validation.
- The model is placed in `eval()` and all parameters have `requires_grad=false`.
- Inference runs inside `torch.inference_mode()`.
- No clinical threshold, calibration override or treatment recommendation is added.
- The official RSNA test set is not accessed.


## AI service API

The classifier is exposed through a dedicated research-only endpoint and is not
registered as a segmentation model:

```http
POST /degenerative-findings/subarticular/predict
```

Request:

```json
{
  "inputId": "inp_internal_registered_id",
  "instanceNumber": 9,
  "x": 192.5,
  "y": 210.0,
  "side": "left",
  "level": "L4-L5"
}
```

Response root:

```json
{
  "degenerativeFindings": {
    "schemaVersion": "pfi.degenerative-findings.v1",
    "findings": []
  },
  "model": {
    "modelId": "rsna_subarticular_axial_t2_2p5d",
    "checkpointSha256": "d41262d57b13c146a48ab15f5e183cc6a55fc92724b7d0c286cea1f2ce26e84a",
    "device": "cpu"
  },
  "humanReviewRequired": true,
  "notClinicalDiagnosis": true,
  "autonomousDiagnosis": false,
  "warnings": ["roi_requires_external_anatomical_coordinate"]
}
```

The request intentionally does not accept filesystem paths, DICOM UIDs, patient
identifiers, checkpoint overrides or localization claims. `side`, `level`, `x`,
`y` and `instanceNumber` are operator-provided research coordinates until a
validated localizer exists.

Expected public errors:

- `400`: invalid ROI, side, level or coordinates.
- `404`: registered input not found.
- `409`: registered input exists but is not axial.
- `422`: registered input is axial but not a compatible DICOM series.
- `503`: checkpoint not configured or not installed.
- `500`: checkpoint hash mismatch or incompatible frozen checkpoint.

The error payload is sanitized and must not contain local paths, stack traces,
internal hosts, tokens or original DICOM identifiers.

## Runtime status

`/health`, `/warmup`, `/readiness`, `/models` and `/models/runtime` publish a
`degenerativeFindingModels.subarticular` block with status only. These endpoints
do not load the checkpoint and do not expose `PFI_SUBARTICULAR_CHECKPOINT_PATH`.
The segmentation models remain reported separately as segmentation models.

## Smoke test

PowerShell:

```powershell
$env:PFI_SUBARTICULAR_CHECKPOINT_PATH="C:\ruta\frozen_subarticular_checkpoint.pt"
$env:PYTHONPATH="ai_service"
python scripts/smoke_test_subarticular_runtime.py
```

Bash:

```bash
export PFI_SUBARTICULAR_CHECKPOINT_PATH="/ruta/frozen_subarticular_checkpoint.pt"
export PYTHONPATH="ai_service"
python scripts/smoke_test_subarticular_runtime.py
```

The smoke test uses a deterministic synthetic 3x224x224 tensor. It verifies
artifact compatibility, SHA validation, probability normalization and contract
shape only. It does not validate clinical quality and does not access internal or
official RSNA test cases.

Do not commit the `.pt`, DICOM files, cache directories or generated clinical
outputs to Git.

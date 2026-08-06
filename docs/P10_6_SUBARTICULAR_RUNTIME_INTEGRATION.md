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

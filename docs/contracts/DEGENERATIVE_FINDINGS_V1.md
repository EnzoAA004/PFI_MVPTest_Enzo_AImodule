# Degenerative Findings V1

Version: `pfi.degenerative-findings.v1`

This contract defines research outputs for **hallazgos degenerativos asociados a estenosis lumbar**. It is a data contract only: it does not connect the RSNA models to productive inference, does not add training, and does not create a clinical diagnostic output.

## Scope

Allowed finding types:

- `central_canal_stenosis`
- `neural_foraminal_narrowing`
- `subarticular_stenosis`

Allowed severity labels:

- `normal_mild`
- `moderate`
- `severe`

Out of scope for this version:

- protrusion;
- extrusion;
- disc dehydration;
- other pathologies not trained by the current models;
- productive localization integration;
- clinical diagnosis, disease or lesion claims.

## Root Shape

```json
{
  "schemaVersion": "pfi.degenerative-findings.v1",
  "findings": []
}
```

Each finding contains:

- stable `findingId`;
- `findingType`;
- `anatomy.level` and `anatomy.side`;
- `classification.label`;
- `classification.probabilities` for `normal_mild`, `moderate`, `severe`;
- `evaluation.status` and optional `evaluation.reasonCode`;
- `sourceSeries.role` and `sourceSeries.position`;
- `localization.source` and `localization.researchOnly`;
- `model.modelId` and `model.modelSha256`;
- `review.required=true` and `review.status`;
- `notClinicalDiagnosis=true`.

## Validation Rules

- Probabilities must be finite, between 0 and 1, and sum to 1 within tolerance.
- `classification.label` must match the maximum probability.
- `central_canal_stenosis` requires `anatomy.side=null`.
- `neural_foraminal_narrowing` and `subarticular_stenosis` require `anatomy.side=left|right`.
- Valid levels are `L1-L2`, `L2-L3`, `L3-L4`, `L4-L5`, `L5-S1`.
- `sourceSeries.role` must be one of `sagittal_t2`, `sagittal_t1`, `axial_t2`.
- `sourceSeries.position` is a non-negative integer.
- `evaluation.status` is one of `evaluated`, `not_evaluated`, `unsupported`, `failed`.
- `notClinicalDiagnosis` and `review.required` must be `true`.
- `external_coordinate` must use `researchOnly=true`.
- `model_generated_roi` is allowed by the contract but is not currently claimed as implemented or validated.

## Privacy

Payloads must not include:

- `SeriesInstanceUID`;
- `StudyInstanceUID`;
- `SOPInstanceUID`;
- `FrameOfReferenceUID`;
- `PatientID`;
- `PatientName`;
- `AccessionNumber`;
- `patientId`;
- DICOM original identifiers under alternate casing.

Stable opaque IDs, hashes or truncated non-reversible identifiers are acceptable.

## Frontend Compatibility Notes

The current frontend canonical run contract keeps measurements, landmarks and review state separate. This v1 contract is intentionally parallel and additive: it can be transported later as an optional findings block without changing successful multiplanar DTOs.

The frontend reading layer groups display rows by lumbar level. For this reason `anatomy.level` is mandatory and uses the same lumbar disc-space labels accepted by the product UI.

## Current Limitations

- No production inference route emits this payload yet.
- No Backend persistence table is required by this ticket.
- No UI rendering is required by this ticket.
- Localization is contract-ready only. Productive model-generated ROI and external-coordinate rendering require later validation.

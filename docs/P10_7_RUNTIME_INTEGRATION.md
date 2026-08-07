# P10.7 Runtime Integration

## Scope

P10.7 SPIDER is integrated as an additive, isolated disc-level degenerative
findings runtime. It does not modify or extend `pfi.degenerative-findings.v1`.
The new contract is:

```text
pfi.disc-degenerative-findings.v1
root: discDegenerativeFindings
```

The P10.6 `degenerativeFindings` contract remains unchanged.

## Artifact

The frozen research export is configured outside Git:

```text
PFI_P10_7_CHECKPOINT_PATH=/path/to/frozen_p10_7_spider_degenerative_multitask.pt
expected SHA-256: 16eccff327e6794b127fe372ecd03ea619a0f69d939b84ae1aa2e904191c6293
schemaVersion: pfi.p10-7-research-export.v1
```

The runtime validates file existence, SHA-256, schemaVersion, task order,
categorical tasks, binary tasks, trainConfig, architecture, and
`load_state_dict(strict=True)`. It never falls back to random weights.

## Endpoint

```http
POST /degenerative-findings/disc-multitask/predict
```

Request shape:

```json
{
  "caseId": "case-opaque",
  "levels": [
    {
      "level": "L4-L5",
      "sourceSeries": [
        {"role": "sagittal_t1", "inputId": "inp_t1", "available": true, "positions": [10, 11, 12]},
        {"role": "sagittal_t2", "inputId": "inp_t2", "available": true, "positions": [9, 10, 11]}
      ]
    }
  ]
}
```

The request rejects unknown fields and does not accept local paths, UIDs, patient
identifiers, or pixel arrays from Frontend.

## Current Runtime Gate

Productive `inputId` inference is intentionally blocked with:

```text
422 DISC_DEGENERATIVE_PREPROCESSING_NOT_AVAILABLE
```

Reasons:

```text
AUTOMATIC_DISC_LOCALIZATION_VALIDATED = false
REAL_RUNTIME_PREPROCESSING_PARITY_VALIDATED = false
```

The Notebook 66 preprocessing uses SPIDER masks and raw disc labels:

```text
raw_disc_label = 200 + ivd_label
ivd_index = ivd_label - 1
```

The current multiplanar runtime does not yet prove that its `discLevels`,
segmentation masks, bounds, and registered sagittal T1/T2 series reproduce the
same 2.5D crop used in training. Therefore the runtime exposes readiness and
contract validation, but refuses product inference rather than approximating.

## Level Mapping

The technical mapping used by tests is derived from the SPIDER IVD order used in
the notebooks:

```text
L1-L2 -> ivd_label 1 -> ivd_index 0 -> raw_disc_label 201
L2-L3 -> ivd_label 2 -> ivd_index 1 -> raw_disc_label 202
L3-L4 -> ivd_label 3 -> ivd_index 2 -> raw_disc_label 203
L4-L5 -> ivd_label 4 -> ivd_index 3 -> raw_disc_label 204
L5-S1 -> ivd_label 5 -> ivd_index 4 -> raw_disc_label 205
```

This mapping is not enough by itself to validate automatic runtime localization.

## Readiness

`/health`, `/warmup`, `/models`, `/readiness`, and `/models/runtime` expose:

```json
{
  "degenerativeFindingModels": {
    "discMultitask": {
      "status": "not_configured|artifact_missing|available|loaded|invalid_hash",
      "preprocessingParityValidated": false,
      "automaticDiscLocalizationValidated": false
    }
  }
}
```

No server path is exposed.

## Contract Output

When preprocessed crops are available internally and valid, the runtime emits
eight findings for each requested level, one for every task:

```text
pfirrmann_grade
modic_change
upper_endplate_change
lower_endplate_change
spondylolisthesis
disc_herniation
disc_narrowing
disc_bulging
```

Deployment status:

```text
supported_internal:
- upper_endplate_change
- lower_endplate_change
- disc_narrowing
- disc_bulging

experimental:
- pfirrmann_grade

not_product_supported:
- modic_change
- spondylolisthesis
- disc_herniation
```

All findings require professional review, are research-only, and are not a
clinical diagnosis.

# Post-E50 Sagittal Level Localization

## Objective

Build a reproducible pipeline from Sagittal T2 through the existing sagittal_spider segmentation to individual disc instances, anatomical cranio-caudal ordering, and (attempted) absolute lumbar level naming with explicit abstention when evidence is insufficient.

## Scope

Sagittal-only (Sagittal T2 primary, Sagittal T1 as secondary evidence only). No Axial T2, no cross-frame registration, no Notebook 67B, no training, no productive code changes, no change to AUTOMATIC_DISC_LOCALIZATION_VALIDATED.

## Frozen model

```json
{
  "model_id": "sagittal_spider",
  "checkpoint_path": "models\\final\\sagittal_spider_multiclass_final_best.pt",
  "checkpoint_sha256": "cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944",
  "model_card_path": "models\\final\\sagittal_spider_multiclass_final_best.pt.modelcard.md",
  "model_commit_source": "6013e160f45c9263fd4ae50e864ceb37245323e2",
  "training_notebook": "notebooks/45_gcs_spider_final_training.ipynb",
  "device": "cpu",
  "inference_dtype": "float32",
  "target_size": [
    256,
    256
  ],
  "num_classes": 4
}
```
- class_names: `{0: 'background', 1: 'vertebra_group', 2: 'canal', 3: 'disc_group'}`
- GATE A (checkpoint identity): `PASS`

## Data sources

- Real DICOM used: `True` (discovery method: `sibling_search`)
- ZIP SHA-256 matches historical expected value: `True`
- SPIDER available locally: `False` (discovery method: `not_found:no_spider_named_directory_in_project_siblings`)

## Leakage audit

Not applicable in this run: SPIDER is UNAVAILABLE, so no train/val/test split leakage audit was performed. Logic reserved for a future run with `PFI_POST_E50_SPIDER_ROOT` set.

## Sagittal inference

- GATE B (inference executable): `PASS`
- Slices processed: `12`
- Best slice (by `sagittal_slice_quality_score`): `5`
- Top-K useful slices: `[5, 4, 6, 3, 8]`

| study_opaque_id   |   slice_index |   physical_position |   vertebra_area |   disc_area |   canal_area |   disc_component_count |   foreground_fraction |   mean_confidence | border_touch   |   candidate_quality | warnings                                                  |
|:------------------|--------------:|--------------------:|----------------:|------------:|-------------:|-----------------------:|----------------------:|------------------:|:---------------|--------------------:|:----------------------------------------------------------|
| 206aa67ee4e6      |             0 |          -22.7104   |             108 |          10 |            0 |                      0 |           0.00180054  |          0.998623 | False          |            0        | no_disc_components_detected; very_low_foreground_fraction |
| 206aa67ee4e6      |             1 |          -17.2104   |            1321 |         304 |            0 |                      4 |           0.0247955   |          0.996178 | False          |            0.277088 |                                                           |
| 206aa67ee4e6      |             2 |          -11.7104   |            4614 |         671 |            0 |                      7 |           0.0806427   |          0.993811 | True           |            0.273513 | segmentation_touches_image_border                         |
| 206aa67ee4e6      |             3 |           -6.21047  |            4467 |         933 |          215 |                      7 |           0.0856781   |          0.992438 | True           |            0.363181 | segmentation_touches_image_border                         |
| 206aa67ee4e6      |             4 |           -0.710491 |            4275 |        1053 |         1513 |                      7 |           0.104385    |          0.993924 | True           |            0.596368 | segmentation_touches_image_border                         |
| 206aa67ee4e6      |             5 |            4.78949  |            5816 |        1081 |         2126 |                      7 |           0.13768     |          0.991492 | True           |            0.77     | segmentation_touches_image_border                         |
| 206aa67ee4e6      |             6 |           10.2894   |            4171 |         965 |         1556 |                      7 |           0.102112    |          0.993288 | True           |            0.578625 | segmentation_touches_image_border                         |
| 206aa67ee4e6      |             7 |           15.7894   |            4382 |         727 |            9 |                      7 |           0.0780945   |          0.993875 | True           |            0.277973 | segmentation_touches_image_border                         |
| 206aa67ee4e6      |             8 |           21.2895   |            2633 |         412 |            0 |                      5 |           0.046463    |          0.993452 | False          |            0.358461 |                                                           |
| 206aa67ee4e6      |             9 |           26.7894   |             406 |          30 |            0 |                      1 |           0.00665283  |          0.99789  | False          |            0.11439  | very_low_foreground_fraction                              |
| 206aa67ee4e6      |            10 |           32.2894   |              71 |           0 |            0 |                      0 |           0.00108337  |          0.999594 | False          |            0        | no_disc_components_detected; very_low_foreground_fraction |
| 206aa67ee4e6      |            11 |           37.7893   |              48 |           0 |            0 |                      0 |           0.000732422 |          0.999655 | False          |            0        | no_disc_components_detected; very_low_foreground_fraction |

## Disc instance extraction

- GATE C: `PASS`
- Raw components extracted: `33`

## Multi-slice consensus

- GATE D: `PASS`
| instance_id   | supporting_slices   |   supporting_slice_count |   representative_slice | centroid_patient_xyz                                          |   centroid_spread_mm |   area_mean_px |   area_std_px |   position_along_spine_mm |   ordinal_cranial_to_caudal | level   | level_status   |   instance_confidence |
|:--------------|:--------------------|-------------------------:|-----------------------:|:--------------------------------------------------------------|---------------------:|---------------:|--------------:|--------------------------:|----------------------------:|:--------|:---------------|----------------------:|
| disc_inst_00  | [3, 4, 5, 6]        |                        4 |                      5 | [-2.259055810919259, 42.6881095392793, 176.78101473441808]    |              8.52953 |          70.25 |       6.75154 |                    0      |                           1 |         | abstain        |                0.8695 |
| disc_inst_01  | [3, 4, 5, 6]        |                        4 |                      5 | [-2.1153143112739308, 37.149452398238026, 145.15102700071978] |              8.43025 |         108.75 |      10.0457  |                   32.1094 |                           2 |         | abstain        |                0.871  |
| disc_inst_02  | [3, 4, 5, 6, 8]     |                        5 |                      4 | [-5.737115985364591, 32.91765992849467, 111.69532482348286]   |             15.4342  |         102    |      30.7246  |                   65.7912 |                           3 |         | abstain        |                0.7667 |
| disc_inst_03  | [3, 4, 5, 6, 8]     |                        5 |                      4 | [-5.564682512292885, 27.29867155155144, 77.60584645425534]    |             15.4212  |         140.6  |      34.1072  |                  100.341  |                           4 |         | abstain        |                0.7669 |
| disc_inst_04  | [3, 4, 5, 6, 8]     |                        5 |                      4 | [-5.512439425166436, 18.91452385163152, 42.15915925159309]    |             15.4368  |         169    |      27.5045  |                  136.675  |                           5 |         | abstain        |                0.7666 |
| disc_inst_05  | [3, 4, 5, 6, 8]     |                        5 |                      5 | [-5.253839782797391, 14.246216147374302, 5.1642445826718415]  |             15.4042  |         173.6  |      40.8203  |                  173.94   |                           6 |         | abstain        |                0.7664 |
| disc_inst_06  | [3, 4, 5, 6, 8]     |                        5 |                      5 | [-4.491563245922584, 19.753092330370826, -31.453619026724585] |             15.43    |         160.4  |      60.6778  |                  209.197  |                           7 |         | abstain        |                0.766  |

## Spine axis

- GATE E (anatomical ordering): `PASS`
- spine_axis_vector (points caudally, patient LPS): `[0.005079097440910813, -0.1609425549314756, -0.9869507063583848]`
- Vertebral instances separable: `True`

## Absolute level naming

- GATE F: `PARTIAL`
- target_lumbar_window_status: `AMBIGUOUS`
- anchor_quality_gates (auxiliary information only, never sufficient cause): `{'instance_count_is_five': False, 'instance_count_in_plausible_range': False, 'no_caudal_clipping': True, 'no_cranial_clipping': True, 'multi_slice_support_majority': True}`
- absolute_level_status: `abstain`
- abstain_primary_reason: `ABSOLUTE_ANATOMICAL_ANCHOR_UNAVAILABLE`
- levels_assigned: `0`

## Abstention logic

Three separate tasks: (1) disc instance detection, (2) target lumbar level selection (which of the detected instances are L1-L2...L5-S1), (3) absolute level naming. A field-of-view containing more or fewer than 5 candidates is NOT by itself evidence that instance detection is wrong -- the sagittal FOV can legitimately include discs outside the lumbar target window. Absolute naming abstains because `target_lumbar_window_status = AMBIGUOUS`: no genuine anatomical anchor (SPIDER reference, a dedicated sacral/S1 class, a validated vertebral-instance anchor, or dataset metadata) is available in this run. The primary cause is `ABSOLUTE_ANATOMICAL_ANCHOR_UNAVAILABLE`, not the instance count, which is recorded only as auxiliary information.

## SPIDER held-out validation

- GATE G: `UNAVAILABLE` -- dataset not found locally in this run.

## Real DICOM exploratory result

- GATE H: `PASS`
- This is NOT ground truth. No levels were assigned; all instances are `abstain`.

### `ordered_disc_instance_profile` (cranial -> caudal, no levels assigned)

|   ordered_index | instance_id   |   supporting_slice_count |   representative_slice |   position_along_spine_mm |   distance_to_previous_mm |   distance_to_next_mm |   centroid_spread_mm |   median_area_mm2 | bbox_size   | border_touch   |   instance_confidence | quality_flags   |
|----------------:|:--------------|-------------------------:|-----------------------:|--------------------------:|--------------------------:|----------------------:|---------------------:|------------------:|:------------|:---------------|----------------------:|:----------------|
|               1 | disc_inst_00  |                        4 |                      5 |                    0      |                  nan      |               32.1094 |              8.52953 |           211.46  | [5, 22]     | False          |                0.8695 |                 |
|               2 | disc_inst_01  |                        4 |                      5 |                   32.1094 |                   32.1094 |               33.6818 |              8.43025 |           211.46  | [5, 22]     | False          |                0.871  |                 |
|               3 | disc_inst_02  |                        5 |                      4 |                   65.7912 |                   33.6818 |               34.5498 |             15.4342  |           168.893 | [5, 24]     | False          |                0.7667 |                 |
|               4 | disc_inst_03  |                        5 |                      4 |                  100.341  |                   34.5498 |               36.3338 |             15.4212  |           168.893 | [5, 24]     | False          |                0.7669 |                 |
|               5 | disc_inst_04  |                        5 |                      4 |                  136.675  |                   36.3338 |               37.2648 |             15.4368  |           168.893 | [5, 24]     | False          |                0.7666 |                 |
|               6 | disc_inst_05  |                        5 |                      5 |                  173.94   |                   37.2648 |               35.2576 |             15.4042  |           168.893 | [5, 22]     | False          |                0.7664 |                 |
|               7 | disc_inst_06  |                        5 |                      5 |                  209.197  |                   35.2576 |              nan      |             15.43    |           168.893 | [5, 22]     | False          |                0.766  |                 |

[
  {
    "schemaVersion": "pfi.post-e50.disc-localization.v0",
    "studyOpaqueId": "206aa67ee4e6",
    "instanceId": "disc_inst_00",
    "level": null,
    "levelStatus": "abstain",
    "sourceSeriesRole": "sagittal_t2",
    "representativeSlice": 5,
    "centroidPatientXYZ": [
      -2.259055810919259,
      42.6881095392793,
      176.78101473441808
    ],
    "bboxPixel": [
      4,
      9,
      109,
      131
    ],
    "bboxPhysicalMm": null,
    "supportingSlices": [
      3,
      4,
      5,
      6
    ],
    "instanceConfidence": 0.8695,
    "levelNamingConfidence": 0.4403,
    "warnings": []
  },
  {
    "schemaVersion": "pfi.post-e50.disc-localization.v0",
    "studyOpaqueId": "206aa67ee4e6",
    "instanceId": "disc_inst_01",
    "level": null,
    "levelStatus": "abstain",
    "sourceSeriesRole": "sagittal_t2",
    "representativeSlice": 5,
    "centroidPatientXYZ": [
      -2.1153143112739308,
      37.149452398238026,
      145.15102700071978
    ],
    "bboxPixel": [
      4,
      9,
      109,
      131
    ],
    "bboxPhysicalMm": null,
    "supportingSlices": [
      3,
      4,
      5,
      6
    ],
    "instanceConfidence": 0.871,
    "levelNamingConfidence": 0.4403,
    "warnings": []
  },
  {
    "schemaVersion": "pfi.post-e50.disc-localization.v0",
    "studyOpaqueId": "206aa67ee4e6",
    "instanceId": "disc_inst_02",
    "level": null,
    "levelStatus": "abstain",
    "sourceSeriesRole": "sagittal_t2",
    "representativeSlice": 4,
    "centroidPatientXYZ": [
      -5.737115985364591,
      32.91765992849467,
      111.69532482348286
    ],
    "bboxPixel": [
      4,
      9,
      109,
      133
    ],
    "bboxPhysicalMm": null,
    "supportingSlices": [
      3,
      4,
      5,
      6,
      8
    ],
    "instanceConfidence": 0.7667,
    "levelNamingConfidence": 0.4403,
    "warnings": []
  },
  {
    "schemaVersion": "pfi.post-e50.disc-localization.v0",
    "studyOpaqueId": "206aa67ee4e6",
    "instanceId": "disc_inst_03",
    "level": null,
    "levelStatus": "abstain",
    "sourceSeriesRole": "sagittal_t2",
    "representativeSlice": 4,
    "centroidPatientXYZ": [
      -5.564682512292885,
      27.29867155155144,
      77.60584645425534
    ],
    "bboxPixel": [
      4,
      9,
      109,
      133
    ],
    "bboxPhysicalMm": null,
    "supportingSlices": [
      3,
      4,
      5,
      6,
      8
    ],
    "instanceConfidence": 0.7669,
    "levelNamingConfidence": 0.4403,
    "warnings": []
  },
  {
    "schemaVersion": "pfi.post-e50.disc-localization.v0",
    "studyOpaqueId": "206aa67ee4e6",
    "instanceId": "disc_inst_04",
    "level": null,
    "levelStatus": "abstain",
    "sourceSeriesRole": "sagittal_t2",
    "representativeSlice": 4,
    "centroidPatientXYZ": [
      -5.512439425166436,
      18.91452385163152,
      42.15915925159309
    ],
    "bboxPixel": [
      4,
      9,
      109,
      133
    ],
    "bboxPhysicalMm": null,
    "supportingSlices": [
      3,
      4,
      5,
      6,
      8
    ],
    "instanceConfidence": 0.7666,
    "levelNamingConfidence": 0.4403,
    "warnings": []
  },
  {
    "schemaVersion": "pfi.post-e50.disc-localization.v0",
    "studyOpaqueId": "206aa67ee4e6",
    "instanceId": "disc_inst_05",
    "level": null,
    "levelStatus": "abstain",
    "sourceSeriesRole": "sagittal_t2",
    "representativeSlice": 5,
    "centroidPatientXYZ": [
      -5.253839782797391,
      14.246216147374302,
      5.1642445826718415
    ],
    "bboxPixel": [
      4,
      9,
      109,
      131
    ],
    "bboxPhysicalMm": null,
    "supportingSlices": [
      3,
      4,
      5,
      6,
      8
    ],
    "instanceConfidence": 0.7664,
    "levelNamingConfidence": 0.4403,
    "warnings": []
  },
  {
    "schemaVersion": "pfi.post-e50.disc-localization.v0",
    "studyOpaqueId": "206aa67ee4e6",
    "instanceId": "disc_inst_06",
    "level": null,
    "levelStatus": "abstain",
    "sourceSeriesRole": "sagittal_t2",
    "representativeSlice": 5,
    "centroidPatientXYZ": [
      -4.491563245922584,
      19.753092330370826,
      -31.453619026724585
    ],
    "bboxPixel": [
      4,
      9,
      109,
      131
    ],
    "bboxPhysicalMm": null,
    "supportingSlices": [
      3,
      4,
      5,
      6,
      8
    ],
    "instanceConfidence": 0.766,
    "levelNamingConfidence": 0.4403,
    "warnings": []
  }
]

## Quality gates

| gate                            | status      |
|:--------------------------------|:------------|
| GATE_A_checkpoint_identity      | PASS        |
| GATE_B_inference_executable     | PASS        |
| GATE_C_disc_instance_extraction | PASS        |
| GATE_D_multi_slice_consensus    | PASS        |
| GATE_E_anatomical_ordering      | PASS        |
| GATE_F_absolute_level_naming    | PARTIAL     |
| GATE_G_spider_validation        | UNAVAILABLE |
| GATE_H_real_dicom_localization  | PASS        |
| GATE_I_privacy                  | PASS        |

## Results

- Overall quality gate: `PARTIAL`
- Decision: `SAGITTAL_LEVEL_LOCALIZATION_BASELINE_ESTABLISHED`
- ready_for_axial_cluster_pairing: `False`
- ready_for_spider_heldout_validation: `True`
- ready_for_absolute_level_anchor_research: `True`
- blocking_requirement: `ABSOLUTE_LEVEL_ANCHOR_VALIDATION`

## Failure cases

- Absolute level naming ABSTAINED for all 7 instances: `ABSOLUTE_ANATOMICAL_ANCHOR_UNAVAILABLE`. This is not interpreted as an instance-detection failure.

## Limitations

- SPIDER held-out validation is UNAVAILABLE in this run: the dataset was not found locally (only unrelated lumbar-MRI datasets were present under Downloads). PFI_POST_E50_SPIDER_ROOT is supported for a future local/Colab/Drive run.
- Non-diagnostic research output; not a medical device.
- Sagittal-only in this notebook: Axial T2 and cross-frame registration were explicitly out of scope.
- The sagittal_spider segmentation model may exhibit domain shift on this external real-world DICOM study (trained on SPIDER).
- disc_group is not a native level-labelled output of the model -- level naming is a downstream heuristic, not a model capability.
- Absolute level naming requires a confident anatomical anchor; this pipeline has no dedicated sacral/S1 class, so the inferior anchor is never treated as fully validated -- only as a heuristic candidate, marked 'inferred'.
- Transitional or atypical anatomy, partial coverage, or segmentation fragmentation trigger ABSTAIN rather than a forced 5-level assignment.
- The real DICOM study provides NO ground truth: any levels produced are labeled INFERRED, never VALIDATED.
- A strong 'validated' claim requires SPIDER held-out validation with a real, leakage-audited test split -- unavailable in this run.
- No axial pairing was attempted; readiness for Notebook 67B reflects only sagittal-side stability.
- AUTOMATIC_DISC_LOCALIZATION_VALIDATED was NOT changed by this notebook.
- SPIDER dataset not found locally; held-out validation will be UNAVAILABLE (GATE G).
- Absolute level naming ABSTAINED: ABSOLUTE_ANATOMICAL_ANCHOR_UNAVAILABLE (target_lumbar_window_status=AMBIGUOUS; instance_count=7 is auxiliary information only, not the cause).

## What Notebook 67 proves

> The frozen sagittal_spider checkpoint can be run across the real study's Sagittal T2; disc_group can be transformed into instance candidates; those candidates can be consolidated across slices; the consolidated instances can be ordered reproducibly in patient-space; and reproducible centroids/ROIs are generated for each. Absolute lumbar level naming remains unresolved -- it is NOT claimed here.

## What Notebook 67 does not prove

- no clinical validation
- automatic lumbar level localization is NOT validated -- only instance detection, consensus, and ordering are demonstrated
- no resolved target lumbar window: which detected instances fall in L1-L2...L5-S1 is unknown
- no dedicated sacral/S1 anchor -- absolute level naming has no anatomical anchor to rely on for this model/study
- no proof of generalization beyond this single real study
- no axial or multiplanar correspondence (out of scope, see Notebook 66B)
- AUTOMATIC_DISC_LOCALIZATION_VALIDATED remains False and unchanged

## Ready for Notebook 67B?

`ready_for_axial_cluster_pairing = False` -- Notebook 67B needs to know which centroids correspond to L1-L2...L5-S1 to pair them against axial clusters, and no validated absolute level naming exists yet.

## Recommended next experiment

**`67A_postE50_spider_level_anchor_validation.ipynb`** -- use SPIDER held-out data to: (1) validate instance detection against real ground truth; (2) measure centroid error; (3) study how many discs appear across different fields of view; (4) learn/validate how to select the target lumbar window; (5) validate absolute level naming L1-L2...L5-S1; (6) measure abstention correctly. Only after 67A resolves the anchor question should **`67B_postE50_axial_cluster_level_pairing.ipynb`** attempt to pair axial clusters to named levels.

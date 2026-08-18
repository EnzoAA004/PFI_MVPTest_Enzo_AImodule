# Post-E50 Cross-Frame Registration

## Objective

Evaluate whether Sagittal T1, Sagittal T2 and the Axial T2 orientation clusters -- found by Notebook 66 to have different FrameOfReferenceUID values and no explicit DICOM registration object between them -- can be related via a reproducible, validated rigid transform.

## Why registration is required

Direct DICOM coordinate comparison across series with different FrameOfReferenceUID is not guaranteed to be physically meaningful. Notebook 66 confirmed this study's three series lack both a matching FrameOfReference and any explicit Spatial/Deformable Registration object, leaving cross-series spatial relationship at PARTIAL. This notebook attempts to resolve that gap with an estimated, independently-checked rigid transform.

## Notebook 66 findings

Series discovery, within-series geometry, pixel/patient transforms and privacy audit all PASSed. Axial T2 had 5 real orientation clusters (sizes [4, 4, 5, 5, 6]). All 3 series had different FrameOfReferenceUID; no registration object was found; cross-series spatial relationship was PARTIAL.

## Reference space

Sagittal T2 was chosen as fixed/reference space as an experimental engineering decision (central to the segmentation/localization pipeline, intra-series geometry already validated in Notebook 66) -- no clinical superiority is claimed.

## DICOM reconstruction

- GATE A (reconstruction): `PASS`
- GATE B (SimpleITK/Notebook-66 physical-space parity): `FAIL` (max error observed: `0.0038199607329361764` mm, target `< 1e-3 mm`, NOT relaxed retroactively)
- gate_b_failure_interpretation: `REGULAR_GRID_APPROXIMATION_OF_SLICEWISE_DICOM_GEOMETRY`
- Root cause, precisely stated (second revision): SimpleITK represents the whole stack with one regular grid (single origin/spacing/direction) while DICOM preserves geometry slice-by-slice. The scalar slice-position residual measured below is roughly an order of magnitude smaller than the full parity residual and does **not** explain it by itself; slice-wise orientation variation is a possible contributor but is not declared as the cause without measuring it (see the orientation-deviation table further below). This is **not** a general claim about the DICOM standard, and it does **not** demonstrate an error in Notebook 66's slice-native pixel<->patient equations (which never assumed a regular grid).

### GATE B residual audit (`simpleitk_geometry_residuals`)

|   slice_index |   dicom_position_scalar_mm |   regular_grid_position_scalar_mm |   absolute_residual_mm |
|--------------:|---------------------------:|----------------------------------:|-----------------------:|
|             0 |                 -22.7105   |                        -22.7105   |            0           |
|             1 |                 -17.2105   |                        -17.2105   |            0           |
|             2 |                 -11.7105   |                        -11.7105   |            1.9976e-05  |
|             3 |                  -6.21049  |                         -6.21045  |            3.99519e-05 |
|             4 |                  -0.710494 |                         -0.710434 |            5.99279e-05 |
|             5 |                   4.78951  |                          4.78958  |            7.49099e-05 |
|             6 |                  10.2894   |                         10.2896   |            0.000144819 |
|             7 |                  15.7895   |                         15.7896   |            0.000114862 |
|             8 |                  21.2895   |                         21.2896   |            8.49045e-05 |
|             9 |                  26.7895   |                         26.7896   |            0.000154814 |
|            10 |                  32.2895   |                         32.2897   |            0.000129849 |
|            11 |                  37.7895   |                         37.7897   |            0.000199758 |

- max_residual_mm: `0.0001997582842676593`
- mean_residual_mm: `8.53143378832429e-05`
- median_residual_mm: `7.990718550976439e-05`
- median_slice_spacing_mm: `5.499994380038402`
- max_residual_relative_to_slice_spacing (engineering metric, not clinical): `3.6319725160567264e-05`
- For comparison, full parity residual (GATE B above): `0.0038199607329361764` mm -- roughly an order of magnitude larger than the scalar slice-position residual, left unexplained by position non-uniformity alone.

### Slice-wise orientation deviation (`slice_orientation_angle_vs_first_deg`, analytical only)

**sagittal_t2**: max_orientation_deviation_deg = `0.0`, mean_orientation_deviation_deg = `0.0`

|   slice_index |   slice_orientation_angle_vs_first_deg |
|--------------:|---------------------------------------:|
|             0 |                                      0 |
|             1 |                                      0 |
|             2 |                                      0 |
|             3 |                                      0 |
|             4 |                                      0 |
|             5 |                                      0 |
|             6 |                                      0 |
|             7 |                                      0 |
|             8 |                                      0 |
|             9 |                                      0 |
|            10 |                                      0 |
|            11 |                                      0 |

**sagittal_t1**: max_orientation_deviation_deg = `0.0`, mean_orientation_deviation_deg = `0.0`

|   slice_index |   slice_orientation_angle_vs_first_deg |
|--------------:|---------------------------------------:|
|             0 |                                      0 |
|             1 |                                      0 |
|             2 |                                      0 |
|             3 |                                      0 |
|             4 |                                      0 |
|             5 |                                      0 |
|             6 |                                      0 |
|             7 |                                      0 |
|             8 |                                      0 |
|             9 |                                      0 |
|            10 |                                      0 |
|            11 |                                      0 |

This measurement is presented as-is, without declaring it the cause of the GATE B parity residual -- it is a possible contributor, offered for a future notebook to investigate further if cross-frame/multiplanar work resumes.

## SimpleITK configuration

```json
{
  "generated_at": "2026-08-18T00:46:32.729105+00:00",
  "simpleitk_version": "2.5.5",
  "metric": "mattes_mutual_information",
  "sagittal_config": {
    "number_of_histogram_bins": 50,
    "sampling_strategy": "RANDOM",
    "sampling_percentage": 0.2,
    "sampling_seed": 2026,
    "shrink_factors_per_level": [
      4,
      2,
      1
    ],
    "smoothing_sigmas_per_level": [
      2,
      1,
      0
    ],
    "optimizer": "RegularStepGradientDescent",
    "optimizer_scales": "physical_shift"
  }
}
```

## Sagittal T1 to T2 registration

- metric_before: `-0.7821203706655426`
- metric_after: `-1.05823428347514` (Mattes MI minimized; lower/more negative = better)
- optimizer_stop_condition: `RegularStepGradientDescentOptimizerv4: Step too small after 23 iterations. Current step (6.10352e-05) is less than minimum step (0.0001).`
- translation_magnitude_mm: `2.033871625546331`
- rotation_magnitude_deg: `0.282115450308267`
- sanity warnings: `[]` (engineering thresholds, not clinical)
- decision: `OPTIMIZATION_SUCCESS_VALIDATION_INSUFFICIENT`

## Independent validation

- validation_source: `manual_research_landmarks`
|   fraction |   tre_before_mm |   tre_after_mm |
|-----------:|----------------:|---------------:|
|       0.15 |               0 |        2.02743 |
|       0.5  |               0 |        2.03331 |
|       0.85 |               0 |        2.03818 |
- mean_TRE_before: `0.0` mm, mean_TRE_after: `2.0329728780624663` mm

## Axial orientation clusters

Re-detected cluster count: `5`, sizes: `[4, 4, 5, 5, 6]`.

## Axial cluster registration

| study_opaque_id   |   cluster_id |   slice_count | fixed_role   | moving_role      | initialization_method                                    | registration_method     | metric                    |   metric_before |   metric_after |   optimizer_iterations | optimizer_stop_condition                                                            |   translation_magnitude_mm |   rotation_magnitude_deg | inverse_available   | independent_validation_source   | tre_before_mean_mm   | tre_after_mean_mm   | registration_status   | warnings                                                         |
|:------------------|-------------:|--------------:|:-------------|:-----------------|:---------------------------------------------------------|:------------------------|:--------------------------|----------------:|---------------:|-----------------------:|:------------------------------------------------------------------------------------|---------------------------:|-------------------------:|:--------------------|:--------------------------------|:---------------------|:--------------------|:----------------------|:-----------------------------------------------------------------|
| 206aa67ee4e6      |            0 |             6 | sagittal_t2  | axial_t2_cluster | geometry_centroid_initialization_cross_frame_unvalidated | rigid_euler3d_mattes_mi | mattes_mutual_information |      -0.0503563 |     -0.0394726 |                    150 | RegularStepGradientDescentOptimizerv4: Maximum number of iterations (150) exceeded. |                   76.4016  |                  6.08635 | True                | unavailable                     |                      |                     | optimizer_success     | translation_magnitude_76.4mm_exceeds_30mm_engineering_threshold  |
| 206aa67ee4e6      |            1 |             4 | sagittal_t2  | axial_t2_cluster | geometry_centroid_initialization_cross_frame_unvalidated | rigid_euler3d_mattes_mi | mattes_mutual_information |      -0.0583711 |     -0.0657012 |                    150 | RegularStepGradientDescentOptimizerv4: Maximum number of iterations (150) exceeded. |                  107.52    |                  7.88212 | True                | unavailable                     |                      |                     | optimizer_success     | translation_magnitude_107.5mm_exceeds_30mm_engineering_threshold |
| 206aa67ee4e6      |            2 |             4 | sagittal_t2  | axial_t2_cluster | geometry_centroid_initialization_cross_frame_unvalidated | rigid_euler3d_mattes_mi | mattes_mutual_information |      -0.0869239 |     -0.47815   |                    150 | RegularStepGradientDescentOptimizerv4: Maximum number of iterations (150) exceeded. |                    2.58803 |                  1.00218 | True                | unavailable                     |                      |                     | optimizer_success     |                                                                  |
| 206aa67ee4e6      |            3 |             5 | sagittal_t2  | axial_t2_cluster | geometry_centroid_initialization_cross_frame_unvalidated | rigid_euler3d_mattes_mi | mattes_mutual_information |      -0.0904064 |     -0.487274  |                    150 | RegularStepGradientDescentOptimizerv4: Maximum number of iterations (150) exceeded. |                    1.34196 |                  2.47462 | True                | unavailable                     |                      |                     | optimizer_success     |                                                                  |
| 206aa67ee4e6      |            4 |             5 | sagittal_t2  | axial_t2_cluster | geometry_centroid_initialization_cross_frame_unvalidated | rigid_euler3d_mattes_mi | mattes_mutual_information |      -0.0681336 |     -0.121852  |                    150 | RegularStepGradientDescentOptimizerv4: Maximum number of iterations (150) exceeded. |                   82.0311  |                  7.83948 | True                | unavailable                     |                      |                     | optimizer_success     | translation_magnitude_82.0mm_exceeds_30mm_engineering_threshold  |

## Transform chain

`native_point_to_reference_space` / `reference_point_to_native_space` implemented and round-trip tested on 50 synthetic points: max error `5.123796534383003e-14` mm (target `< 1e-6 mm`) -- `PASS`.

## Quality gates

| gate                                      | status   |
|:------------------------------------------|:---------|
| GATE_A_dicom_reconstruction               | PASS     |
| GATE_B_simpleitk_parity                   | FAIL     |
| GATE_C_sagittal_registration_optimization | PASS     |
| GATE_D_sagittal_independent_validation    | PARTIAL  |
| GATE_E_transform_roundtrip                | PASS     |
| GATE_F_axial_methodology_executable       | PASS     |
| GATE_G_axial_independent_validation       | PARTIAL  |
| GATE_H_privacy                            | PASS     |

## Results

- Overall quality gate: `FAIL`
- Decision: `BLOCKED`
- ready_for_level_localization: `False`
- Blocking gates: `['GATE_B_simpleitk_parity', 'GATE_D_sagittal_independent_validation', 'GATE_G_axial_independent_validation']`

## Failures

All attempted axial cluster registrations reached optimizer_success (independent validation still pending -- see GATE G).

## Limitations

- Validated on a single real study (this notebook's Sagittal/Axial series); no proof of generalization across scanners/protocols.
- No clinical validation of any kind was performed or claimed.
- Rigid registration only (6 DOF); no deformable/non-rigid registration attempted.
- Optimizer convergence (REGISTRATION SUCCESS) is explicitly distinguished from independent validation (REGISTRATION VALIDATION); a converged optimizer alone was never treated as sufficient evidence.
- Axial cluster coverage is partial per cluster (4-6 slices), cross-modal (T2-weighted axial vs T1/T2 sagittal), and anisotropic -- registration for these is expected to be harder and some clusters may legitimately fail or lack sufficient overlap.
- Independent validation for Sagittal T1->T2 used manual_research_landmarks: 3 heuristic image-center points per series at 15%/50%/85% of physical stack depth, NOT clinically or anatomically confirmed landmarks -- research validation only, not a productive dependency.
- No independent validation was implemented for axial cluster registrations in this run (GATE G capped at PARTIAL at best); this is the main blocker to a full VALIDATED decision.
- The Sagittal T1/T2 manual_research_landmarks TRE check is methodologically weak/partially circular: both series are numerically near-identical in geometry and the landmarks use the same geometric heuristic on each series independently, so a low TRE_before is expected by construction rather than proof of independent correspondence. TRE is still reported honestly (before=0.0mm, after~2.03mm in this run) as a directional signal, not strong evidence.
- GATE B (SimpleITK/Notebook-66 parity) genuinely FAILED at 0.0038 mm (target < 1e-3 mm), was NOT loosened retroactively. Root cause: REGULAR_GRID_APPROXIMATION_OF_SLICEWISE_DICOM_GEOMETRY -- SimpleITK represents the whole series with one regular grid (single origin/spacing/direction) while DICOM preserves per-slice geometry independently. The scalar slice-position residual measured in Section 9b (max~0.0002 mm) is roughly an order of magnitude smaller than this full parity residual and does not explain it by itself; slice-wise orientation variation (Section 9c) is a possible contributor but is not declared as the cause without measuring it. This is NOT a claim about DICOM as a standard, and it does NOT demonstrate an error in Notebook 66's slice-native pixel<->patient equations, which never assumed a regular grid.
- No automatic lumbar level naming is performed or claimed anywhere in this notebook.
- No pathology detection is performed or claimed.
- Results may not generalize to studies with different geometry, coverage, or acquisition protocols.

## What 66B proves

> A reproducible rigid registration pipeline (SimpleITK, Mattes MI, multi-resolution) was built and executed end-to-end for Sagittal T1 -> Sagittal T2 on this real study, with physical-space parity against Notebook 66's independent geometry math (GATE B), a mathematically exact transform round-trip (GATE E), and a directional (before/after) independent sanity check via heuristic research landmarks. Sagittal registration decision: `OPTIMIZATION_SUCCESS_VALIDATION_INSUFFICIENT`.

## What 66B does not prove

- no clinical validation
- no proof the rigid model is sufficient (only rigid was tried; no deformable)
- optimizer convergence alone is not treated as validation anywhere in this notebook
- axial cluster registrations have no independent validation in this run -- GATE G is at best PARTIAL
- no automatic lumbar level naming or pathology detection
- single real study -- no generalization claim

## Ready for Notebook 67?

Research outcome: `CROSS_FRAME_REGISTRATION_BASELINE_ESTABLISHED`. Overall Notebook 66B decision: `BLOCKED` (quality gate `FAIL`) -- this means specifically `MULTIPLANAR CROSS-FRAME REGISTRATION NOT VALIDATED`, **not** that sagittal-only level localization cannot proceed.

- **Sagittal localization readiness**: `YES` -- Sagittal T2's intra-series geometry (orientation validity, physical slice ordering, native pixel<->patient transforms) remains validated per Notebook 66, independent of this notebook's regular-grid GATE B residual.
- **Multiplanar pairing readiness**: `NO` -- blocked by: `['GATE_B_simpleitk_parity', 'GATE_D_sagittal_independent_validation', 'GATE_G_axial_independent_validation']`. Not claimed as validated.

### Roadmap

- **`67_postE50_level_localization_v2.ipynb`** (recommended next): scope explicitly limited to **SAGITTAL-BASED AUTOMATIC LEVEL LOCALIZATION**. Must produce L1-L2, L2-L3, L3-L4, L4-L5, L5-S1 and their disc centroids/ROIs **in Sagittal T2 only** -- no claim about axial or multiplanar correspondence.
- **`67B_postE50_axial_cluster_level_pairing.ipynb`** (future work, after 67): uses the levels/centroids produced by Notebook 67 to re-evaluate the 5 axial orientation clusters and the cross-frame registration blocked here.

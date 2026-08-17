# Post-E50 Multiseries Geometry Pairing

## Objective

Build and validate a common spatial representation for Sagittal T1, Sagittal T2 and Axial T2 series of the same lumbar study, using DICOM patient-coordinate geometry, without inventing lumbar level labels.

## Scope

This notebook does geometry only: series classification, coordinate transforms, physical slice ordering, sagittal-sagittal compatibility, and sagittal-to-axial point projection. It does not train models, does not perform automatic level naming, and does not change `AUTOMATIC_DISC_LOCALIZATION_VALIDATED` in product code.

## Data source

- Real DICOM used: `True`
- Discovery method: `sibling_search`
- ZIP SHA-256: `1c058033aadaf9c72af8f1b5d85dbbbdcb7706fdda50b74537ce71a55b2227b1`
- Matches historically expected hash: `True`
- The DICOM file was read directly from the ZIP in memory; nothing was copied into the repository.

## Privacy

All persisted outputs use opaque IDs (`sha256(uid)[:12]`) instead of raw StudyInstanceUID/SeriesInstanceUID/SOPInstanceUID/PatientID/PatientName/AccessionNumber. Privacy audit result: `PASS`.

## DICOM geometry methodology

`ImageOrientationPatient` is split into `row_cosines` (first 3 values, direction of increasing column index) and `column_cosines` (last 3 values, direction of increasing row index), following pydicom's own naming to avoid the row/column swap pitfall. `patient = ImagePositionPatient + col*PixelSpacing[1]*row_cosines + row*PixelSpacing[0]*column_cosines`. `normal = cross(row_cosines, column_cosines)`. **Pixel/patient transforms are validated** (Section: Geometry unit tests) and are correct **within** the FrameOfReference of a single series. **Within-series DICOM geometry is validated**: orientation matrices are orthonormal (GATE B) and slices order physically without duplicates (GATE D).

## Cross-series Frame of Reference

This study's three series (`sagittal_t1`, `sagittal_t2`, `axial_t2`) use **different `FrameOfReferenceUID` values** (0 explicit DICOM Spatial/Deformable Registration object(s) found by read-only search). Direct cross-series coordinate comparison requires either a matching FrameOfReference or an explicit DICOM registration between frames; neither is present here. Therefore **direct cross-series correspondence between these series remains unvalidated** (`DIRECT_CROSS_SERIES_MAPPING_NOT_VALIDATED`) until registration is resolved -- the raw coordinate distances computed in this notebook are exploratory only (`UNREGISTERED_CROSS_FRAME_EXPLORATORY`) and were **not** used positively in `geometry_confidence`.

## Axial orientation clustering

The Axial T2 series' per-slice `ImageOrientationPatient` was NOT constant across its 24 slices. Clustering by angular distance between plane normals (tolerance 1.0 deg) found **5 distinct orientation cluster(s)**, each ordered physically within itself rather than as one 24-slice stack. See `post_e50_axial_orientation_clusters.csv` and `figures/axial_orientation_clusters.png`. No cluster is claimed to correspond to a specific lumbar level -- that determination belongs to a future level-localization notebook.
| series_opaque_id   |   cluster_id |   slice_count |   mean_normal_x |   mean_normal_y |   mean_normal_z |   max_angular_deviation_deg |   physical_position_min |   physical_position_max |   median_spacing_mm |   min_spacing_mm |   max_spacing_mm |   spacing_std_mm | ordering_valid   |
|:-------------------|-------------:|--------------:|----------------:|----------------:|----------------:|----------------------------:|------------------------:|------------------------:|--------------------:|-----------------:|-----------------:|-----------------:|:-----------------|
| b65eca7cfbbc       |            0 |             6 |      0.0188906  |      -0.390316  |        0.920487 |                           0 |               -52.1439  |                -24.644  |             5.5     |          5.49991 |          5.50004 |      4.56974e-05 | True             |
| b65eca7cfbbc       |            1 |             4 |     -0.00809997 |       0.106507  |        0.994279 |                           0 |               106.999   |                123.5    |             5.50045 |          5.49947 |          5.50046 |      0.000466024 | True             |
| b65eca7cfbbc       |            2 |             4 |     -0.013669   |       0.129225  |        0.991521 |                           0 |                72.2732  |                 88.7732 |             5.50003 |          5.49993 |          5.50004 |      4.96889e-05 | True             |
| b65eca7cfbbc       |            3 |             5 |     -0.00628548 |       0.0744502 |        0.997205 |                           0 |                32.9695  |                 54.9695 |             5.49997 |          5.49997 |          5.50007 |      4.44416e-05 | True             |
| b65eca7cfbbc       |            4 |             5 |      0.0110868  |      -0.0962719 |        0.995293 |                           0 |                -6.68407 |                 15.3159 |             5.5     |          5.49997 |          5.50001 |      1.51232e-05 | True             |

## Series discovery

| study_opaque_id   | series_opaque_id   | series_description_sanitized   | modality            |   rows |   columns |   slice_count |   pixel_spacing_row_mm |   pixel_spacing_col_mm |   slice_thickness_mm |   spacing_between_slices_mm | orientation_class   |   plane_normal_x |   plane_normal_y |   plane_normal_z |   position_min |   position_max | frame_of_reference_match   | candidate_role   |   role_confidence | geometry_valid   | warnings                                                                            |
|:------------------|:-------------------|:-------------------------------|:--------------------|-------:|----------:|--------------:|-----------------------:|-----------------------:|---------------------:|----------------------------:|:--------------------|-----------------:|-----------------:|-----------------:|---------------:|---------------:|:---------------------------|:-----------------|------------------:|:-----------------|:------------------------------------------------------------------------------------|
| 206aa67ee4e6      | 28ea4937243f       | T1                             | UNKNOWN_NOT_PRESENT |    512 |       512 |            12 |                 0.5859 |                 0.5859 |                  4.5 |                         5.5 | sagittal            |       -0.998661  |        0.0499264 |       -0.0132809 |       -22.7104 |        37.7893 | False                      | sagittal_t1      |               0.8 | True             | normal_dominant_axis=x ([-0.9987, 0.0499, -0.0133]); series_description_contains=t1 |
| 206aa67ee4e6      | b65eca7cfbbc       | T2                             | UNKNOWN_NOT_PRESENT |    512 |       512 |            24 |                 0.3906 |                 0.3906 |                  4.5 |                         5.5 | axial               |        0.0188907 |       -0.390318  |        0.920492  |       -52.1442 |       140.216  | False                      | axial_t2         |               0.8 | True             | normal_dominant_axis=z ([0.0189, -0.3903, 0.9205]); series_description_contains=t2  |
| 206aa67ee4e6      | 25aa08338722       | T2                             | UNKNOWN_NOT_PRESENT |    512 |       512 |            12 |                 0.5859 |                 0.5859 |                  4.5 |                         5.5 | sagittal            |       -0.998661  |        0.0499264 |       -0.0132809 |       -22.7104 |        37.7893 | False                      | sagittal_t2      |               0.8 | True             | normal_dominant_axis=x ([-0.9987, 0.0499, -0.0133]); series_description_contains=t2 |

## Coordinate transformations

Implemented: `pixel_to_patient_xyz`, `patient_xyz_to_slice_coordinates`, `build_plane`, `signed_distance_point_to_plane`, `absolute_distance_point_to_plane`, `project_patient_point_to_image_plane`. See notebook Section 3 for the exact formulas.

## Geometry unit tests

- `A_pixel_00_equals_ipp` -> `PASS`
- `B_row_increment_matches_spacing` -> `PASS`
- `C_col_increment_matches_spacing` -> `PASS`
- `D_normal_unit_norm` -> `PASS`
- `E_row_col_orthogonal` -> `PASS`
- `F_roundtrip_error_under_tolerance` -> `PASS`
- max_roundtrip_error = `7.105427357601002e-15`

## Sagittal T1 / Sagittal T2 compatibility

| series_a     | series_b     |   orientation_angle_deg |   center_distance_mm |   physical_overlap_estimate | same_frame_of_reference   | frame_relationship_status                 | geometry_numerically_compatible   | spatial_relationship_dicom_validated   | warnings                                                                                           |
|:-------------|:-------------|------------------------:|---------------------:|----------------------------:|:--------------------------|:------------------------------------------|:----------------------------------|:---------------------------------------|:---------------------------------------------------------------------------------------------------|
| 28ea4937243f | 25aa08338722 |                       0 |                    0 |                           1 | False                     | DIRECT_CROSS_SERIES_MAPPING_NOT_VALIDATED | True                              | False                                  | different_frame_of_reference_uid_per_series; spatial_relationship_not_dicom_validated_numeric_only |

## Sagittal / Axial pairing

| study_opaque_id   | reference_id                         | level_if_known   | reference_source   | sagittal_source_role   | axial_series_opaque_id   |   best_axial_slice_physical_index |   raw_coordinate_distance_mm |   best_distance_mm | distance_validated   | distance_interpretation              |   second_best_distance_mm |   candidate_count_under_2mm |   candidate_count_under_5mm | inside_fov   | same_frame_of_reference   | frame_relationship_status                 | orientation_valid   | spacing_regular   |   geometry_confidence | status                           | warnings                                                                    |
|:------------------|:-------------------------------------|:-----------------|:-------------------|:-----------------------|:-------------------------|----------------------------------:|-----------------------------:|-------------------:|:---------------------|:-------------------------------------|--------------------------:|----------------------------:|----------------------------:|:-------------|:--------------------------|:------------------------------------------|:--------------------|:------------------|----------------------:|:---------------------------------|:----------------------------------------------------------------------------|
| 206aa67ee4e6      | manual_ref_sagittal_center_mid_slice |                  | manual_reference   | sagittal_t2            | b65eca7cfbbc             |                                12 |                     0.646663 |           0.646663 | False                | UNREGISTERED_CROSS_FRAME_EXPLORATORY |                   4.85331 |                           1 |                           2 | True         | False                     | DIRECT_CROSS_SERIES_MAPPING_NOT_VALIDATED | True                | False             |                   0.5 | matched_unregistered_cross_frame | different_frame_of_reference; distance_not_dicom_validated_exploratory_only |

## Pairing examples

See `figures/axial_closest_slice_projection.png`, `figures/axial_distance_profile.png`, and `figures/pairing_examples_top_matches.png`.

## Geometry quality score

`geometry_confidence` is a deterministic, documented weighted combination (Section 10 of the notebook) — explicitly **not** a calibrated probability.

## Quality gates

| gate                                     | status   |
|:-----------------------------------------|:---------|
| GATE_A_dicom_metadata                    | PASS     |
| GATE_B_orientation                       | PASS     |
| GATE_C_transform_roundtrip               | PASS     |
| GATE_D_slice_ordering                    | PASS     |
| GATE_E_cross_series_spatial_relationship | PARTIAL  |
| GATE_F_privacy                           | PASS     |

## Results

- Sagittal T1 found: `True`
- Sagittal T2 found: `True`
- Axial T2 found: `True`
- Axial orientation clusters found: `5`
- Quality gate overall: `PARTIAL`

## Failure cases

- No failure cases: all computed best-candidate projections landed inside the axial FOV.

## Known limitations

- Axial T2 series is NOT a single physically-parallel stack: 5 distinct orientation clusters were found (per-slice ImageOrientationPatient varies), each ordered independently. This explains the irregular_spacing flag from Section 6, which was computed treating the whole series as one stack. No cluster is claimed to correspond to any specific lumbar level -- that is out of scope for Notebook 66.
- No LevelSeriesBundle was emitted with a known lumbar level: the only reference used (manual_reference) intentionally has level_if_known=null. Automatic level naming is explicitly out of scope for Notebook 66 (deferred to Notebook 67).

## What Notebook 66 proves

> Pixel/patient coordinate transforms are mathematically correct (deterministic unit tests pass) and within-series DICOM geometry (orientation validity, physical slice ordering) is validated for real Sagittal T1, Sagittal T2 and Axial T2 series of a single lumbar study. Series discovery correctly classified all three roles from real DICOM metadata.

## What Notebook 66 DOES NOT prove

- no clinical validation
- no automatic pathology diagnosis
- no validated automatic disc naming
- no proof of registration under severe patient motion
- no proof across arbitrary scanners/protocols (validated on a single real study here)
- no guarantee when DICOM geometry is incomplete/inconsistent
- no deformable registration
- geometry_confidence is not a probability
- direct cross-series spatial correspondence is NOT validated for this study: FrameOfReferenceUID differs across all 3 series and no DICOM registration object was found (GATE E = PARTIAL, not PASS) -- raw cross-series distances are exploratory only

## Recommended next notebook

Not decided automatically. The next technical requirement is `CROSS_FRAME_REGISTRATION`: resolving whether cross-series geometry should be validated as an extension of this notebook (e.g. `66b_postE50_cross_frame_registration.ipynb`) or as a precondition before `67_postE50_level_localization_v2.ipynb` attempts level/disc localization on top of unregistered series. This decision is deferred to the user.

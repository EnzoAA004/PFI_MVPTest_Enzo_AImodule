"""Robust runner for notebook 15: Al-Kafri axial license and curated subset.

Use in Colab after pulling the repo:
    %run /content/drive/MyDrive/PFI_MVP/repo/notebooks/15_E7_alkafri_axial_license_and_curated_subset_fixed.py

This script is intentionally conservative: it does not train models and does not create a
curated dataset unless enough accepted pairs pass sanity checks.
"""

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from scipy import ndimage
from skimage.transform import resize
from tqdm.auto import tqdm

pd.set_option('display.max_columns', 180)
pd.set_option('display.max_colwidth', 120)

ALKAFRI_ROOT = Path('/content/drive/MyDrive/PFI_MVP/data/AXIAL_ALKAFRI')
INVENTORY_RESULTS_ROOT = Path('/content/drive/MyDrive/PFI_MVP/results/E6_alkafri_inventory')
PAIRING_ROOT = Path('/content/drive/MyDrive/PFI_MVP/results/E6_alkafri_pairing')
PAIRING_VALIDATION_ROOT = Path('/content/drive/MyDrive/PFI_MVP/results/E6_alkafri_pairing_validation')
CURATION_ROOT = Path('/content/drive/MyDrive/PFI_MVP/results/E7_alkafri_axial_curated_subset')
CURATED_PROCESSED_ROOT = ALKAFRI_ROOT / 'processed' / 'axial_curated_v1'
FIGURES_ROOT = Path('/content/drive/MyDrive/PFI_MVP/figures')
DOCS_ROOT = Path('/content/drive/MyDrive/PFI_MVP/docs')

for path in [CURATION_ROOT, CURATED_PROCESSED_ROOT, FIGURES_ROOT, DOCS_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

MAX_CURATION_CANDIDATES = 150
MAX_STRICT_CANDIDATES = 5000
MAX_IMAGES_PER_CASE_MODALITY = 30
MAX_GT_PER_CASE_MODALITY = 30
MIN_CURATED_PAIRS = 30
RANDOM_SEED = 42

print('CURATION_ROOT:', CURATION_ROOT)


def display_if_available(df, n=5):
    try:
        display(df.head(n))
    except Exception:
        print(df.head(n).to_string())


def read_csv_any(candidates, required=False, name=''):
    for path in candidates:
        path = Path(path)
        if path.exists():
            print('OK', name or path.name, '->', path)
            return pd.read_csv(path, low_memory=False)
    print('FALTA', name or str(candidates[0]))
    if required:
        raise FileNotFoundError(f'No se encontró {name}: {candidates}')
    return pd.DataFrame()


def read_json_any(candidates, required=False, name=''):
    for path in candidates:
        path = Path(path)
        if path.exists():
            print('OK', name or path.name, '->', path)
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    print('FALTA', name or str(candidates[0]))
    if required:
        raise FileNotFoundError(f'No se encontró {name}: {candidates}')
    return {}


def normalize_case_id(value):
    if pd.isna(value):
        return None
    nums = re.findall(r'\d+', str(value))
    if not nums:
        return None
    candidates = [n for n in nums if 1 <= len(n) <= 4]
    n = candidates[-1] if candidates else nums[-1]
    try:
        return str(int(n)).zfill(4)
    except Exception:
        return None


def infer_case_id_from_path(path):
    text = str(path)
    m = re.search(r'/(\d{4})[_/]', text)
    if m:
        return m.group(1)
    m = re.search(r'[Tt][12]_(\d{4})_', text)
    if m:
        return m.group(1)
    return normalize_case_id(text)


def infer_modality_from_text(*values):
    parts = []
    for value in values:
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        parts.append(str(value))
    text = ' '.join(parts).lower()
    if re.search(r'(^|[^a-z0-9])t1([^a-z0-9]|$)|t1_', text):
        return 'T1'
    if re.search(r'(^|[^a-z0-9])t2([^a-z0-9]|$)|t2_', text):
        return 'T2'
    return None


def infer_disc_or_slice_from_path(path):
    text = str(path)
    m = re.search(r'_D(\d+)', text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+', Path(text).stem)
    if nums:
        try:
            return int(nums[-1])
        except Exception:
            return None
    return None


def infer_gt_type(path):
    text = str(path).lower()
    if '05_final_ground_truth_data' in text or 'final' in text:
        return 'final'
    if '04_intermediary_ground_truth_data' in text or 'intermediary' in text:
        return 'intermediary'
    if '03_manual_label_data' in text or 'manual' in text:
        return 'manual'
    return 'unknown'


def infer_labeller(path):
    m = re.search(r'Labeller\s*(\d+)', str(path), flags=re.IGNORECASE)
    if m:
        return str(int(m.group(1))).zfill(2)
    return None


def read_ima_shape(path):
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return (int(ds.Rows), int(ds.Columns))
    except Exception:
        return None


def read_mask_shape(path):
    try:
        with Image.open(path) as img:
            w, h = img.size
        return (h, w)
    except Exception:
        return None


def read_dicom_pixels(path):
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    return arr, ds


def normalize_image(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 <= p1:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, p1, p99)
    return (arr - p1) / (p99 - p1 + 1e-8)


def read_mask_array(path):
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:
        return np.any(arr[..., :3] > 0, axis=-1).astype(np.uint8)
    if arr.max() > 1:
        return (arr > 0).astype(np.uint8)
    return arr.astype(np.uint8)


def resize_mask_for_display(mask, target_shape):
    if tuple(mask.shape) == tuple(target_shape):
        return mask
    resized = resize(mask.astype(np.float32), target_shape, order=0, preserve_range=True, anti_aliasing=False)
    return (resized > 0.5).astype(np.uint8)


def mask_sanity(mask):
    if mask is None or mask.size == 0:
        return {'mask_empty': True, 'mask_full': False, 'foreground_ratio': 0.0, 'component_count': 0}
    fg = mask > 0
    ratio = float(fg.mean())
    _, ncomp = ndimage.label(fg)
    return {'mask_empty': bool(ratio == 0.0), 'mask_full': bool(ratio > 0.95), 'foreground_ratio': ratio, 'component_count': int(ncomp)}


# -----------------------------------------------------------------------------
# 1. License audit
# -----------------------------------------------------------------------------
license_rows = [
    {'dataset_name': 'Lumbar Spine MRI Dataset', 'source_url': 'https://data.mendeley.com/datasets/k57fr854j2/2', 'doi': '10.17632/k57fr854j2.2', 'license': 'CC BY 4.0', 'access_type': 'public_download', 'redistributable': True, 'academic_use': True, 'notes': 'MRI anonimizada de 515 pacientes; incluye cortes sagitales y axiales.', 'decision_for_pfi': 'usable'},
    {'dataset_name': 'Label Image Ground Truth Data for Lumbar Spine MRI Dataset', 'source_url': 'https://data.mendeley.com/datasets/zbf6b4pttk/2', 'doi': '10.17632/zbf6b4pttk.2', 'license': 'CC BY 4.0', 'access_type': 'public_download', 'redistributable': True, 'academic_use': True, 'notes': 'Label images/ground truth axial para SegNet: IVD, PE, TS y AAP.', 'decision_for_pfi': 'usable'},
    {'dataset_name': 'Radiologists Notes for Lumbar Spine MRI Dataset', 'source_url': 'https://data.mendeley.com/datasets/s6bgczr8s2/2', 'doi': '10.17632/s6bgczr8s2.2', 'license': 'CC BY 4.0', 'access_type': 'public_download', 'redistributable': True, 'academic_use': True, 'notes': 'Notas de radiólogos; usar como metadata/referencia.', 'decision_for_pfi': 'usable_as_metadata_reference'},
    {'dataset_name': 'MATLAB source code for Lumbar Spine MRI Dataset', 'source_url': 'https://data.mendeley.com/datasets/8cp2cp7km8/2', 'doi': '10.17632/8cp2cp7km8.2', 'license': 'GPLv3', 'access_type': 'public_download', 'redistributable': True, 'academic_use': True, 'notes': 'Usar como referencia; evitar copiar código MATLAB al producto.', 'decision_for_pfi': 'reference_only_avoid_product_code_copy'},
    {'dataset_name': 'RSNA/LumbarDISC', 'source_url': 'MIRA/Kaggle restricted sources', 'doi': '', 'license': 'non-commercial/restrictive', 'access_type': 'controlled_or_competition', 'redistributable': False, 'academic_use': 'reference_only', 'notes': 'No usar como dataset experimental principal del PFI público.', 'decision_for_pfi': 'reference_only_not_experimental_dataset'},
    {'dataset_name': 'SPIDER', 'source_url': 'project documented source', 'doi': '', 'license': 'CC BY 4.0', 'access_type': 'public_download', 'redistributable': True, 'academic_use': True, 'notes': 'Dataset principal sagital ya validado en MVP.', 'decision_for_pfi': 'main_sagittal_dataset'},
    {'dataset_name': 'SSMSpine/SymTC', 'source_url': 'https://github.com/jiasongchen/SSMSpine', 'doi': '', 'license': 'research/synthetic repository', 'access_type': 'public_repository', 'redistributable': 'review_repository_terms', 'academic_use': True, 'notes': 'Sintético/mid-sagittal; no resuelve axial.', 'decision_for_pfi': 'supplemental_benchmark_or_pretraining_only'},
]
dataset_license_audit_df = pd.DataFrame(license_rows)
license_audit_csv_path = CURATION_ROOT / 'E7_alkafri_dataset_license_audit.csv'
dataset_license_audit_df.to_csv(license_audit_csv_path, index=False)
print('\nLicense audit:')
display_if_available(dataset_license_audit_df)


# -----------------------------------------------------------------------------
# 2. Load previous outputs with fallbacks
# -----------------------------------------------------------------------------
extracted_inventory_df = read_csv_any([INVENTORY_RESULTS_ROOT / 'E6_alkafri_extracted_file_inventory.csv'], name='extracted_inventory')
series_orientation_df = read_csv_any([INVENTORY_RESULTS_ROOT / 'E6_alkafri_series_orientation_candidates.csv'], name='series_orientation')
ground_truth_inventory_df = read_csv_any([INVENTORY_RESULTS_ROOT / 'E6_alkafri_ground_truth_inventory.csv'], name='ground_truth_inventory')
axial_ima_df = read_csv_any([PAIRING_ROOT / 'E6_alkafri_axial_ima_candidates.csv'], name='axial_ima_candidates')
axial_series_summary_df = read_csv_any([PAIRING_ROOT / 'E6_alkafri_axial_series_summary.csv'], name='axial_series_summary')
ground_truth_real_df = read_csv_any([PAIRING_ROOT / 'E6_alkafri_ground_truth_real_files.csv'], name='ground_truth_real_files')
pairing_candidates_df = read_csv_any([PAIRING_ROOT / 'E6_alkafri_pairing_candidates.csv'], name='pairing_candidates')
image_specific_tokens_df = read_csv_any([PAIRING_VALIDATION_ROOT / 'E6_alkafri_image_specific_tokens.csv', PAIRING_ROOT / 'E6_alkafri_image_path_tokens.csv'], name='image_specific_tokens')
gt_specific_tokens_df = read_csv_any([PAIRING_VALIDATION_ROOT / 'E6_alkafri_gt_specific_tokens.csv', PAIRING_ROOT / 'E6_alkafri_ground_truth_path_tokens.csv'], name='gt_specific_tokens')
final_gt_tokens_df = read_csv_any([PAIRING_VALIDATION_ROOT / 'E6_alkafri_final_gt_png_tokens.csv'], name='final_gt_tokens')
source_metadata_preview_df = read_csv_any([PAIRING_VALIDATION_ROOT / 'E6_alkafri_source_metadata_tables_preview.csv', PAIRING_ROOT / 'E6_alkafri_pairing_strategy_d_source_metadata.csv'], name='source_metadata_preview')
pairing_validation_report = read_json_any([PAIRING_VALIDATION_ROOT / 'E6_alkafri_pairing_validation_report.json'], name='pairing_validation_report')
inventory_report = read_json_any([INVENTORY_RESULTS_ROOT / 'E6_alkafri_inventory_report.json'], name='inventory_report')

previous_summary = {
    'extracted_inventory_rows': int(len(extracted_inventory_df)),
    'axial_ima_candidates_raw': int(len(axial_ima_df)),
    'axial_series_summary_raw': int(len(axial_series_summary_df)),
    'ground_truth_real_files_raw': int(len(ground_truth_real_df)),
    'image_specific_tokens': int(len(image_specific_tokens_df)),
    'gt_specific_tokens': int(len(gt_specific_tokens_df)),
    'final_gt_tokens': int(len(final_gt_tokens_df)),
    'pairing_candidates_raw': int(len(pairing_candidates_df)),
    'pairing_validation_decision': pairing_validation_report.get('pairing_v1_decision'),
    'pairing_validation_recommendation': pairing_validation_report.get('recommendation_for_notebook_14'),
}
previous_outputs_summary_json_path = CURATION_ROOT / 'E7_alkafri_previous_outputs_summary.json'
previous_outputs_summary_json_path.write_text(json.dumps(previous_summary, indent=2, ensure_ascii=False), encoding='utf-8')
print('\nPrevious outputs summary:')
print(json.dumps(previous_summary, indent=2, ensure_ascii=False))


# -----------------------------------------------------------------------------
# 3. Rebuild image and ground-truth case indexes robustly
# -----------------------------------------------------------------------------
if len(axial_ima_df) == 0 and len(image_specific_tokens_df) > 0:
    print('Reconstruyendo axial_ima_df desde image_specific_tokens_df')
    axial_ima_df = image_specific_tokens_df.copy()
    if 'image_file_path' in axial_ima_df.columns and 'file_path' not in axial_ima_df.columns:
        axial_ima_df['file_path'] = axial_ima_df['image_file_path']
    if 'image_relative_path' in axial_ima_df.columns and 'relative_path' not in axial_ima_df.columns:
        axial_ima_df['relative_path'] = axial_ima_df['image_relative_path']
    if 'series_id' in axial_ima_df.columns and 'SeriesInstanceUID' not in axial_ima_df.columns:
        axial_ima_df['SeriesInstanceUID'] = axial_ima_df['series_id']
    if 'series_uid' in axial_ima_df.columns and 'SeriesInstanceUID' not in axial_ima_df.columns:
        axial_ima_df['SeriesInstanceUID'] = axial_ima_df['series_uid']
    if 'series_description' in axial_ima_df.columns and 'SeriesDescription' not in axial_ima_df.columns:
        axial_ima_df['SeriesDescription'] = axial_ima_df['series_description']
    if 'instance_number' in axial_ima_df.columns and 'InstanceNumber' not in axial_ima_df.columns:
        axial_ima_df['InstanceNumber'] = axial_ima_df['instance_number']

if len(ground_truth_real_df) == 0 and len(gt_specific_tokens_df) > 0:
    print('Reconstruyendo ground_truth_real_df desde gt_specific_tokens_df')
    ground_truth_real_df = gt_specific_tokens_df.copy()
    if 'gt_file_path' in ground_truth_real_df.columns and 'file_path' not in ground_truth_real_df.columns:
        ground_truth_real_df['file_path'] = ground_truth_real_df['gt_file_path']
    if 'gt_relative_path' in ground_truth_real_df.columns and 'relative_path' not in ground_truth_real_df.columns:
        ground_truth_real_df['relative_path'] = ground_truth_real_df['gt_relative_path']
    if 'file_path' in ground_truth_real_df.columns and 'file_name' not in ground_truth_real_df.columns:
        ground_truth_real_df['file_name'] = ground_truth_real_df['file_path'].astype(str).apply(lambda x: Path(x).name)
    if 'extension' not in ground_truth_real_df.columns and 'file_path' in ground_truth_real_df.columns:
        ground_truth_real_df['extension'] = ground_truth_real_df['file_path'].astype(str).apply(lambda x: Path(x).suffix.lower())

image_rows = []
for _, row in axial_ima_df.iterrows():
    file_path = row.get('file_path', row.get('image_file_path', None))
    rel_path = row.get('relative_path', row.get('image_relative_path', None))
    if file_path is None or pd.isna(file_path):
        continue
    series_id = row.get('SeriesInstanceUID', row.get('series_id', row.get('series_uid', None)))
    series_description = row.get('SeriesDescription', row.get('series_description', ''))
    instance_number = row.get('InstanceNumber', row.get('instance_number', row.get('slice_id_candidate', None)))
    case_id = row.get('case_id', row.get('patient_id_candidate', None))
    case_id = normalize_case_id(case_id) if case_id is not None and not pd.isna(case_id) else infer_case_id_from_path(rel_path or file_path)
    modality = row.get('modality', None)
    if modality is None or pd.isna(modality):
        modality = infer_modality_from_text(series_description, rel_path, file_path)
    text_for_localizer = ' '.join([str(series_description), str(rel_path), str(file_path)]).lower()
    image_rows.append({
        'image_file_path': str(file_path),
        'image_relative_path': str(rel_path) if rel_path is not None else '',
        'case_id': case_id,
        'modality': modality,
        'series_id': series_id,
        'series_description': series_description,
        'instance_number': instance_number,
        'is_posdisp_or_localizer': bool('posdisp' in text_for_localizer or 'localizer' in text_for_localizer or 'survey' in text_for_localizer),
    })
image_case_index_df = pd.DataFrame(image_rows)
for col in ['image_file_path', 'image_relative_path', 'case_id', 'modality', 'series_id', 'series_description', 'instance_number', 'is_posdisp_or_localizer']:
    if col not in image_case_index_df.columns:
        image_case_index_df[col] = None

gt_source_df = ground_truth_real_df.copy()
if len(final_gt_tokens_df) > 0:
    final_tmp = final_gt_tokens_df.copy()
    if 'gt_file_path' in final_tmp.columns and 'file_path' not in final_tmp.columns:
        final_tmp['file_path'] = final_tmp['gt_file_path']
    if 'gt_relative_path' in final_tmp.columns and 'relative_path' not in final_tmp.columns:
        final_tmp['relative_path'] = final_tmp['gt_relative_path']
    if 'extension' not in final_tmp.columns and 'file_path' in final_tmp.columns:
        final_tmp['extension'] = final_tmp['file_path'].astype(str).apply(lambda x: Path(x).suffix.lower())
    gt_source_df = pd.concat([gt_source_df, final_tmp], ignore_index=True)
    if 'file_path' in gt_source_df.columns:
        gt_source_df = gt_source_df.drop_duplicates(subset=['file_path'])

gt_rows = []
for _, row in gt_source_df.iterrows():
    file_path = row.get('file_path', row.get('gt_file_path', None))
    rel_path = row.get('relative_path', row.get('gt_relative_path', None))
    if file_path is None or pd.isna(file_path):
        continue
    extension = row.get('extension', Path(str(file_path)).suffix.lower())
    case_id = row.get('case_id', row.get('patient_id_candidate', None))
    case_id = normalize_case_id(case_id) if case_id is not None and not pd.isna(case_id) else infer_case_id_from_path(rel_path or file_path)
    modality = row.get('modality', None)
    if modality is None or pd.isna(modality):
        modality = infer_modality_from_text(row.get('file_name', ''), rel_path, file_path)
    disc_or_slice_id = row.get('disc_or_slice_id', row.get('slice_id_candidate', None))
    if disc_or_slice_id is None or pd.isna(disc_or_slice_id) or str(disc_or_slice_id) == '':
        disc_or_slice_id = infer_disc_or_slice_from_path(file_path)
    gt_type = row.get('gt_type', None)
    if gt_type is None or pd.isna(gt_type):
        gt_type = infer_gt_type(rel_path or file_path)
    labeller = row.get('labeller', None)
    if labeller is None or pd.isna(labeller):
        labeller = infer_labeller(rel_path or file_path)
    gt_rows.append({
        'gt_file_path': str(file_path),
        'gt_relative_path': str(rel_path) if rel_path is not None else '',
        'case_id': case_id,
        'modality': modality,
        'disc_or_slice_id': disc_or_slice_id,
        'labeller': labeller,
        'gt_type': gt_type,
        'extension': str(extension).lower(),
    })
gt_case_index_df = pd.DataFrame(gt_rows)
for col in ['gt_file_path', 'gt_relative_path', 'case_id', 'modality', 'disc_or_slice_id', 'labeller', 'gt_type', 'extension']:
    if col not in gt_case_index_df.columns:
        gt_case_index_df[col] = None

image_case_index_csv_path = CURATION_ROOT / 'E7_alkafri_axial_image_case_index.csv'
gt_case_index_csv_path = CURATION_ROOT / 'E7_alkafri_gt_case_index.csv'
image_case_index_df.to_csv(image_case_index_csv_path, index=False)
gt_case_index_df.to_csv(gt_case_index_csv_path, index=False)

summary = {
    'axial_ima_df_rows_after_fallback': int(len(axial_ima_df)),
    'image_case_index_rows': int(len(image_case_index_df)),
    'gt_source_rows': int(len(gt_source_df)),
    'gt_case_index_rows': int(len(gt_case_index_df)),
    'image_case_non_null': int(image_case_index_df['case_id'].notna().sum()) if len(image_case_index_df) else 0,
    'image_modality_non_null': int(image_case_index_df['modality'].notna().sum()) if len(image_case_index_df) else 0,
    'gt_case_non_null': int(gt_case_index_df['case_id'].notna().sum()) if len(gt_case_index_df) else 0,
    'gt_modality_non_null': int(gt_case_index_df['modality'].notna().sum()) if len(gt_case_index_df) else 0,
}
(CURATION_ROOT / 'E7_alkafri_case_index_rebuild_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print('\nCase index summary:')
print(json.dumps(summary, indent=2, ensure_ascii=False))
display_if_available(image_case_index_df)
display_if_available(gt_case_index_df)


# -----------------------------------------------------------------------------
# 4. Overlap diagnosis
# -----------------------------------------------------------------------------
image_cases = set(image_case_index_df['case_id'].dropna().astype(str))
gt_cases = set(gt_case_index_df['case_id'].dropna().astype(str))
common_cases = sorted(image_cases & gt_cases)
print('\nCasos imagen:', len(image_cases))
print('Casos GT:', len(gt_cases))
print('Casos en común:', len(common_cases))
print('Primeros casos comunes:', common_cases[:30])

image_case_modality_df = image_case_index_df.dropna(subset=['case_id', 'modality']).groupby(['case_id', 'modality']).size().reset_index(name='n_images')
gt_case_modality_df = gt_case_index_df.dropna(subset=['case_id', 'modality']).groupby(['case_id', 'modality']).size().reset_index(name='n_gt')
case_modality_overlap_df = image_case_modality_df.merge(gt_case_modality_df, on=['case_id', 'modality'], how='inner')
case_modality_overlap_csv_path = CURATION_ROOT / 'E7_alkafri_case_modality_overlap_diagnosis.csv'
case_modality_overlap_df.to_csv(case_modality_overlap_csv_path, index=False)
print('Cruces caso+modalidad:', len(case_modality_overlap_df))
display_if_available(case_modality_overlap_df, 30)


# -----------------------------------------------------------------------------
# 5. Strict candidates
# -----------------------------------------------------------------------------
image_filtered_df = image_case_index_df[
    image_case_index_df['case_id'].notna()
    & image_case_index_df['modality'].notna()
    & ~image_case_index_df['is_posdisp_or_localizer'].fillna(False)
].copy()

gt_filtered_df = gt_case_index_df[
    gt_case_index_df['case_id'].notna()
    & gt_case_index_df['modality'].notna()
    & gt_case_index_df['extension'].astype(str).str.lower().eq('.png')
    & gt_case_index_df['gt_type'].isin(['final', 'intermediary'])
].copy()

print('\nimage_filtered:', len(image_filtered_df))
print('gt_filtered:', len(gt_filtered_df))

image_reduced_df = image_filtered_df.sort_values(['case_id', 'modality', 'instance_number']).groupby(['case_id', 'modality'], group_keys=False).head(MAX_IMAGES_PER_CASE_MODALITY).copy()
gt_reduced_df = gt_filtered_df.sort_values(['case_id', 'modality', 'gt_type', 'disc_or_slice_id']).groupby(['case_id', 'modality'], group_keys=False).head(MAX_GT_PER_CASE_MODALITY).copy()

candidate_base_df = image_reduced_df.merge(gt_reduced_df, on=['case_id', 'modality'], how='inner', suffixes=('_img', '_gt'))
print('candidate_base inicial:', len(candidate_base_df))
if len(candidate_base_df) > MAX_STRICT_CANDIDATES:
    candidate_base_df = candidate_base_df.sample(n=MAX_STRICT_CANDIDATES, random_state=RANDOM_SEED).copy()
    print('candidate_base recortado a:', len(candidate_base_df))

image_shape_cache = {}
mask_shape_cache = {}
strict_rows = []
for _, row in tqdm(candidate_base_df.iterrows(), total=len(candidate_base_df), desc='Strict candidates'):
    image_path = row['image_file_path']
    gt_path = row['gt_file_path']
    if image_path not in image_shape_cache:
        image_shape_cache[image_path] = read_ima_shape(image_path)
    if gt_path not in mask_shape_cache:
        mask_shape_cache[gt_path] = read_mask_shape(gt_path)
    img_shape = image_shape_cache[image_path]
    mask_shape = mask_shape_cache[gt_path]
    shape_match = bool(img_shape is not None and mask_shape is not None and tuple(img_shape) == tuple(mask_shape))
    has_disc_or_slice = bool(pd.notna(row.get('disc_or_slice_id')) and str(row.get('disc_or_slice_id')) != '')
    confidence = 'media' if row['gt_type'] in ['final', 'intermediary'] and shape_match else 'baja'
    strict_rows.append({
        'image_file_path': image_path,
        'gt_file_path': gt_path,
        'image_relative_path': row['image_relative_path'],
        'gt_relative_path': row['gt_relative_path'],
        'case_id': row['case_id'],
        'modality': row['modality'],
        'gt_type': row['gt_type'],
        'disc_or_slice_id': row['disc_or_slice_id'],
        'instance_number': row['instance_number'],
        'image_shape': str(img_shape),
        'mask_shape': str(mask_shape),
        'evidence_case_match': True,
        'evidence_modality_match': True,
        'evidence_slice_match': has_disc_or_slice,
        'evidence_shape_match': shape_match,
        'evidence_source_metadata': bool(len(source_metadata_preview_df)),
        'confidence_previsual': confidence,
        'reason': json.dumps({'gt_type': row['gt_type'], 'shape_match': shape_match, 'has_disc_or_slice': has_disc_or_slice, 'note': 'D3/D4/D5 no se asume igual a InstanceNumber; requiere visual/manual review'}, ensure_ascii=False),
        'needs_manual_review': True,
    })
strict_candidate_pairs_df = pd.DataFrame(strict_rows)
strict_candidate_pairs_csv_path = CURATION_ROOT / 'E7_alkafri_axial_strict_candidate_pairs.csv'
strict_candidate_pairs_df.to_csv(strict_candidate_pairs_csv_path, index=False)
print('Strict candidates:', len(strict_candidate_pairs_df))
if len(strict_candidate_pairs_df):
    display_if_available(strict_candidate_pairs_df['confidence_previsual'].value_counts().rename_axis('confidence').reset_index(name='n'))
    display_if_available(strict_candidate_pairs_df)


# -----------------------------------------------------------------------------
# 6. Visual figures and automatic sanity
# -----------------------------------------------------------------------------
figure_rows = []
sanity_rows = []
if len(strict_candidate_pairs_df):
    candidates_for_visual_df = strict_candidate_pairs_df.sort_values(['confidence_previsual', 'case_id'], ascending=[False, True]).head(MAX_CURATION_CANDIDATES).copy()
else:
    candidates_for_visual_df = pd.DataFrame()

print('Candidatos para visualización:', len(candidates_for_visual_df))

for i, (_, row) in enumerate(tqdm(candidates_for_visual_df.iterrows(), total=len(candidates_for_visual_df), desc='Curated candidate figures'), start=1):
    candidate_id = f'candidate_{i:03d}'
    image_path = row['image_file_path']
    gt_path = row['gt_file_path']
    sanity = {'candidate_id': candidate_id, 'image_file_path': image_path, 'gt_file_path': gt_path, 'case_id': row.get('case_id'), 'modality': row.get('modality'), 'gt_type': row.get('gt_type'), 'disc_or_slice_id': row.get('disc_or_slice_id'), 'instance_number': row.get('instance_number'), 'confidence_previsual': row.get('confidence_previsual'), 'overlay_generated': False, 'resize_needed': None, 'image_read_error': None, 'mask_read_error': None, 'auto_sanity_status': 'error'}
    try:
        img, ds = read_dicom_pixels(image_path)
        img_norm = normalize_image(img)
        sanity['image_shape_actual'] = str(tuple(img.shape))
        sanity['image_min'] = float(np.nanmin(img))
        sanity['image_max'] = float(np.nanmax(img))
        sanity['image_mean'] = float(np.nanmean(img))
    except Exception as exc:
        sanity['image_read_error'] = repr(exc)
        sanity_rows.append(sanity)
        continue
    try:
        mask = read_mask_array(gt_path)
        sanity['mask_shape_actual'] = str(tuple(mask.shape))
    except Exception as exc:
        sanity['mask_read_error'] = repr(exc)
        sanity_rows.append(sanity)
        continue
    resize_needed = tuple(mask.shape) != tuple(img.shape)
    mask_for_display = resize_mask_for_display(mask, img.shape)
    ms = mask_sanity(mask_for_display)
    sanity.update(ms)
    sanity['resize_needed'] = bool(resize_needed)
    auto_ok = (not ms['mask_empty'] and not ms['mask_full'] and 0.0005 <= ms['foreground_ratio'] <= 0.70 and row.get('confidence_previsual') in ['media', 'alta'])
    sanity['auto_sanity_status'] = 'ok' if auto_ok else 'review'
    sanity['overlay_generated'] = True
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_norm, cmap='gray'); axes[0].set_title('Axial IMA'); axes[0].axis('off')
    axes[1].imshow(mask_for_display, cmap='gray'); axes[1].set_title('Mask/GT'); axes[1].axis('off')
    axes[2].imshow(img_norm, cmap='gray'); axes[2].imshow(np.ma.masked_where(mask_for_display <= 0, mask_for_display), alpha=0.45, cmap='autumn'); axes[2].set_title('Overlay'); axes[2].axis('off')
    fig.suptitle(f"{candidate_id} | case={row.get('case_id')} | {row.get('modality')} | gt={row.get('gt_type')} | D/slice={row.get('disc_or_slice_id')} | inst={row.get('instance_number')} | conf={row.get('confidence_previsual')}", fontsize=10)
    fig.tight_layout()
    fig_path = FIGURES_ROOT / f'E7_alkafri_curated_candidate_{i:03d}.png'
    fig.savefig(fig_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    figure_rows.append({'candidate_id': candidate_id, 'figure_path': str(fig_path), 'image_file_path': image_path, 'gt_file_path': gt_path, 'case_id': row.get('case_id'), 'modality': row.get('modality'), 'gt_type': row.get('gt_type'), 'disc_or_slice_id': row.get('disc_or_slice_id'), 'instance_number': row.get('instance_number'), 'confidence_previsual': row.get('confidence_previsual'), 'auto_sanity_status': sanity['auto_sanity_status'], 'manual_accept': '', 'manual_reject_reason': '', 'notes': ''})
    sanity_rows.append(sanity)

curation_review_sheet_df = pd.DataFrame(figure_rows)
candidate_sanity_df = pd.DataFrame(sanity_rows)
curation_review_sheet_csv_path = CURATION_ROOT / 'E7_alkafri_axial_curation_review_sheet.csv'
candidate_sanity_csv_path = CURATION_ROOT / 'E7_alkafri_axial_candidate_sanity_checks.csv'
curation_review_sheet_df.to_csv(curation_review_sheet_csv_path, index=False)
candidate_sanity_df.to_csv(candidate_sanity_csv_path, index=False)
print('Review sheet:', curation_review_sheet_csv_path)
print('Sanity checks:', candidate_sanity_csv_path)
print('Figuras generadas:', len(curation_review_sheet_df))
display_if_available(curation_review_sheet_df)
display_if_available(candidate_sanity_df)


# -----------------------------------------------------------------------------
# 7. Auto/manual curation and optional dataset creation
# -----------------------------------------------------------------------------
if len(curation_review_sheet_df) and len(candidate_sanity_df):
    auto_curated_df = curation_review_sheet_df.merge(candidate_sanity_df[['candidate_id', 'mask_empty', 'mask_full', 'foreground_ratio', 'component_count', 'resize_needed', 'overlay_generated']], on='candidate_id', how='left')
    auto_curated_df['auto_accept_candidate'] = (auto_curated_df['overlay_generated'].fillna(False) & auto_curated_df['confidence_previsual'].isin(['media', 'alta']) & ~auto_curated_df['mask_empty'].fillna(True) & ~auto_curated_df['mask_full'].fillna(True) & auto_curated_df['foreground_ratio'].between(0.0005, 0.70, inclusive='both') & auto_curated_df['gt_type'].isin(['final', 'intermediary']))
else:
    auto_curated_df = pd.DataFrame()

auto_curated_csv_path = CURATION_ROOT / 'E7_alkafri_axial_auto_curated_candidates.csv'
auto_curated_df.to_csv(auto_curated_csv_path, index=False)
manual_sheet_path = CURATION_ROOT / 'E7_alkafri_axial_curation_review_sheet_MANUAL.csv'
if manual_sheet_path.exists():
    print('Usando revisión manual:', manual_sheet_path)
    manual_df = pd.read_csv(manual_sheet_path, low_memory=False)
    accepted_df = manual_df[manual_df['manual_accept'].astype(str).str.lower().eq('yes')].copy()
    curation_source = 'manual'
else:
    print('No existe revisión manual. Usando auto_accept_candidate como curación preliminar.')
    accepted_df = auto_curated_df[auto_curated_df['auto_accept_candidate'] == True].copy() if len(auto_curated_df) else pd.DataFrame()
    curation_source = 'curated_automatic_preliminary'
print('Accepted candidates:', len(accepted_df))
display_if_available(accepted_df)


def save_curated_sample(sample_id, image_path, mask_path, row, output_root):
    img, ds = read_dicom_pixels(image_path)
    img_norm = normalize_image(img).astype(np.float32)
    mask = read_mask_array(mask_path)
    mask = resize_mask_for_display(mask, img.shape).astype(np.uint8)
    sample_dir = output_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_npy = sample_dir / 'image.npy'
    mask_npy = sample_dir / 'mask.npy'
    metadata_json = sample_dir / 'metadata.json'
    np.save(image_npy, img_norm)
    np.save(mask_npy, mask)
    metadata = {'sample_id': sample_id, 'image_file_path': image_path, 'gt_file_path': mask_path, 'case_id': row.get('case_id'), 'modality': row.get('modality'), 'gt_type': row.get('gt_type'), 'disc_or_slice_id': row.get('disc_or_slice_id'), 'instance_number': str(row.get('instance_number')), 'image_shape': tuple(img_norm.shape), 'mask_shape': tuple(mask.shape), 'curation_source': curation_source, 'confidence': row.get('confidence_previsual'), 'notes': row.get('notes', '')}
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'sample_id': sample_id, 'image_npy': str(image_npy), 'mask_npy': str(mask_npy), 'metadata_json': str(metadata_json), 'image_file_path': image_path, 'gt_file_path': mask_path, 'case_id': row.get('case_id'), 'modality': row.get('modality'), 'gt_type': row.get('gt_type'), 'disc_or_slice_id': row.get('disc_or_slice_id'), 'instance_number': row.get('instance_number'), 'image_shape': str(tuple(img_norm.shape)), 'mask_shape': str(tuple(mask.shape)), 'curation_source': curation_source, 'confidence': row.get('confidence_previsual'), 'notes': row.get('notes', '')}

curated_dataset_created = False
curated_index_rows = []
if len(accepted_df) >= MIN_CURATED_PAIRS:
    if CURATED_PROCESSED_ROOT.exists():
        shutil.rmtree(CURATED_PROCESSED_ROOT)
    CURATED_PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    for idx, (_, row) in enumerate(tqdm(accepted_df.iterrows(), total=len(accepted_df), desc='Saving curated subset'), start=1):
        sample_id = f'axial_curated_{idx:04d}'
        try:
            curated_index_rows.append(save_curated_sample(sample_id, row['image_file_path'], row['gt_file_path'], row, CURATED_PROCESSED_ROOT))
        except Exception as exc:
            curated_index_rows.append({'sample_id': sample_id, 'image_file_path': row.get('image_file_path'), 'gt_file_path': row.get('gt_file_path'), 'error': repr(exc)})
    curated_dataset_created = len([r for r in curated_index_rows if 'error' not in r]) >= MIN_CURATED_PAIRS
else:
    print(f'No hay al menos {MIN_CURATED_PAIRS} pares aceptados ({len(accepted_df)} disponibles). No se crea dataset entrenable.')
curated_index_df = pd.DataFrame(curated_index_rows)
curated_index_csv_path = CURATION_ROOT / 'E7_alkafri_axial_curated_v1_index.csv'
curated_index_df.to_csv(curated_index_csv_path, index=False)


# -----------------------------------------------------------------------------
# 8. Final report
# -----------------------------------------------------------------------------
license_ok = bool(dataset_license_audit_df[dataset_license_audit_df['dataset_name'].isin(['Lumbar Spine MRI Dataset', 'Label Image Ground Truth Data for Lumbar Spine MRI Dataset'])]['license'].astype(str).str.contains('CC BY 4.0', case=False, na=False).all())
feasibility_assessment = {
    'license_ok': license_ok,
    'redistributable_under_cc_by': license_ok,
    'has_native_axial': int(len(image_case_index_df)) > 0,
    'has_ground_truth': int(len(gt_case_index_df)) > 0,
    'previous_pairing_was_ambiguous': True,
    'case_modality_overlap_count': int(len(case_modality_overlap_df)),
    'strict_candidates_count': int(len(strict_candidate_pairs_df)),
    'auto_curated_count': int(auto_curated_df['auto_accept_candidate'].sum()) if len(auto_curated_df) and 'auto_accept_candidate' in auto_curated_df.columns else 0,
    'accepted_count': int(len(accepted_df)),
    'curated_dataset_created': bool(curated_dataset_created),
    'curated_dataset_path': str(CURATED_PROCESSED_ROOT) if curated_dataset_created else None,
    'usable_for_axial_segmentation_baseline': bool(curated_dataset_created),
    'recommended_next_step': 'notebook_16_axial_baseline_segmentation' if curated_dataset_created else 'manual_review_or_continue_dataset_search',
}
feasibility_path = CURATION_ROOT / 'E7_alkafri_axial_curated_feasibility_assessment.json'
feasibility_path.write_text(json.dumps(feasibility_assessment, indent=2, ensure_ascii=False), encoding='utf-8')

report = {
    'dataset': 'Al-Kafri/Sudirman Lumbar Spine MRI Dataset',
    'license_audit': str(license_audit_csv_path),
    'previous_outputs_summary': str(previous_outputs_summary_json_path),
    'image_case_index': str(CURATION_ROOT / 'E7_alkafri_axial_image_case_index.csv'),
    'gt_case_index': str(CURATION_ROOT / 'E7_alkafri_gt_case_index.csv'),
    'case_modality_overlap': str(case_modality_overlap_csv_path),
    'strict_candidates': str(CURATION_ROOT / 'E7_alkafri_axial_strict_candidate_pairs.csv'),
    'curation_review_sheet': str(curation_review_sheet_csv_path),
    'candidate_sanity': str(candidate_sanity_csv_path),
    'auto_curated_candidates': str(auto_curated_csv_path),
    'curated_index': str(curated_index_csv_path),
    'feasibility_assessment': feasibility_assessment,
    'figures_generated': curation_review_sheet_df['figure_path'].tolist() if len(curation_review_sheet_df) else [],
    'limitations': ['D3/D4/D5 is treated as level/disc evidence, not direct DICOM InstanceNumber.', 'Automatically accepted pairs remain preliminary unless manually reviewed.', 'Research-only exploratory curation; no clinical use.'],
}
report_path = CURATION_ROOT / 'E7_alkafri_axial_license_and_curated_subset_report.json'
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

conclusion_lines = [
    '# E7 - Al-Kafri axial: licencia y subset curado', '',
    '## Objetivo', '',
    'Auditar la viabilidad legal y técnica de Al-Kafri/Sudirman como fuente axial para el PFI público, y construir un subset axial curado solo si hay evidencia suficiente.', '',
    '## Licencia y fuentes', '',
    'Al-Kafri/Sudirman se documenta como fuente pública de Mendeley Data con licencia CC BY 4.0. RSNA/LumbarDISC se mantiene como referencia metodológica, no como dataset experimental principal, por restricciones de redistribución/publicación.', '',
    '## Estado técnico', '',
    f'- Imágenes indexadas: {len(image_case_index_df)}.',
    f'- Ground truth indexado: {len(gt_case_index_df)}.',
    f'- Cruces caso+modalidad: {feasibility_assessment["case_modality_overlap_count"]}.',
    f'- Candidatos estrictos: {feasibility_assessment["strict_candidates_count"]}.',
    f'- Candidatos aceptados automáticamente/preliminarmente: {feasibility_assessment["auto_curated_count"]}.',
    f'- Dataset curado creado: {feasibility_assessment["curated_dataset_created"]}.', '',
    '## Decisión técnica', '',
    f'`usable_for_axial_segmentation_baseline = {feasibility_assessment["usable_for_axial_segmentation_baseline"]}`', '',
    '## Próximo paso', '',
    f'`{feasibility_assessment["recommended_next_step"]}`', '',
    '## Limitaciones', '',
    '- La curación automática no reemplaza revisión manual/profesional.',
    '- D3/D4/D5 se interpreta como evidencia de nivel/disco, no como número de corte DICOM directo.',
    '- El notebook no entrena modelos ni valida uso clínico.',
]
conclusion_path = DOCS_ROOT / 'E7_alkafri_axial_license_and_curated_subset_conclusion.md'
conclusion_path.write_text('\n'.join(conclusion_lines), encoding='utf-8')

print('\nFeasibility:')
print(json.dumps(feasibility_assessment, indent=2, ensure_ascii=False))
print('Report:', report_path)
print('Conclusion:', conclusion_path)

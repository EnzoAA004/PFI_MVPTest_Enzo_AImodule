"""Train-only RSNA LumbarDISC preflight helpers for P10.6-AI.

The module intentionally avoids reading the official test set. All inventory,
label, sequence and coordinate checks operate on ``train.csv``,
``train_label_coordinates.csv``, ``train_series_descriptions.csv`` and
``train_images/`` only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import time


CONDITIONS = [
    "spinal_canal_stenosis",
    "neural_foraminal_narrowing_left",
    "neural_foraminal_narrowing_right",
    "subarticular_stenosis_left",
    "subarticular_stenosis_right",
]
LEVELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
SEVERITY = ["normal_mild", "moderate", "severe"]
EXCLUDED_FROM_RSNA_TRAINING = {
    "disc_bulge",
    "disc_protrusion",
    "disc_extrusion",
    "disc_sequestration",
    "tumor",
    "infection",
    "fracture",
}
SCOPE_CONFIG_PATH = Path("configs/p10_6_rsna_scope_v1.yaml")
SENSITIVE_DICOM_FIELDS = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "InstitutionName",
    "AccessionNumber",
}


@dataclass(frozen=True)
class RsnaPreflightConfig:
    pfi_root: Path
    rsna_root: Path
    train_images: Path
    output_root: Path
    model_root: Path
    synthetic: bool
    dicom_hash_opt_in: bool
    seed: int = 2026


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} debe ser booleano; obtenido {value!r}")


def build_config() -> RsnaPreflightConfig:
    pfi_root = Path(os.getenv("PFI_ROOT", "/content/drive/MyDrive/PFI_MVP"))
    rsna_root = Path(os.getenv("PFI_RSNA_ROOT", str(pfi_root / "data" / "RSNA_LUMBAR_DISC")))
    train_images = Path(os.getenv("PFI_RSNA_TRAIN_IMAGES", str(rsna_root / "train_images")))
    output_root = Path(os.getenv("PFI_P10_6_OUTPUT_ROOT", str(pfi_root / "results" / "P10_6_rsna_findings")))
    model_root = Path(os.getenv("PFI_P10_6_MODEL_ROOT", str(pfi_root / "models" / "p10_6_rsna_findings")))
    return RsnaPreflightConfig(
        pfi_root=pfi_root,
        rsna_root=rsna_root,
        train_images=train_images,
        output_root=output_root,
        model_root=model_root,
        synthetic=env_bool("PFI_RSNA_PREFLIGHT_SYNTHETIC", False),
        dicom_hash_opt_in=env_bool("PFI_RSNA_HASH_DICOM_OPT_IN", False),
    )


def runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    for module_name, label in [
        ("torch", "pytorch"),
        ("pydicom", "pydicom"),
        ("SimpleITK", "simpleitk"),
        ("monai", "monai"),
        ("timm", "timm"),
    ]:
        try:
            module = __import__(module_name)
            versions[label] = getattr(module, "__version__", "installed")
        except Exception as exc:
            versions[label] = f"missing:{exc.__class__.__name__}"
    try:
        import torch

        versions["cudaAvailable"] = bool(torch.cuda.is_available())
        versions["cudaVersion"] = torch.version.cuda
        versions["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        versions["cudaAvailable"] = False
        versions["cudaVersion"] = None
        versions["gpu"] = None
    return versions


def set_reproducible_seed(seed: int = 2026) -> None:
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_condition(value: Any) -> str | None:
    text = normalize_text(value)
    mapping = {
        "spinal_canal_stenosis": "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing": "neural_foraminal_narrowing_left",
        "neural_foraminal_narrowing_left": "neural_foraminal_narrowing_left",
        "right_neural_foraminal_narrowing": "neural_foraminal_narrowing_right",
        "neural_foraminal_narrowing_right": "neural_foraminal_narrowing_right",
        "left_subarticular_stenosis": "subarticular_stenosis_left",
        "subarticular_stenosis_left": "subarticular_stenosis_left",
        "right_subarticular_stenosis": "subarticular_stenosis_right",
        "subarticular_stenosis_right": "subarticular_stenosis_right",
    }
    return mapping.get(text)


def normalize_level(value: Any) -> str | None:
    text = str(value).strip().upper().replace("_", "-").replace("/", "-")
    text = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"L([1-5])-L([2-5]|S1)", text)
    if match:
        candidate = f"L{match.group(1)}-L{match.group(2)}"
        return candidate if candidate in LEVELS else None
    return text if text in LEVELS else None


def normalize_severity(value: Any) -> str | None:
    text = normalize_text(value)
    mapping = {
        "normal_mild": "normal_mild",
        "normal_or_mild": "normal_mild",
        "normal": "normal_mild",
        "mild": "normal_mild",
        "moderate": "moderate",
        "severe": "severe",
    }
    return mapping.get(text)


def parse_label_column(column: str) -> tuple[str | None, str | None]:
    tokens = column.lower().split("_")
    if len(tokens) < 3:
        return None, None
    level = normalize_level("-".join(tokens[-2:]).upper())
    condition = normalize_condition("_".join(tokens[:-2]))
    return condition, level


def normalizer_mapping() -> dict[str, Any]:
    return {
        "conditions": {
            "Spinal Canal Stenosis": "spinal_canal_stenosis",
            "Left Neural Foraminal Narrowing": "neural_foraminal_narrowing_left",
            "Right Neural Foraminal Narrowing": "neural_foraminal_narrowing_right",
            "Left Subarticular Stenosis": "subarticular_stenosis_left",
            "Right Subarticular Stenosis": "subarticular_stenosis_right",
        },
        "levels": {level: level for level in LEVELS},
        "severity": {
            "Normal/Mild": "normal_mild",
            "Moderate": "moderate",
            "Severe": "severe",
        },
    }


def normalize_series_description(value: Any) -> dict[str, Any]:
    raw = str(value or "")
    tokens = [token for token in re.split(r"[^a-z0-9]+", raw.lower()) if token]
    token_set = set(tokens)
    plane = "unknown"
    if {"sag", "sagittal"} & token_set:
        plane = "sagittal"
    elif {"ax", "axial"} & token_set:
        plane = "axial"

    sequence = "unknown"
    if "stir" in token_set:
        sequence = "T2_STIR"
    elif "t2" in token_set or "t2w" in token_set:
        sequence = "T2"
    elif "t1" in token_set or "t1w" in token_set:
        sequence = "T1"

    category = "unknown"
    if plane == "sagittal" and sequence == "T1":
        category = "sagittal_t1"
    elif plane == "sagittal" and sequence in {"T2", "T2_STIR"}:
        category = "sagittal_t2_stir"
    elif plane == "axial" and sequence == "T2":
        category = "axial_t2"

    return {
        "seriesDescriptionRaw": raw,
        "seriesDescriptionNormalized": normalize_text(raw),
        "tokens": tokens,
        "plane": plane,
        "sequence": sequence,
        "sequenceCategory": category,
    }


def required_dataset_paths(rsna_root: Path, train_images: Path) -> dict[str, Path]:
    return {
        "train.csv": rsna_root / "train.csv",
        "train_label_coordinates.csv": rsna_root / "train_label_coordinates.csv",
        "train_series_descriptions.csv": rsna_root / "train_series_descriptions.csv",
        "train_images": train_images,
    }


def validate_dataset_structure(config: RsnaPreflightConfig) -> dict[str, Any]:
    paths = required_dataset_paths(config.rsna_root, config.train_images)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Estructura RSNA incompleta; faltan: "
            + ", ".join(missing)
            + f". Configurar PFI_RSNA_ROOT={config.rsna_root}"
        )
    official_test_present = (config.rsna_root / "test_images").exists() or (config.rsna_root / "test_series_descriptions.csv").exists()
    return {
        "rsnaRoot": sanitize_path(config.rsna_root),
        "trainImagesRoot": sanitize_path(config.train_images),
        "requiredPresent": sorted(paths.keys()),
        "officialTestPresent": bool(official_test_present),
        "officialTestAccessed": False,
    }


def sanitize_path(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if "/content/drive/MyDrive/" in text:
        return text[text.index("/content/drive/MyDrive/") :]
    return f"<path:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}>"


def opaque_id(value: Any, prefix: str) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def is_sensitive_log_text(text: str) -> bool:
    lowered = text.lower()
    blocked = ["patientname", "patientid", "birthdate", "institutionname", "accessionnumber", "seriesinstanceuid", "sopinstanceuid"]
    return any(item in lowered for item in blocked)


def read_csvs(config: RsnaPreflightConfig):
    import pandas as pd

    train = pd.read_csv(config.rsna_root / "train.csv", dtype={"study_id": "string"})
    coordinates = pd.read_csv(
        config.rsna_root / "train_label_coordinates.csv",
        dtype={"study_id": "string", "series_id": "string", "instance_number": "Int64"},
    )
    series = pd.read_csv(
        config.rsna_root / "train_series_descriptions.csv",
        dtype={"study_id": "string", "series_id": "string", "series_description": "string"},
    )
    return train, coordinates, series


def long_label_frame(train_df):
    import pandas as pd

    rows = []
    for column in train_df.columns:
        if column == "study_id":
            continue
        condition, level = parse_label_column(str(column))
        for _, row in train_df[["study_id", column]].iterrows():
            severity = normalize_severity(row[column])
            rows.append(
                {
                    "study_id": str(row.study_id),
                    "condition": condition,
                    "level": level,
                    "severity": severity,
                    "rsna_label_column": str(column),
                    "rsna_label_value": row[column],
                }
            )
    return pd.DataFrame(rows)


def build_series_inventory(train_images: Path, series_df):
    import pandas as pd

    records: list[dict[str, Any]] = []
    for row in series_df.itertuples(index=False):
        study_id = str(row.study_id)
        series_id = str(row.series_id)
        series_dir = train_images / study_id / series_id
        files = sorted(path for path in series_dir.glob("*") if path.is_file()) if series_dir.exists() else []
        desc = normalize_series_description(row.series_description)
        records.append(
            {
                "study_id_hash": opaque_id(study_id, "study"),
                "series_id_hash": opaque_id(series_id, "series"),
                "study_id": study_id,
                "series_id": series_id,
                "series_description": str(row.series_description),
                "series_description_normalized": desc["seriesDescriptionNormalized"],
                "plane": desc["plane"],
                "sequence": desc["sequence"],
                "sequence_category": desc["sequenceCategory"],
                "dicom_count": int(len(files)),
                "series_dir_present": bool(series_dir.exists()),
            }
        )
    return pd.DataFrame(records)


def sequence_availability(series_inventory):
    categories = ["sagittal_t1", "sagittal_t2_stir", "axial_t2"]
    rows = []
    grouped = series_inventory.groupby("study_id")
    for study_id, frame in grouped:
        cats = set(str(value) for value in frame["sequence_category"])
        item = {
            "study_id_hash": opaque_id(study_id, "study"),
            "study_id": str(study_id),
            "series_count": int(len(frame)),
        }
        for category in categories:
            item[f"has_{category}"] = bool(category in cats)
        item["missing_sequence_categories"] = ",".join(category for category in categories if category not in cats)
        rows.append(item)
    return series_inventory.__class__(rows)


def label_distribution(long_labels):
    import pandas as pd

    tables = []
    for group_cols in [
        ["condition"],
        ["level"],
        ["severity"],
        ["condition", "level"],
        ["condition", "severity"],
        ["level", "severity"],
        ["condition", "level", "severity"],
    ]:
        counts = long_labels.groupby(group_cols, dropna=False).size().reset_index(name="count")
        total = int(counts["count"].sum()) if len(counts) else 0
        counts["percent"] = counts["count"].astype(float) / max(total, 1) * 100.0
        counts.insert(0, "table", " x ".join(group_cols))
        tables.append(counts)
    laterality_rows = []
    for condition in long_labels["condition"].dropna():
        condition = str(condition)
        if condition.endswith("_left"):
            laterality = "left"
        elif condition.endswith("_right"):
            laterality = "right"
        else:
            laterality = "central"
        laterality_rows.append({"laterality": laterality})
    if laterality_rows:
        laterality = pd.DataFrame(laterality_rows).groupby("laterality").size().reset_index(name="count")
        laterality["percent"] = laterality["count"].astype(float) / max(int(laterality["count"].sum()), 1) * 100.0
        laterality.insert(0, "table", "laterality")
        tables.append(laterality)
    by_study = long_labels.groupby("study_id").size().reset_index(name="count")
    by_study["study_id_hash"] = by_study["study_id"].map(lambda value: opaque_id(value, "study"))
    by_study = by_study.drop(columns=["study_id"])
    by_study.insert(0, "table", "study")
    tables.append(by_study)
    return pd.concat(tables, ignore_index=True, sort=False)


def missing_data_summary(train_df, coordinates_df, series_df, long_labels):
    rows = []
    for name, frame in [("train", train_df), ("coordinates", coordinates_df), ("series", series_df), ("labels_long", long_labels)]:
        for column in frame.columns:
            null_count = int(frame[column].isna().sum())
            if null_count:
                rows.append({"table": name, "column": str(column), "nullCount": null_count})
    unknown_conditions = int(long_labels["condition"].isna().sum()) if "condition" in long_labels else 0
    unknown_levels = int(long_labels["level"].isna().sum()) if "level" in long_labels else 0
    unknown_severity = int(long_labels["severity"].isna().sum()) if "severity" in long_labels else 0
    rows.extend(
        [
            {"table": "labels_long", "column": "condition", "nullCount": unknown_conditions, "issue": "unrecognized"},
            {"table": "labels_long", "column": "level", "nullCount": unknown_levels, "issue": "unrecognized"},
            {"table": "labels_long", "column": "severity", "nullCount": unknown_severity, "issue": "unrecognized"},
        ]
    )
    return train_df.__class__(rows)


def read_minimal_dicom_metadata(path: Path) -> dict[str, Any]:
    import pydicom

    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    sensitive_present = sorted(name for name in SENSITIVE_DICOM_FIELDS if getattr(dataset, name, None) not in (None, ""))
    return {
        "rows": int(getattr(dataset, "Rows", 0) or 0),
        "columns": int(getattr(dataset, "Columns", 0) or 0),
        "instanceNumber": int(getattr(dataset, "InstanceNumber", -1) or -1),
        "imagePositionPatient": [float(x) for x in getattr(dataset, "ImagePositionPatient", [])] if getattr(dataset, "ImagePositionPatient", None) is not None else None,
        "sliceLocation": float(getattr(dataset, "SliceLocation")) if getattr(dataset, "SliceLocation", None) is not None else None,
        "sensitiveHeaderFieldsPresent": sensitive_present,
    }


def coordinate_validation(coordinates_df, series_inventory, train_images: Path):
    rows = []
    series_lookup = {
        (str(row.study_id), str(row.series_id)): row
        for row in series_inventory.itertuples(index=False)
    }
    for row in coordinates_df.itertuples(index=False):
        study_id = str(row.study_id)
        series_id = str(row.series_id)
        instance_number = int(row.instance_number) if row.instance_number == row.instance_number else -1
        series_dir = train_images / study_id / series_id
        dicom_path = series_dir / f"{instance_number}.dcm"
        metadata: dict[str, Any] = {}
        dicom_exists = dicom_path.exists()
        if dicom_exists:
            try:
                metadata = read_minimal_dicom_metadata(dicom_path)
            except Exception as exc:
                metadata = {"readError": exc.__class__.__name__}
        x = float(row.x) if row.x == row.x else float("nan")
        y = float(row.y) if row.y == row.y else float("nan")
        rows.append(
            {
                "study_id_hash": opaque_id(study_id, "study"),
                "series_id_hash": opaque_id(series_id, "series"),
                "condition": normalize_condition(row.condition),
                "level": normalize_level(row.level),
                "instance_number": instance_number,
                "series_belongs_to_study": (study_id, series_id) in series_lookup,
                "dicom_exists": bool(dicom_exists),
                "coordinate_inside_image": bool(
                    metadata.get("columns", 0) > x >= 0
                    and metadata.get("rows", 0) > y >= 0
                ) if metadata else None,
                "sensitive_header_fields_present": ",".join(metadata.get("sensitiveHeaderFieldsPresent", [])),
                "read_error": metadata.get("readError"),
            }
        )
    return coordinates_df.__class__(rows)


def order_source_for_series(series_dir: Path) -> str:
    files = sorted(path for path in series_dir.glob("*.dcm") if path.is_file()) if series_dir.exists() else []
    if not files:
        return "missing_or_empty"
    image_positions = 0
    instance_numbers = 0
    slice_locations = 0
    for path in files[: min(len(files), 8)]:
        try:
            metadata = read_minimal_dicom_metadata(path)
        except Exception:
            continue
        if metadata.get("imagePositionPatient") is not None:
            image_positions += 1
        if metadata.get("instanceNumber", -1) >= 0:
            instance_numbers += 1
        if metadata.get("sliceLocation") is not None:
            slice_locations += 1
    if image_positions:
        return "ImagePositionPatient"
    if instance_numbers:
        return "InstanceNumber"
    if slice_locations:
        return "SliceLocation"
    return "explicit_filename_fallback_documented"


def enrich_order_source(series_inventory, train_images: Path):
    frame = series_inventory.copy()
    frame["order_source"] = [
        order_source_for_series(train_images / str(row.study_id) / str(row.series_id))
        for row in frame.itertuples(index=False)
    ]
    return frame


def build_summary(
    *,
    config: RsnaPreflightConfig,
    structure: dict[str, Any],
    train_df,
    coordinates_df,
    series_inventory,
    sequence_df,
    label_df,
    coordinate_df,
) -> dict[str, Any]:
    label_counts = label_df[label_df["table"] == "condition"].sort_values("count", ascending=False)
    min_count = int(label_counts["count"].min()) if len(label_counts) else 0
    max_count = int(label_counts["count"].max()) if len(label_counts) else 0
    return {
        "ticket": "P10.6",
        "dataset": "RSNA_LumbarDISC",
        "purpose": "candidate_degenerative_findings",
        "commercialUse": False,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "autonomousDiagnosis": False,
        "officialTestPresent": bool(structure["officialTestPresent"]),
        "officialTestAccessed": False,
        "rsnaRoot": structure["rsnaRoot"],
        "trainImagesRoot": structure["trainImagesRoot"],
        "outputRoot": sanitize_path(config.output_root),
        "modelRoot": sanitize_path(config.model_root),
        "nStudies": int(train_df["study_id"].nunique()),
        "nSeries": int(series_inventory["series_id"].nunique()),
        "nDicom": int(series_inventory["dicom_count"].sum()),
        "seriesPerStudy": {
            "min": int(series_inventory.groupby("study_id").size().min()) if len(series_inventory) else 0,
            "max": int(series_inventory.groupby("study_id").size().max()) if len(series_inventory) else 0,
        },
        "sequenceAvailability": {
            "studiesWithSagittalT1": int(sequence_df["has_sagittal_t1"].sum()) if len(sequence_df) else 0,
            "studiesWithSagittalT2Stir": int(sequence_df["has_sagittal_t2_stir"].sum()) if len(sequence_df) else 0,
            "studiesWithAxialT2": int(sequence_df["has_axial_t2"].sum()) if len(sequence_df) else 0,
        },
        "labelClassImbalance": {
            "minConditionCount": min_count,
            "maxConditionCount": max_count,
            "maxMinRatio": round(max_count / min_count, 4) if min_count else None,
        },
        "coordinateIssues": {
            "rows": int(len(coordinate_df)),
            "missingDicom": int((coordinate_df["dicom_exists"] == False).sum()) if len(coordinate_df) else 0,  # noqa: E712
            "outsideImage": int((coordinate_df["coordinate_inside_image"] == False).sum()) if len(coordinate_df) else 0,  # noqa: E712
            "sensitiveHeaderRows": int((coordinate_df["sensitive_header_fields_present"].astype(str) != "").sum()) if len(coordinate_df) else 0,
        },
        "csvSha256": {
            "train.csv": sha256_file(config.rsna_root / "train.csv"),
            "train_label_coordinates.csv": sha256_file(config.rsna_root / "train_label_coordinates.csv"),
            "train_series_descriptions.csv": sha256_file(config.rsna_root / "train_series_descriptions.csv"),
        },
        "normalizationMapping": normalizer_mapping(),
        "futureMetrics": [
            "recall_sensitivity",
            "precision",
            "specificity",
            "macro_f1",
            "balanced_accuracy",
            "roc_auc_one_vs_rest",
            "confusion_matrix",
            "weighted_log_loss",
            "calibration",
            "severe_false_negatives",
            "metrics_by_level",
        ],
        "limitations": [
            "No final training was started.",
            "Official test, if present, was not accessed.",
            "RSNA labels are not used for disc protrusion or extrusion.",
        ],
    }


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# P10.6 RSNA preflight report",
        "",
        "Este reporte resume train-only para clasificacion asistida de hallazgo candidato; requiere revision profesional.",
        "",
        f"- Dataset: {summary['dataset']}",
        f"- Uso comercial: {str(summary['commercialUse']).lower()}",
        f"- humanReviewRequired: {str(summary['humanReviewRequired']).lower()}",
        f"- notClinicalDiagnosis: {str(summary['notClinicalDiagnosis']).lower()}",
        f"- officialTestPresent: {str(summary['officialTestPresent']).lower()}",
        f"- officialTestAccessed: {str(summary['officialTestAccessed']).lower()}",
        f"- Estudios train: {summary['nStudies']}",
        f"- Series train: {summary['nSeries']}",
        f"- DICOM train: {summary['nDicom']}",
        "",
        "## Disponibilidad de secuencias",
        "",
        f"- Estudios con Sagittal T1: {summary['sequenceAvailability']['studiesWithSagittalT1']}",
        f"- Estudios con Sagittal T2/STIR: {summary['sequenceAvailability']['studiesWithSagittalT2Stir']}",
        f"- Estudios con Axial T2: {summary['sequenceAvailability']['studiesWithAxialT2']}",
        "",
        "## Hashes CSV",
        "",
    ]
    for name, digest in summary["csvSha256"].items():
        lines.append(f"- {name}: `{digest}`")
    lines.extend(
        [
            "",
            "## Limitaciones",
            "",
            "- No se entreno ningun modelo final.",
            "- No se accedio al conjunto oficial de test.",
            "- Protrusion y extrusion no se entrenan con RSNA.",
            "- Accuracy sola no sera una metrica suficiente.",
            "",
            "## Proximo paso",
            "",
            "Notebook 54 debera definir split interno por `study_id` y fracciones finales solo despues de revisar esta distribucion.",
            "",
        ]
    )
    text = "\n".join(lines)
    if is_sensitive_log_text(text):
        raise RuntimeError("El reporte sanitizado contiene texto sensible.")
    return text


def write_outputs(
    *,
    config: RsnaPreflightConfig,
    summary: dict[str, Any],
    series_inventory,
    label_df,
    sequence_df,
    coordinate_df,
    missing_df,
) -> dict[str, str]:
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    inventory_path = output_root / "dataset_inventory.json"
    report_json_path = output_root / "rsna_preflight_report.json"
    report_md_path = output_root / "rsna_preflight_report.md"
    atomic_write_json(inventory_path, summary)
    atomic_write_json(report_json_path, summary)
    atomic_write_text(report_md_path, report_markdown(summary))
    csv_outputs = {
        "series_inventory.csv": series_inventory.drop(columns=["study_id", "series_id"], errors="ignore"),
        "label_distribution.csv": label_df,
        "sequence_availability.csv": sequence_df.drop(columns=["study_id"], errors="ignore"),
        "coordinate_validation.csv": coordinate_df,
        "missing_data_summary.csv": missing_df,
    }
    written = {
        "dataset_inventory.json": str(inventory_path),
        "rsna_preflight_report.json": str(report_json_path),
        "rsna_preflight_report.md": str(report_md_path),
    }
    for name, frame in csv_outputs.items():
        destination = output_root / name
        frame.to_csv(destination, index=False)
        written[name] = str(destination)
    return written


def create_synthetic_rsna_dataset(root: Path) -> None:
    import numpy as np
    import pandas as pd
    import pydicom
    from pydicom.dataset import FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    root.mkdir(parents=True, exist_ok=True)
    train_images = root / "train_images"
    rows = []
    series_rows = []
    coord_rows = []
    condition_columns = {}
    for condition in CONDITIONS:
        for level in LEVELS:
            column = rsna_label_column(condition, level)
            condition_columns[column] = "Normal/Mild"
    for study_index, study_id in enumerate(["1001", "1002"]):
        row = {"study_id": study_id, **condition_columns}
        if study_index == 0:
            row["spinal_canal_stenosis_l4_l5"] = "Moderate"
            row["left_neural_foraminal_narrowing_l5_s1"] = "Severe"
        rows.append(row)
        for series_id, description in [
            (f"{study_id}01", "Sagittal T1"),
            (f"{study_id}02", "Sagittal T2/STIR"),
            (f"{study_id}03", "Axial T2"),
        ]:
            series_rows.append({"study_id": study_id, "series_id": series_id, "series_description": description})
            series_dir = train_images / study_id / series_id
            series_dir.mkdir(parents=True, exist_ok=True)
            for instance in [1, 2]:
                path = series_dir / f"{instance}.dcm"
                file_meta = pydicom.Dataset()
                file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
                ds.is_little_endian = True
                ds.is_implicit_VR = False
                ds.SOPClassUID = generate_uid()
                ds.SOPInstanceUID = generate_uid()
                ds.Modality = "MR"
                ds.Rows = 16
                ds.Columns = 16
                ds.InstanceNumber = instance
                ds.ImagePositionPatient = [0.0, 0.0, float(instance)]
                ds.SliceLocation = float(instance)
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.SamplesPerPixel = 1
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0
                ds.PixelData = np.zeros((16, 16), dtype=np.uint16).tobytes()
                ds.save_as(path, write_like_original=False)
        coord_rows.extend(
            [
                {"study_id": study_id, "series_id": f"{study_id}02", "instance_number": 1, "condition": "Spinal Canal Stenosis", "level": "L4/L5", "x": 8.0, "y": 8.0},
                {"study_id": study_id, "series_id": f"{study_id}01", "instance_number": 1, "condition": "Left Neural Foraminal Narrowing", "level": "L5/S1", "x": 7.0, "y": 8.0},
                {"study_id": study_id, "series_id": f"{study_id}01", "instance_number": 1, "condition": "Right Neural Foraminal Narrowing", "level": "L5/S1", "x": 9.0, "y": 8.0},
                {"study_id": study_id, "series_id": f"{study_id}03", "instance_number": 1, "condition": "Left Subarticular Stenosis", "level": "L4/L5", "x": 7.0, "y": 9.0},
                {"study_id": study_id, "series_id": f"{study_id}03", "instance_number": 1, "condition": "Right Subarticular Stenosis", "level": "L4/L5", "x": 9.0, "y": 9.0},
            ]
        )
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(coord_rows).to_csv(root / "train_label_coordinates.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "train_series_descriptions.csv", index=False)
    (root / "test_images").mkdir(exist_ok=True)


def rsna_label_column(condition: str, level: str) -> str:
    reverse = {
        "spinal_canal_stenosis": "spinal_canal_stenosis",
        "neural_foraminal_narrowing_left": "left_neural_foraminal_narrowing",
        "neural_foraminal_narrowing_right": "right_neural_foraminal_narrowing",
        "subarticular_stenosis_left": "left_subarticular_stenosis",
        "subarticular_stenosis_right": "right_subarticular_stenosis",
    }
    return f"{reverse[condition]}_{level.lower().replace('-', '_')}"


def run_preflight(config: RsnaPreflightConfig | None = None) -> dict[str, Any]:
    start = time.time()
    config = config or build_config()
    if config.synthetic:
        synthetic_root = Path(tempfile.mkdtemp(prefix="pfi-rsna-synthetic-"))
        create_synthetic_rsna_dataset(synthetic_root)
        config = RsnaPreflightConfig(
            pfi_root=config.pfi_root,
            rsna_root=synthetic_root,
            train_images=synthetic_root / "train_images",
            output_root=config.output_root,
            model_root=config.model_root,
            synthetic=True,
            dicom_hash_opt_in=config.dicom_hash_opt_in,
            seed=config.seed,
        )
    set_reproducible_seed(config.seed)
    structure = validate_dataset_structure(config)
    train_df, coordinates_df, series_df = read_csvs(config)
    long_labels = long_label_frame(train_df)
    series_inventory = build_series_inventory(config.train_images, series_df)
    series_inventory = enrich_order_source(series_inventory, config.train_images)
    sequence_df = sequence_availability(series_inventory)
    labels_df = label_distribution(long_labels)
    coordinate_df = coordinate_validation(coordinates_df, series_inventory, config.train_images)
    missing_df = missing_data_summary(train_df, coordinates_df, series_df, long_labels)
    summary = build_summary(
        config=config,
        structure=structure,
        train_df=train_df,
        coordinates_df=coordinates_df,
        series_inventory=series_inventory,
        sequence_df=sequence_df,
        label_df=labels_df,
        coordinate_df=coordinate_df,
    )
    summary["runtime"] = runtime_versions()
    summary["syntheticMode"] = bool(config.synthetic)
    summary["durationSeconds"] = round(time.time() - start, 3)
    summary["outputs"] = write_outputs(
        config=config,
        summary=summary,
        series_inventory=series_inventory,
        label_df=labels_df,
        sequence_df=sequence_df,
        coordinate_df=coordinate_df,
        missing_df=missing_df,
    )
    return summary


def verify_report_has_no_full_ids(report_text: str) -> bool:
    return not bool(re.search(r"\b\d{5,}\b", report_text))


def assert_no_official_test_access(paths: Iterable[str]) -> None:
    forbidden = ["test_images", "test_series_descriptions", "sample_submission"]
    offending = [path for path in paths if any(item in path.replace("\\", "/") for item in forbidden)]
    if offending:
        raise RuntimeError(f"Acceso prohibido al conjunto test o sample_submission: {offending[:3]}")

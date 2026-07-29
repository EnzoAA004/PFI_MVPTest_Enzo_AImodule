"""Whole-study DICOM zip ingestion.

A real lumbar MRI study is a single archive holding several *series* (sagittal T1,
sagittal T2, axial T2, ...). This module extracts such a zip, classifies each series
by plane (from ``ImageOrientationPatient``) and weighting (from ``SeriesDescription``
/ echo time), and selects one series per plane for inference:

- sagittal: prefer T2, fall back to T1, then to the largest sagittal series.
- axial: require T2 (the axial model was trained on axial T2); otherwise flagged.

The selected series are registered as ordinary per-plane inputs so the existing
``/multiplanar/run`` contract (sagittalInputId + axialInputId) is unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .input_registry import (
    InputRegistryError,
    max_upload_bytes,
    register_series_files,
    safe_extract_zip,
    upload_root,
    write_limited_upload,
)

_T2_TOKENS = ("t2", "t2w", "t2_", "t2-", "t2 ")
_T1_TOKENS = ("t1", "t1w", "t1_", "t1-", "t1 ")


def _plane_from_orientation(iop: Any) -> str | None:
    if iop is None or len(iop) < 6:
        return None
    try:
        row = [float(iop[0]), float(iop[1]), float(iop[2])]
        col = [float(iop[3]), float(iop[4]), float(iop[5])]
    except (TypeError, ValueError):
        return None
    normal = [
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    ]
    axis = max(range(3), key=lambda i: abs(normal[i]))
    return {0: "sagittal", 1: "coronal", 2: "axial"}[axis]


def _weighting(description: str, echo_time: float | None) -> str:
    text = f" {description.lower()} "
    if any(token in text for token in _T2_TOKENS):
        return "t2"
    if any(token in text for token in _T1_TOKENS):
        return "t1"
    if echo_time is not None:
        if echo_time >= 80:
            return "t2"
        if echo_time <= 30:
            return "t1"
    return "unknown"


def classify_study_series(root: Path) -> list[dict[str, Any]]:
    """Return one descriptor per DICOM series found under ``root``."""
    try:
        import pydicom
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - dependency guard
        raise InputRegistryError("SimpleITK/pydicom requeridos para leer el estudio", status_code=500) from exc

    reader = sitk.ImageSeriesReader()
    series: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dirpath, _dirs, filenames in os.walk(root):
        # No extension gate: GDCM identifies DICOM by content, so this also handles
        # Siemens .ima and extension-less PACS exports, not only *.dcm files.
        if not filenames:
            continue
        for series_id in reader.GetGDCMSeriesIDs(dirpath):
            if series_id in seen:
                continue
            file_names = list(reader.GetGDCMSeriesFileNames(dirpath, series_id))
            if len(file_names) < 1:
                continue
            seen.add(series_id)
            try:
                ds = pydicom.dcmread(file_names[0], stop_before_pixels=True, force=True)
            except Exception:
                continue
            description = str(getattr(ds, "SeriesDescription", "") or "")
            echo_time = getattr(ds, "EchoTime", None)
            echo_time = float(echo_time) if echo_time is not None else None
            series.append({
                "seriesInstanceUid": series_id,
                "description": description,
                "plane": _plane_from_orientation(getattr(ds, "ImageOrientationPatient", None)),
                "weighting": _weighting(description, echo_time),
                "sliceCount": len(file_names),
                "files": file_names,
            })
    return series


def _select_series(series: list[dict[str, Any]], plane: str, prefer: str) -> dict[str, Any] | None:
    candidates = [s for s in series if s["plane"] == plane]
    if not candidates:
        return None
    order = [prefer, "t1" if prefer == "t2" else "t2", "unknown"]

    def rank(item: dict[str, Any]) -> tuple[int, int]:
        weight_rank = order.index(item["weighting"]) if item["weighting"] in order else len(order)
        return (weight_rank, -item["sliceCount"])

    return sorted(candidates, key=rank)[0]


def _summarize(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "seriesInstanceUid": item["seriesInstanceUid"],
        "description": item["description"],
        "plane": item["plane"],
        "weighting": item["weighting"],
        "sliceCount": item["sliceCount"],
    }


def register_study_zip(*, case_id: str, stream: Any) -> dict[str, Any]:
    study_id = uuid4().hex
    study_root = upload_root() / "studies" / study_id
    raw_dir = study_root / "raw"
    zip_path = study_root / "study.zip"
    study_root.mkdir(parents=True, exist_ok=True)

    try:
        write_limited_upload(stream, zip_path, max_upload_bytes())
        try:
            _total, count = safe_extract_zip(zip_path, raw_dir)
        finally:
            if zip_path.exists():
                zip_path.unlink()
        if count == 0:
            raise InputRegistryError("zip vacio", status_code=400)

        series = classify_study_series(raw_dir)
        if not series:
            raise InputRegistryError("no se encontraron series DICOM en el zip", status_code=400)

        warnings: list[str] = []
        sagittal = _select_series(series, "sagittal", prefer="t2")
        axial = _select_series(series, "axial", prefer="t2")

        result: dict[str, Any] = {
            "caseId": case_id,
            "studyId": study_id,
            "seriesFound": [_summarize(s) for s in series],
            "warnings": warnings,
        }
        if sagittal is None:
            warnings.append("no se encontro serie sagital en el estudio")
        else:
            if sagittal["weighting"] != "t2":
                warnings.append(f"sagital sin T2 disponible; se usa {sagittal['weighting']}")
            meta = register_series_files(case_id=case_id, plane="sagittal", file_paths=sagittal["files"])
            result["sagittal"] = {**meta, **_summarize(sagittal)}
        if axial is None:
            warnings.append("no se encontro serie axial en el estudio")
        else:
            if axial["weighting"] != "t2":
                warnings.append(f"axial sin T2 real; el modelo axial espera T2 (encontrado {axial['weighting']})")
            meta = register_series_files(case_id=case_id, plane="axial", file_paths=axial["files"])
            result["axial"] = {**meta, **_summarize(axial)}

        if "sagittal" not in result and "axial" not in result:
            raise InputRegistryError("no se pudo identificar serie sagital ni axial en el estudio", status_code=422)
        return result
    finally:
        import shutil

        shutil.rmtree(study_root, ignore_errors=True)

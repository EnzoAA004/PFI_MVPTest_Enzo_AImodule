from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from pfi_ai_service.api import app
from pfi_ai_service.asset_registry import clear_asset_registry, registered_assets_for_run
from pfi_ai_service.input_registry import (
    InputRegistryError,
    resolve_input_id,
    safe_extract_zip,
)
from pfi_ai_service.real_inference_runtime import (
    LoadedInput,
    build_volume_slice_catalog,
    load_input,
    read_dicom_series,
    save_slice_catalog_assets,
)
from pfi_ai_service.study_ingestion import classify_study_series, register_study_zip

SAGITTAL = [0, 1, 0, 0, 0, -1]
AXIAL = [1, 0, 0, 0, 1, 0]
CORONAL = [1, 0, 0, 0, 0, -1]


def normal_from_orientation(orientation: list[int]) -> list[float]:
    row = [float(value) for value in orientation[:3]]
    col = [float(value) for value in orientation[3:6]]
    return [
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    ]


def dicom_slice_bytes(
    *,
    series_uid: str,
    description: str,
    orientation: list[int],
    index: int,
    echo_time: float | None = None,
    rows: int = 16,
    cols: int = 16,
) -> bytes:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.SeriesDescription = description
    if echo_time is not None:
        ds.EchoTime = echo_time
    ds.InstanceNumber = index + 1
    ds.ImageOrientationPatient = [float(value) for value in orientation]
    normal = normal_from_orientation(orientation)
    ds.ImagePositionPatient = [float(value * index * 3.0) for value in normal]
    ds.PixelSpacing = [0.7, 0.8]
    ds.SliceThickness = 3.0
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    pixels = (np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols) + index).astype(np.uint16)
    ds.PixelData = pixels.tobytes()
    buffer = io.BytesIO()
    import pydicom

    pydicom.dcmwrite(buffer, ds, write_like_original=False)
    return buffer.getvalue()


def series_entries(
    *,
    folder: str,
    description: str,
    orientation: list[int],
    count: int,
    extension: str = ".dcm",
    echo_time: float | None = None,
) -> list[tuple[str, bytes]]:
    series_uid = generate_uid()
    entries: list[tuple[str, bytes]] = []
    for index in range(count):
        suffix = extension
        name = f"{index:03d}{suffix}" if suffix else f"{index:03d}"
        entries.append((
            f"{folder}/{name}",
            dicom_slice_bytes(
                series_uid=series_uid,
                description=description,
                orientation=orientation,
                index=index,
                echo_time=echo_time,
            ),
        ))
    return entries


def zip_bytes(entries: Iterable[tuple[str, bytes]]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    buffer.seek(0)
    return buffer


def write_series(root: Path, *, description: str, orientation: list[int], count: int, extension: str = ".dcm") -> Path:
    folder = root / description
    folder.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    for index in range(count):
        suffix = extension
        name = f"{index:03d}{suffix}" if suffix else f"{index:03d}"
        (folder / name).write_bytes(dicom_slice_bytes(
            series_uid=series_uid,
            description=description,
            orientation=orientation,
            index=index,
        ))
    return folder


def assert_no_internal_paths(value: object, tmp_path: Path) -> None:
    text = json.dumps(value, sort_keys=True)
    forbidden = [
        str(tmp_path).replace("\\", "\\\\"),
        str(tmp_path),
        "uploads/",
        "uploads\\",
        "raw/",
        "raw\\",
        "study.zip",
        "PixelData",
        "ImagePositionPatient",
        "ImageOrientationPatient",
    ]
    assert not any(item and item in text for item in forbidden), text[:500]


def test_valid_zip_registers_sagittal_axial_inputs_and_no_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    payload = zip_bytes([
        *series_entries(folder="nested/sag-t1", description="T1_TSE_SAG_320", orientation=SAGITTAL, count=2, echo_time=12),
        *series_entries(folder="nested/sag-t2", description="T2_TSE_SAG_320", orientation=SAGITTAL, count=3, echo_time=100),
        *series_entries(folder="nested/ax-t2", description="POSDISP_[4]_T2_TSE_TRA_384", orientation=AXIAL, count=4, echo_time=110),
    ])

    body = register_study_zip(case_id="CASE-P105F-ZIP", stream=payload)

    assert body["caseId"] == "CASE-P105F-ZIP"
    assert body["studyId"]
    assert len(body["seriesFound"]) == 3
    assert body["sagittal"]["inputId"].startswith("inp_")
    assert body["sagittal"]["weighting"] == "t2"
    assert body["sagittal"]["sliceCount"] == 3
    assert body["axial"]["inputId"].startswith("inp_")
    assert body["axial"]["weighting"] == "t2"
    assert body["axial"]["sliceCount"] == 4
    assert resolve_input_id(body["sagittal"]["inputId"], case_id="CASE-P105F-ZIP", plane="sagittal").format == "dicom_series"
    assert resolve_input_id(body["axial"]["inputId"], case_id="CASE-P105F-ZIP", plane="axial").format == "dicom_series"
    assert_no_internal_paths(body, tmp_path)


def test_api_study_zip_endpoint_returns_trace_and_opaque_inputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    payload = zip_bytes([
        *series_entries(folder="sag", description="T2_TSE_SAG_320", orientation=SAGITTAL, count=2),
        *series_entries(folder="ax", description="T2_TSE_TRA_384", orientation=AXIAL, count=2),
    ])

    response = TestClient(app).post(
        "/inputs/study",
        headers={"X-Trace-Id": "trace-p10-5-f-study"},
        data={"caseId": "CASE-P105F-ENDPOINT"},
        files={"file": ("study.zip", payload, "application/zip")},
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Trace-Id"] == "trace-p10-5-f-study"
    body = response.json()
    assert body["sagittal"]["inputId"].startswith("inp_")
    assert body["axial"]["inputId"].startswith("inp_")
    assert_no_internal_paths(body, tmp_path)


def test_ima_and_extensionless_dicoms_are_detected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    payload = zip_bytes([
        *series_entries(folder="sag-ima", description="T2_TSE_SAG_320", orientation=SAGITTAL, count=2, extension=".ima"),
        *series_entries(folder="ax-no-ext", description="T2_TSE_TRA_384", orientation=AXIAL, count=2, extension=""),
    ])

    body = register_study_zip(case_id="CASE-P105F-IMA", stream=payload)

    assert {item["plane"] for item in body["seriesFound"]} == {"sagittal", "axial"}
    assert body["sagittal"]["sliceCount"] == 2
    assert body["axial"]["sliceCount"] == 2
    assert all(path.suffix == ".dcm" for path in resolve_input_id(body["sagittal"]["inputId"], case_id="CASE-P105F-IMA", plane="sagittal").path.iterdir())


def test_series_selection_prefers_sagittal_t2_over_larger_t1(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    body = register_study_zip(case_id="CASE-P105F-SAG-T2", stream=zip_bytes([
        *series_entries(folder="sag-t1-large", description="T1_TSE_SAG_320", orientation=SAGITTAL, count=5, echo_time=10),
        *series_entries(folder="sag-t2-small", description="T2_TSE_SAG_320", orientation=SAGITTAL, count=2, echo_time=100),
    ]))

    assert body["sagittal"]["description"] == "T2_TSE_SAG_320"
    assert body["sagittal"]["weighting"] == "t2"


def test_series_selection_falls_back_to_sagittal_t1(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    body = register_study_zip(case_id="CASE-P105F-SAG-T1", stream=zip_bytes(
        series_entries(folder="sag-t1", description="T1_TSE_SAG_320", orientation=SAGITTAL, count=3, echo_time=12)
    ))

    assert body["sagittal"]["weighting"] == "t1"
    assert any("sagital sin T2" in warning for warning in body["warnings"])


def test_series_selection_prefers_axial_t2_and_reports_missing_axial(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    with_axial = register_study_zip(case_id="CASE-P105F-AX-T2", stream=zip_bytes([
        *series_entries(folder="sag", description="T2_TSE_SAG_320", orientation=SAGITTAL, count=2),
        *series_entries(folder="ax-t1", description="T1_TSE_TRA_384", orientation=AXIAL, count=4, echo_time=12),
        *series_entries(folder="ax-t2", description="T2_TSE_TRA_384", orientation=AXIAL, count=2, echo_time=100),
    ]))
    missing_axial = register_study_zip(case_id="CASE-P105F-NO-AX", stream=zip_bytes(
        series_entries(folder="sag", description="T2_TSE_SAG_320", orientation=SAGITTAL, count=2)
    ))

    assert with_axial["axial"]["description"] == "T2_TSE_TRA_384"
    assert "axial" not in missing_axial
    assert "no se encontro serie axial en el estudio" in missing_axial["warnings"]


def test_multiple_candidates_choose_largest_same_weighting(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    body = register_study_zip(case_id="CASE-P105F-MULTI", stream=zip_bytes([
        *series_entries(folder="sag-small", description="T2_TSE_SAG_SMALL", orientation=SAGITTAL, count=2),
        *series_entries(folder="sag-large", description="T2_TSE_SAG_LARGE", orientation=SAGITTAL, count=5),
    ]))

    assert body["sagittal"]["description"] == "T2_TSE_SAG_LARGE"
    assert body["sagittal"]["sliceCount"] == 5


def test_zip_empty_corrupt_zipslip_absolute_and_limits_cleanup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))

    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w"):
        pass
    empty.seek(0)
    with pytest.raises(InputRegistryError, match="zip vacio"):
        register_study_zip(case_id="CASE-EMPTY", stream=empty)

    with pytest.raises(InputRegistryError, match="archivo zip invalido"):
        register_study_zip(case_id="CASE-CORRUPT", stream=io.BytesIO(b"not-a-zip"))

    for bad_name in ("../evil.dcm", "/absolute/evil.dcm", "C:/absolute/evil.dcm"):
        with pytest.raises(InputRegistryError, match="ruta invalida"):
            register_study_zip(case_id="CASE-BAD-PATH", stream=zip_bytes([(bad_name, b"bad")]))
        assert not (tmp_path / "evil.dcm").exists()

    monkeypatch.setenv("PFI_MAX_SERIES_FILES", "1")
    with pytest.raises(InputRegistryError, match="demasiados archivos"):
        register_study_zip(case_id="CASE-MANY", stream=zip_bytes([("a.dcm", b"a"), ("b.dcm", b"b")]))
    monkeypatch.delenv("PFI_MAX_SERIES_FILES")

    monkeypatch.setenv("PFI_MAX_UPLOAD_BYTES", "8")
    with pytest.raises(InputRegistryError, match="excede el limite"):
        register_study_zip(case_id="CASE-COMPRESSED-LIMIT", stream=zip_bytes([("a.dcm", b"0123456789")]))
    monkeypatch.delenv("PFI_MAX_UPLOAD_BYTES")

    zip_path = tmp_path / "oversized.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("big.dcm", b"0123456789")
    monkeypatch.setenv("PFI_MAX_SERIES_UNCOMPRESSED_BYTES", "4")
    extract_dir = tmp_path / "partial"
    with pytest.raises(InputRegistryError, match="descomprimido excede"):
        safe_extract_zip(zip_path, extract_dir)
    assert not extract_dir.exists()


def test_input_id_case_plane_guards_after_study_registration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    body = register_study_zip(case_id="CASE-P105F-GUARDS", stream=zip_bytes(
        series_entries(folder="sag", description="T2_TSE_SAG_320", orientation=SAGITTAL, count=2)
    ))
    input_id = body["sagittal"]["inputId"]

    assert resolve_input_id(input_id, case_id="CASE-P105F-GUARDS", plane="sagittal").input_id == input_id
    with pytest.raises(InputRegistryError, match="caseId"):
        resolve_input_id(input_id, case_id="CASE-OTHER", plane="sagittal")
    with pytest.raises(InputRegistryError, match="plano"):
        resolve_input_id(input_id, case_id="CASE-P105F-GUARDS", plane="axial")


def test_read_dicom_series_as_volume_and_catalog_from_all_slices(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    clear_asset_registry()
    series_dir = write_series(tmp_path / "dicom", description="T2_TSE_TRA_384", orientation=AXIAL, count=4, extension=".ima")

    array, spacing, metadata = read_dicom_series(series_dir)
    loaded = LoadedInput(
        array=array,
        path=series_dir,
        suffix=".dcm",
        spacing_xyz=spacing,
        metadata={"inputShapeNative": list(array.shape), "inputShapeCanonical": list(array.shape), **metadata},
    )
    overlay = tmp_path / "overlay.png"
    overlay.write_bytes(b"overlay")
    outputs = save_slice_catalog_assets("run-p105f", "axial", loaded, 0, 2, (16, 16), str(overlay))
    catalog = build_volume_slice_catalog(
        run_id="run-p105f",
        plane="axial",
        slice_count=int(array.shape[0]),
        selected_slice=2,
        measurement_values=[{"id": "m-1"}],
        landmarks=[{"id": "lm-1"}],
    )

    assert array.ndim == 3
    assert array.shape[0] == 4
    assert metadata["sliceCount"] == 4
    assert len([key for key in outputs if key.startswith("slicePreview")]) == 4
    assert len(catalog) == 4
    assert [item["index"] for item in catalog] == [0, 1, 2, 3]
    assert [item["displayIndex"] for item in catalog] == [1, 2, 3, 4]
    assert [item["hasResults"] for item in catalog] == [False, False, True, False]
    assert catalog[2]["overlayAsset"]["assetName"] == "slice-002-overlay.png"
    assert catalog[0]["overlayAsset"] is None
    assert "slice-002-overlay.png" in registered_assets_for_run("run-p105f", "axial")


def test_mha_sagittal_and_axial_regression(monkeypatch, tmp_path: Path) -> None:
    sitk = pytest.importorskip("SimpleITK")
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    array = np.arange(3 * 8 * 9, dtype=np.float32).reshape(3, 8, 9)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.8, 0.7, 3.0))
    mha_path = tmp_path / "input.mha"
    sitk.WriteImage(image, str(mha_path))

    sagittal = load_input(str(mha_path), "sagittal")
    axial = load_input(str(mha_path), "axial")

    assert sagittal.array.shape == (3, 8, 9)
    assert axial.array.shape == (3, 8, 9)
    assert sagittal.suffix == ".mha"
    assert axial.spacing_xyz == (0.8, 0.7, 3.0)


def test_security_errors_are_sanitized_for_study_zip_endpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    response = TestClient(app).post(
        "/inputs/study",
        headers={"X-Trace-Id": "trace-p105f-security"},
        data={"caseId": "CASE-P105F-SEC"},
        files={"file": ("bad.zip", b"not-a-zip", "application/zip")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["traceId"] == "trace-p105f-security"
    assert body["message"] == "archivo zip invalido o corrupto"
    assert_no_internal_paths(body, tmp_path)


def test_real_study_zip_opt_in(monkeypatch, tmp_path: Path) -> None:
    if os.getenv("RUN_PFI_REAL_STUDY_E2E") != "1":
        pytest.skip("real study ZIP opt-in disabled")
    zip_path = Path(os.environ.get("PFI_E2E_STUDY_ZIP", ""))
    if not zip_path.exists() or not zip_path.is_file():
        pytest.skip("PFI_E2E_STUDY_ZIP no disponible")
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))

    started = time.perf_counter()
    with zip_path.open("rb") as handle:
        body = register_study_zip(case_id="CASE-P105F-REAL", stream=handle)
    elapsed = round(time.perf_counter() - started, 3)
    summary = {
        "seriesCount": len(body["seriesFound"]),
        "planes": [
            {
                "plane": body[key]["plane"],
                "weighting": body[key]["weighting"],
                "sliceCount": body[key]["sliceCount"],
            }
            for key in ("sagittal", "axial")
            if key in body
        ],
        "elapsedSeconds": elapsed,
        "ok": "sagittal" in body or "axial" in body,
    }
    print(json.dumps(summary, sort_keys=True))
    assert summary["ok"] is True
    assert_no_internal_paths(summary, tmp_path)

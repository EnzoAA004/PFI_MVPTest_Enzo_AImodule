from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.asset_registry import (
    clear_asset_registry,
    is_public_browser_asset,
    registered_assets_for_run,
    resolve_run_asset,
)
from pfi_ai_service.real_inference_runtime import (
    LoadedInput,
    build_volume_slice_catalog,
    input_geometry_metadata,
    save_slice_catalog_assets,
)


def configure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("PFI_MODEL_DIR", str(tmp_path / "models"))
    clear_asset_registry()


def ready_model(model_key: str, plane: str) -> dict[str, Any]:
    return {
        "key": model_key,
        "plane": plane,
        "version": "test-v1",
        "readiness": "real_candidate_ready" if plane == "axial" else "real_baseline_ready",
        "trainingStatus": "candidate_below_quality_gate" if plane == "axial" else "final",
        "artifactHash": f"hash-{plane}",
        "baselineReady": plane == "sagittal",
        "availableForRealInference": True,
        "runtimeQualification": "axial_candidate_runtime_ready" if plane == "axial" else None,
        "qualityGatePassed": plane == "sagittal",
        "manifest": {"status": "valid", "valid": True, "trainingStatus": "final"},
    }


def test_slice_catalog_assets_use_loaded_volume_without_internal_paths(monkeypatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    loaded = LoadedInput(
        array=np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6),
        path=tmp_path / "input.mha",
        suffix=".mha",
        spacing_xyz=(1.0, 1.0, 2.0),
        metadata={"inputShapeNative": [4, 5, 6], "inputShapeCanonical": [4, 5, 6]},
    )
    overlay = tmp_path / "outputs" / "real_inference" / "run-cat" / "axial" / "overlay.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"overlay-bytes")

    outputs = save_slice_catalog_assets(
        "run-cat",
        "axial",
        loaded,
        selected_axis=0,
        selected_slice=2,
        target_size=(16, 16),
        overlay_path=str(overlay),
    )

    assert len([key for key in outputs if key.startswith("slicePreview")]) == 4
    assets = registered_assets_for_run("run-cat", "axial")
    assert sorted(name for name in assets if name.startswith("slice-")) == [
        "slice-000.png",
        "slice-001.png",
        "slice-002-overlay.png",
        "slice-002.png",
        "slice-003.png",
    ]
    for name in assets:
        assert is_public_browser_asset(name)
        record = resolve_run_asset("run-cat", "axial", name)
        assert record.path.resolve().is_relative_to((tmp_path / "outputs" / "real_inference" / "run-cat" / "axial").resolve())


def test_catalog_entries_are_ordered_and_only_selected_slice_has_results() -> None:
    catalog = build_volume_slice_catalog(
        run_id="run-cat",
        plane="sagittal",
        slice_count=3,
        selected_slice=1,
        measurement_values=[{"id": "m-1"}, {"id": "m-2"}],
        landmarks=[{"id": "lm-1"}],
    )

    assert [entry["index"] for entry in catalog] == [0, 1, 2]
    assert [entry["displayIndex"] for entry in catalog] == [1, 2, 3]
    assert all(entry["previewAsset"]["assetName"] == f"slice-{entry['index']:03d}.png" for entry in catalog)
    assert [entry["hasResults"] for entry in catalog] == [False, True, False]
    assert catalog[0]["overlayAsset"] is None
    assert catalog[2]["measurementIds"] == []
    assert catalog[1]["overlayAsset"]["assetName"] == "slice-001-overlay.png"
    assert catalog[1]["measurementIds"] == ["m-1", "m-2"]
    assert catalog[1]["landmarkIds"] == ["lm-1"]
    serialized = json.dumps(catalog)
    assert "relativePath" not in serialized
    assert "C:\\" not in serialized
    assert "/tmp/" not in serialized


def test_geometry_metadata_keeps_missing_header_values_honest(tmp_path: Path) -> None:
    loaded = LoadedInput(
        array=np.zeros((3, 8, 8), dtype=np.float32),
        path=tmp_path / "default_header.mha",
        suffix=".mha",
        spacing_xyz=(1.0, 1.0, 1.0),
        metadata={
            "spacingXyz": [1.0, 1.0, 1.0],
            "arrayAxisSpacingCanonical": [1.0, 1.0, 1.0],
            "origin": (0.0, 0.0, 0.0),
            "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        },
    )

    metadata = input_geometry_metadata(loaded)

    assert metadata["originMm"] is None
    assert metadata["directionMatrix"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert metadata["geometryComplete"] is False
    assert metadata["geometryMetadataSource"] == "incomplete_image_header"


def test_v2_response_exposes_sanitized_volume_catalog(monkeypatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)

    def fake_run_pipeline(request):
        slices = build_volume_slice_catalog(
            run_id="ax-run-cat",
            plane=request.plane,
            slice_count=3,
            selected_slice=1,
            measurement_values=[{"id": "axial-m-1"}],
            landmarks=[{"id": "axial-lm-1"}],
        )
        return {
            "runId": "ax-run-cat",
            "traceId": request.metadata["traceId"],
            "caseId": request.case_id,
            "plane": request.plane,
            "modelKey": request.model_key,
            "inferenceMode": "real_baseline",
            "series": [{"id": "series-ax", "plane": request.plane, "sliceCount": 3, "selectedSlice": 1, "status": "real_baseline_ready"}],
            "masks": [{"id": "mask-ax", "className": "raw_50", "classId": 1, "confidence": 0.91, "enabled": True}],
            "landmarks": [{"id": "axial-lm-1", "label": "raw_50 centroid", "x": 4.0, "y": 5.0}],
            "measurementValues": [{"id": "axial-m-1", "label": "area", "value": 12.0, "unit": "px2", "confidence": 0.9}],
            "quality": {"maskCount": 1, "landmarkCount": 1, "measurementCount": 1},
            "metadata": {
                **request.metadata,
                "seriesId": request.input_id,
                "sourceFormat": "mha",
                "inputFormat": ".mha",
                "inputSize": 1024,
                "inputShapeNative": [3, 32, 32],
                "inputShapeCanonical": [3, 32, 32],
                "inputOrientationTransform": "none",
                "spacingXyz": [0.7, 0.7, 3.0],
                "arrayAxisSpacingCanonical": [3.0, 0.7, 0.7],
                "originMm": [12.0, 13.0, 14.0],
                "directionMatrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "geometryComplete": True,
                "selectedSlice": 1,
                "sliceCount": 3,
                "selectedAxis": 0,
                "inPlaneSpacing": [0.7, 0.7],
                "processedShape": [256, 256],
                "slices": slices,
                "assets": {
                    "input.png": {"assetName": "input.png"},
                    "overlay.png": {"assetName": "overlay.png"},
                    "slice-000.png": {"assetName": "slice-000.png"},
                    "slice-001.png": {"assetName": "slice-001.png"},
                    "slice-001-overlay.png": {"assetName": "slice-001-overlay.png"},
                    "slice-002.png": {"assetName": "slice-002.png"},
                },
                "sourcePath": "C:\\internal\\must-not-leak.mha",
                "outputFiles": {"imagePath": "C:\\internal\\input.png"},
            },
        }

    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"]))
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", fake_run_pipeline)

    client = TestClient(app)
    input_id = client.post("/inputs", json={"caseId": "CASE-CAT", "plane": "axial", "sourceKey": "fixture:axial_sample"}).json()["inputId"]
    response = client.post(
        "/v2/multiplanar/run",
        headers={"X-Trace-Id": "trace-cat"},
        json={
            "caseId": "CASE-CAT",
            "inferenceMode": "real_baseline",
            "allowContractFallback": False,
            "planes": {"sagittal": None, "axial": {"inputId": input_id, "modelKey": "axial_t2_alkafri"}},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    axial = body["planes"]["axial"]
    assert axial["input"]["seriesId"] == input_id
    assert axial["input"]["sourceFormat"] == "mha"
    assert axial["input"]["originMm"] == [12.0, 13.0, 14.0]
    assert axial["input"]["geometryComplete"] is True
    assert len(axial["input"]["slices"]) == axial["input"]["sliceCount"] == 3
    assert axial["input"]["selectedSliceIndex"] == 1
    assert [entry["displayIndex"] for entry in axial["input"]["slices"]] == [1, 2, 3]
    assert [entry["hasResults"] for entry in axial["input"]["slices"]] == [False, True, False]
    assert axial["input"]["slices"][1]["overlayAsset"]["assetName"] == "slice-001-overlay.png"
    assert axial["input"]["slices"][1]["measurementIds"] == ["axial-m-1"]
    assert {asset["role"] for asset in axial["assets"]} >= {"slice_preview", "slice_overlay"}
    serialized = json.dumps(body)
    assert "C:\\internal" not in serialized
    assert "sourcePath" not in serialized
    assert "outputFiles" not in serialized

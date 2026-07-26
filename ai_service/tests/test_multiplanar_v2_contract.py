from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pfi_ai_service.api import app


def configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("PFI_MODEL_DIR", str(tmp_path / "models"))


def register_input(client: TestClient, case_id: str, plane: str) -> str:
    source_key = f"fixture:{plane}_sample"
    response = client.post("/inputs", json={"caseId": case_id, "plane": plane, "sourceKey": source_key})
    assert response.status_code == 200, response.text
    return response.json()["inputId"]


def ready_model(model_key: str, plane: str, *, ready: bool = True) -> dict[str, Any]:
    return {
        "key": model_key,
        "plane": plane,
        "version": "sagittal-spider-final-v1" if plane == "sagittal" else "axial-final-v2",
        "readiness": "real_baseline_ready" if ready else "candidate_below_quality_gate",
        "trainingStatus": "final" if plane == "sagittal" else "candidate_below_quality_gate",
        "artifactHash": "cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944" if plane == "sagittal" else "a48cbddd858b5615010fd809412f3d17dae6871fbe12a38f4720e6f6bc70f739",
        "baselineReady": ready,
        "availableForRealInference": ready,
        "manifest": {"status": "valid", "valid": True, "trainingStatus": "final" if plane == "sagittal" else "candidate_below_quality_gate"},
    }


def fake_plane_response(request) -> dict[str, Any]:
    plane = request.plane
    return {
        "run_id": f"{plane}-run-123",
        "runId": f"{plane}-run-123",
        "traceId": request.metadata["traceId"],
        "case_id": request.case_id,
        "caseId": request.case_id,
        "plane": plane,
        "model_key": request.model_key,
        "modelKey": request.model_key,
        "inferenceMode": "real_baseline",
        "inputPath": "C:\\internal\\must-not-leak.mha",
        "series": [{"id": f"series-{plane}-primary", "plane": plane, "sliceCount": 17, "selectedSlice": 7, "status": "real_baseline_ready"}],
        "masks": [
            {"id": f"mask-{plane}-vertebra", "className": "vertebra_group", "confidence": 0.97, "enabled": True},
            {"id": f"mask-{plane}-canal", "className": "canal", "confidence": 0.96, "enabled": True},
            {"id": f"mask-{plane}-disc", "className": "disc_group", "confidence": 0.95, "enabled": True},
        ],
        "landmarks": [
            {"id": "lm-1", "label": "vertebra_group centroid", "x": 10.0, "y": 20.0},
            {"id": "lm-2", "label": "canal centroid", "x": 30.0, "y": 40.0},
            {"id": "lm-3", "label": "disc_group centroid", "x": 50.0, "y": 60.0},
        ],
        "measurementValues": [
            {"id": f"m-{index}", "label": f"measurement {index}", "value": float(index), "unit": "mm", "confidence": 0.9, "linkedLandmarks": []}
            for index in range(9)
        ],
        "quality": {"maskCount": 3, "landmarkCount": 3, "measurementCount": 9, "meanConfidence": 0.98, "meanForegroundConfidence": 0.96, "foregroundRatio": 0.2},
        "metadata": {
            **request.metadata,
            "inputFormat": ".mha",
            "inputSize": 2361432,
            "inputShapeNative": [352, 384, 17],
            "inputShapeCanonical": [352, 384, 17],
            "inputOrientationTransform": "none",
            "spacingXyz": [0.7408, 0.6770, 3.0],
            "arrayAxisSpacingCanonical": [0.7408, 0.6770, 3.0],
            "selectedSlice": 7,
            "sliceCount": 17,
            "selectedAxis": 2,
            "inPlaneSpacing": [0.7408, 0.6770],
            "processedShape": [256, 256],
            "outputFiles": {
                "imagePath": {"generated": True, "fileName": "input.png"},
                "overlayPath": {"generated": True, "fileName": "overlay.png"},
                "maskPath": {"generated": True, "fileName": "mask.npy"},
                "confidencePath": {"generated": True, "fileName": "confidence.npy"},
            },
        },
    }


def walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys |= walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= walk_keys(item)
    return keys


def test_v2_sagittal_only_canonical_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=True))
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", lambda request: calls.append(request) or fake_plane_response(request))
    client = TestClient(app)
    case_id = "CASE-P9A-SAG"
    sagittal_id = register_input(client, case_id, "sagittal")

    response = client.post(
        "/v2/multiplanar/run",
        headers={"X-Trace-Id": "trace-p9a-sag"},
        json={
            "caseId": case_id,
            "inferenceMode": "real_baseline",
            "allowContractFallback": False,
            "planes": {"sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"}, "axial": None},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(calls) == 1
    assert calls[0].plane == "sagittal"
    assert body["schemaVersion"] == "pfi.multiplanar-run.v2"
    assert body["workspaceMode"] == "sagittal_only"
    assert body["requestedPlanes"] == ["sagittal"]
    assert body["completedPlanes"] == ["sagittal"]
    assert body["readiness"] == {"sagittal": True, "axial": False, "dual": False}
    assert body["planes"]["sagittal"]["model"]["artifactHash"] == "cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944"
    assert body["planes"]["sagittal"]["input"]["nativeShape"] == [352, 384, 17]
    assert body["planes"]["sagittal"]["coordinateSpace"]["width"] == 256
    assert body["planes"]["sagittal"]["coordinateSpace"]["height"] == 256
    assert body["planes"]["axial"] is None
    assert body["threeD"]["status"] == "blocked_missing_axial"
    assert body["governance"]["humanReviewRequired"] is True
    assert body["governance"]["notClinicalDiagnosis"] is True
    assert body["governance"]["diagnosisGenerated"] is False
    forbidden = {"run_id", "case_id", "model_key", "input_path", "inputPath", "overlay_path", "overlayPath", "agent_decision", "human_review_required", "not_clinical_diagnosis", "measurementValues"}
    assert not (walk_keys(body) & forbidden)
    serialized = json.dumps(body)
    assert "C:\\" not in serialized
    assert "localhost" not in serialized
    report = tmp_path / "outputs" / "multiplanar_reports_v2" / f"{body['runId']}.json"
    assert json.loads(report.read_text(encoding="utf-8")) == body


def test_v2_axial_only_not_ready_does_not_execute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=(info["plane"] == "sagittal")))
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", lambda request: calls.append(request) or fake_plane_response(request))
    client = TestClient(app)
    case_id = "CASE-P9A-AX"
    axial_id = register_input(client, case_id, "axial")

    response = client.post(
        "/v2/multiplanar/run",
        headers={"X-Trace-Id": "trace-p9a-ax"},
        json={
            "caseId": case_id,
            "inferenceMode": "real_baseline",
            "planes": {"sagittal": None, "axial": {"inputId": axial_id, "modelKey": "axial_t2_alkafri"}},
        },
    )

    assert response.status_code == 409
    assert calls == []
    body = response.json()
    assert body["schemaVersion"] == "pfi.error.v2"
    assert body["code"] == "MODEL_NOT_READY"
    assert body["requestedPlanes"] == ["axial"]
    assert body["details"]["trainingStatus"] == "candidate_below_quality_gate"


def test_v2_dual_blocks_before_sagittal_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=(info["plane"] == "sagittal")))
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", lambda request: calls.append(request) or fake_plane_response(request))
    client = TestClient(app)
    case_id = "CASE-P9A-DUAL"
    sagittal_id = register_input(client, case_id, "sagittal")
    axial_id = register_input(client, case_id, "axial")

    response = client.post(
        "/v2/multiplanar/run",
        json={
            "caseId": case_id,
            "inferenceMode": "real_baseline",
            "planes": {
                "sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"},
                "axial": {"inputId": axial_id, "modelKey": "axial_t2_alkafri"},
            },
        },
    )

    assert response.status_code == 409
    assert calls == []
    assert response.json()["requestedPlanes"] == ["sagittal", "axial"]


def test_v2_no_planes_returns_structured_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch, tmp_path)
    response = TestClient(app).post("/v2/multiplanar/run", json={"caseId": "CASE-P9A-NONE", "planes": {"sagittal": None, "axial": None}})
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "NO_PLANE_REQUESTED"
    assert body["schemaVersion"] == "pfi.error.v2"


def test_v2_contract_reflects_axial_candidate_status() -> None:
    body = TestClient(app).get("/v2/multiplanar/contract").json()
    assert body["schemaVersion"] == "pfi.multiplanar-contract.v2"
    assert body["capabilities"]["supportedWorkspaceModes"] == ["sagittal_only", "axial_only", "dual_plane"]
    assert body["models"]["axial"]["trainingStatus"] == "candidate_below_quality_gate"
    assert body["models"]["axial"]["artifactHash"] == "a48cbddd858b5615010fd809412f3d17dae6871fbe12a38f4720e6f6bc70f739"
    assert body["readiness"]["axial"] is False

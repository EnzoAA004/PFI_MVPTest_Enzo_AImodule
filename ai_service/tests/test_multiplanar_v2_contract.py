from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.asset_registry import clear_asset_registry, register_run_assets


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
        "readiness": "real_baseline_ready" if plane == "sagittal" and ready else "real_candidate_ready" if plane == "axial" and ready else "candidate_below_quality_gate",
        "trainingStatus": "final" if plane == "sagittal" else "candidate_below_quality_gate",
        "artifactHash": "cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944" if plane == "sagittal" else "a48cbddd858b5615010fd809412f3d17dae6871fbe12a38f4720e6f6bc70f739",
        "baselineReady": ready if plane == "sagittal" else False,
        "availableForRealInference": ready,
        "runtimeQualification": "axial_candidate_runtime_ready" if plane == "axial" and ready else None,
        "qualityGatePassed": True if plane == "sagittal" else False,
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
            {"id": f"mask-{plane}-vertebra", "className": "vertebra_group" if plane == "sagittal" else "raw_50", "classId": 1 if plane == "sagittal" else 2, "confidence": 0.97, "enabled": True},
            {"id": f"mask-{plane}-canal", "className": "canal" if plane == "sagittal" else "raw_100", "classId": 2 if plane == "sagittal" else 3, "confidence": 0.96, "enabled": True},
            {"id": f"mask-{plane}-disc", "className": "disc_group" if plane == "sagittal" else "raw_150", "classId": 3 if plane == "sagittal" else 4, "confidence": 0.95, "enabled": True},
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


def fake_plane_response_with_registered_mask(request, tmp_path: Path) -> dict[str, Any]:
    body = fake_plane_response(request)
    run_id = body["runId"]
    plane = request.plane
    output_dir = tmp_path / "outputs" / "real_inference" / run_id / plane
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((256, 256), dtype=np.uint8)
    if plane == "sagittal":
        mask[80:170, 96:150] = 1
        mask[130:190, 120:170] = 2
        mask[190:230, 150:210] = 3
    else:
        mask[90:150, 70:135] = 2
        mask[130:190, 115:180] = 3
        mask[180:230, 150:220] = 4
    mask_path = output_dir / "mask.npy"
    np.save(mask_path, mask)
    register_run_assets(run_id, plane, {"maskPath": str(mask_path)})
    body["metadata"]["assets"] = {"mask.npy": {"assetName": "mask.npy", "generated": True}}
    return body


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
    assert body["synthetic"] is False
    assert body["fallbackReason"] is None
    assert body["planes"]["sagittal"]["synthetic"] is False
    assert body["planes"]["sagittal"]["fallbackReason"] is None
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
    assert body["models"]["axial"]["baselineReady"] is False
    assert body["models"]["axial"]["availableForRealInference"] is True
    assert body["models"]["axial"]["readiness"] == "real_candidate_ready"
    assert body["readiness"]["axial"] is True


def test_v1_route_uses_canonical_executor_not_deprecated_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    called = {"executor": False, "deprecated": False}

    def fail_deprecated(_request):
        called["deprecated"] = True
        raise AssertionError("deprecated run_multiplanar_pipeline should not be called")

    class FakeExecutor:
        def run(self, request, *, trace_id, allow_unregistered_inputs=False):
            called["executor"] = True
            assert allow_unregistered_inputs is True
            assert request.caseId == "CASE-P9A-V1"
            return fake_v2_response(request.caseId, trace_id)

    monkeypatch.setattr("pfi_ai_service.multiplanar_routes.run_multiplanar_pipeline", fail_deprecated, raising=False)
    monkeypatch.setattr("pfi_ai_service.multiplanar_routes.CanonicalMultiplanarExecutor", lambda: FakeExecutor())

    response = TestClient(app).post(
        "/multiplanar/run",
        headers={"X-Trace-Id": "trace-p9a-v1"},
        json={"caseId": "CASE-P9A-V1", "metadata": {"inferenceMode": "contract"}},
    )

    assert response.status_code == 200, response.text
    assert called == {"executor": True, "deprecated": False}
    body = response.json()
    assert body["schemaVersion"] == "multiplanar-run-v1"
    assert body["planes"]["sagittal"]["measurements"]["values"] == []
    assert body["planes"]["sagittal"]["measurementValues"] == []
    assert body["planes"]["sagittal"]["synthetic"] is True


def test_v2_explicit_contract_is_synthetic_and_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", lambda request: (_ for _ in ()).throw(AssertionError("pipeline should not run")))
    client = TestClient(app)
    case_id = "CASE-P9A-CONTRACT"
    sagittal_id = register_input(client, case_id, "sagittal")

    response = client.post(
        "/v2/multiplanar/run",
        json={"caseId": case_id, "inferenceMode": "contract", "planes": {"sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"}, "axial": None}},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    plane = body["planes"]["sagittal"]
    assert body["effectiveInferenceMode"] == "contract"
    assert body["synthetic"] is True
    assert body["fallbackReason"] == "explicit_contract_mode"
    assert plane["synthetic"] is True
    assert plane["series"] == []
    assert plane["assets"] == []
    assert plane["masks"] == []
    assert plane["landmarks"] == []
    assert plane["measurements"] == []
    assert "PAT-DEMO" not in json.dumps(body)
    assert "Sagittal T1" not in json.dumps(body)


def test_v2_explicit_mock_is_synthetic_and_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch, tmp_path)
    client = TestClient(app)
    case_id = "CASE-P9A-MOCK"
    sagittal_id = register_input(client, case_id, "sagittal")
    response = client.post(
        "/v2/multiplanar/run",
        json={"caseId": case_id, "inferenceMode": "mock", "planes": {"sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"}, "axial": None}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["effectiveInferenceMode"] == "mock"
    assert body["synthetic"] is True
    assert body["fallbackReason"] == "explicit_mock_mode"
    assert body["planes"]["sagittal"]["masks"] == []


def test_v2_real_failure_with_fallback_is_global_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=True))

    def fail(request):
        calls.append(request)
        raise RuntimeError("boom with C:\\secret")

    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", fail)
    client = TestClient(app)
    case_id = "CASE-P9A-FALLBACK"
    sagittal_id = register_input(client, case_id, "sagittal")

    response = client.post(
        "/v2/multiplanar/run",
        json={
            "caseId": case_id,
            "inferenceMode": "real_baseline",
            "allowContractFallback": True,
            "planes": {"sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"}, "axial": None},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(calls) == 1
    assert body["requestedInferenceMode"] == "real_baseline"
    assert body["effectiveInferenceMode"] == "contract"
    assert body["synthetic"] is True
    assert body["fallbackReason"]
    assert "C:\\" not in body["fallbackReason"]
    assert body["planes"]["sagittal"]["effectiveInferenceMode"] == "contract"


def test_v2_real_failure_without_fallback_is_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=True))
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", lambda request: (_ for _ in ()).throw(RuntimeError("boom")))
    client = TestClient(app)
    case_id = "CASE-P9A-NO-FALLBACK"
    sagittal_id = register_input(client, case_id, "sagittal")
    response = client.post(
        "/v2/multiplanar/run",
        json={"caseId": case_id, "inferenceMode": "real_baseline", "allowContractFallback": False, "planes": {"sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"}, "axial": None}},
    )
    assert response.status_code == 500
    assert response.json()["code"] == "REAL_INFERENCE_FAILED"


def test_v2_invalid_json_and_non_object_body_are_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch, tmp_path)
    client = TestClient(app)
    invalid = client.post("/v2/multiplanar/run", content="{bad", headers={"content-type": "application/json"})
    assert invalid.status_code == 400
    assert invalid.json()["code"] == "INVALID_MULTIPLANAR_REQUEST"
    for payload in ([], "x", None):
        response = client.post("/v2/multiplanar/run", json=payload)
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_MULTIPLANAR_REQUEST"


def test_v2_generated_and_sanitized_trace_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch, tmp_path)
    client = TestClient(app)
    no_header = client.post("/v2/multiplanar/run", json={"caseId": "CASE-TRACE", "planes": {"sagittal": None, "axial": None}})
    assert no_header.status_code == 400
    assert no_header.json()["traceId"].startswith("trace-")
    assert no_header.headers["X-Trace-Id"] == no_header.json()["traceId"]
    sanitized = client.post(
        "/v2/multiplanar/run",
        headers={"X-Trace-Id": "trace bad !"},
        json={"caseId": "CASE-TRACE", "planes": {"sagittal": None, "axial": None}},
    )
    assert sanitized.json()["traceId"] == "trace-bad--"
    assert sanitized.headers["X-Trace-Id"] == "trace-bad--"


def test_v2_real_coordinate_space_zero_is_structured_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=True))

    def bad_response(request):
        body = fake_plane_response(request)
        body["metadata"]["processedShape"] = [0, 0]
        return body

    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", bad_response)
    client = TestClient(app)
    case_id = "CASE-P9A-BAD-COORD"
    sagittal_id = register_input(client, case_id, "sagittal")
    response = client.post(
        "/v2/multiplanar/run",
        json={"caseId": case_id, "inferenceMode": "real_baseline", "planes": {"sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"}, "axial": None}},
    )
    assert response.status_code == 500
    assert response.json()["code"] == "INVALID_MULTIPLANAR_RESPONSE"


def test_v2_real_missing_series_is_not_fabricated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=True))

    def bad_response(request):
        body = fake_plane_response(request)
        body["series"] = []
        return body

    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", bad_response)
    client = TestClient(app)
    case_id = "CASE-P9A-NO-SERIES"
    sagittal_id = register_input(client, case_id, "sagittal")
    response = client.post(
        "/v2/multiplanar/run",
        json={"caseId": case_id, "inferenceMode": "real_baseline", "planes": {"sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"}, "axial": None}},
    )
    assert response.status_code == 500
    assert response.json()["code"] == "INVALID_MULTIPLANAR_RESPONSE"


def test_v2_dual_real_baseline_blocks_3d_without_explicit_anatomical_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    monkeypatch.delenv("PFI_MULTIPLANAR_3D_ANATOMICAL_MAPPING_JSON", raising=False)
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=True))
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", lambda request: fake_plane_response_with_registered_mask(request, tmp_path))
    client = TestClient(app)
    case_id = "CASE-P9A3-DUAL-3D"
    sagittal_id = register_input(client, case_id, "sagittal")
    axial_id = register_input(client, case_id, "axial")

    response = client.post(
        "/v2/multiplanar/run",
        headers={"X-Trace-Id": "trace-p9a3-dual-3d"},
        json={
            "caseId": case_id,
            "inferenceMode": "real_baseline",
            "allowContractFallback": False,
            "planes": {
                "sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"},
                "axial": {"inputId": axial_id, "modelKey": "axial_t2_alkafri"},
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["workspaceMode"] == "dual_plane"
    assert body["effectiveInferenceMode"] == "real_baseline"
    assert body["planes"]["axial"]["model"]["baselineReady"] is False
    assert body["planes"]["axial"]["model"]["availableForRealInference"] is True
    assert body["planes"]["axial"]["model"]["runtimeQualification"] == "axial_candidate_runtime_ready"
    assert body["planes"]["axial"]["model"]["qualityGatePassed"] is False
    assert body["threeD"]["enabled"] is False
    assert body["threeD"]["status"] == "experimental_blocked_missing_anatomical_mapping"
    assert body["threeD"]["assets"] == []
    assert "missing_explicit_anatomical_mapping" in body["threeD"]["reconstruction"]["blockedReasons"]


def test_v2_dual_real_baseline_generates_experimental_3d_proxy_with_explicit_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    monkeypatch.setenv("PFI_MULTIPLANAR_3D_ANATOMICAL_MAPPING_JSON", '{"vertebra_group":["raw_50"],"canal":["raw_100"],"disc_group":["raw_150"]}')
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=True))
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", lambda request: fake_plane_response_with_registered_mask(request, tmp_path))
    client = TestClient(app)
    case_id = "CASE-P9A3-DUAL-3D-MAPPING"
    sagittal_id = register_input(client, case_id, "sagittal")
    axial_id = register_input(client, case_id, "axial")

    response = client.post(
        "/v2/multiplanar/run",
        headers={"X-Trace-Id": "trace-p9a3-dual-3d-mapping"},
        json={
            "caseId": case_id,
            "inferenceMode": "real_baseline",
            "allowContractFallback": False,
            "planes": {
                "sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"},
                "axial": {"inputId": axial_id, "modelKey": "axial_t2_alkafri"},
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["threeD"]["enabled"] is True
    assert body["threeD"]["status"] == "experimental_ready"
    assert body["threeD"]["reconstruction"]["source"] == "real_segmentation_masks"
    assert body["threeD"]["reconstruction"]["kind"] == "experimental_geometric_proxy"
    assert body["threeD"]["reconstruction"]["method"] == "dual_plane_bbox_proxy"
    assert body["threeD"]["reconstruction"]["anatomicalReconstruction"] is False
    assert body["threeD"]["reconstruction"]["volumetricReconstruction"] is False
    assert body["threeD"]["reconstruction"]["coordinateSystem"] == "local_proxy_space"
    assert body["threeD"]["reconstruction"]["structureCount"] >= 1
    assert body["threeD"]["reconstruction"]["traceability"]["sourcePlaneRunIds"] == {
        "sagittal": "sagittal-run-123",
        "axial": "axial-run-123",
    }
    assert body["threeD"]["assets"] == [
        {
            "assetName": "lumbar-3d-mesh.json",
            "role": "mesh_3d",
            "contentType": "application/json",
            "generated": True,
            "relativePath": f"/assets/{body['runId']}/workspace/lumbar-3d-mesh.json",
        }
    ]
    assert "experimental" in json.dumps(body["threeD"])

    asset = client.get(body["threeD"]["assets"][0]["relativePath"])
    assert asset.status_code == 200
    assert asset.headers["content-type"].startswith("application/json")
    mesh = asset.json()
    assert mesh["schemaVersion"] == "pfi.lumbar-geometric-proxy.v1"
    assert mesh["experimental"] is True
    assert mesh["kind"] == "experimental_geometric_proxy"
    assert mesh["method"] == "dual_plane_bbox_proxy"
    assert mesh["anatomicalReconstruction"] is False
    assert mesh["volumetricReconstruction"] is False
    assert mesh["coordinateSystem"] == "local_proxy_space"
    assert mesh["units"] == "normalized"
    assert mesh["vertices"]
    assert mesh["faces"]
    assert mesh["traceability"]["models"]["axial"]["artifactHash"] == "a48cbddd858b5615010fd809412f3d17dae6871fbe12a38f4720e6f6bc70f739"
    assert mesh["traceability"]["parameters"]["explicitAnatomicalMapping"] == {
        "vertebra_group": ["raw_50"],
        "canal": ["raw_100"],
        "disc_group": ["raw_150"],
    }
    assert mesh["traceability"]["transforms"]["sagittal"]["depthSpacingMm"] is None
    assert "extrusion" not in json.dumps(mesh).lower()
    assert "patient_relative_mm" not in json.dumps(mesh)

    clear_asset_registry()
    rehydrated = client.get(body["threeD"]["assets"][0]["relativePath"])
    assert rehydrated.status_code == 200
    assert rehydrated.json()["schemaVersion"] == "pfi.lumbar-geometric-proxy.v1"


def test_v2_dual_fallback_does_not_generate_3d_mesh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=True))

    def maybe_synthetic(request):
        if request.plane == "axial":
            raise RuntimeError("axial unavailable")
        return fake_plane_response_with_registered_mask(request, tmp_path)

    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", maybe_synthetic)
    client = TestClient(app)
    case_id = "CASE-P9A3-FALLBACK-3D"
    sagittal_id = register_input(client, case_id, "sagittal")
    axial_id = register_input(client, case_id, "axial")

    response = client.post(
        "/v2/multiplanar/run",
        json={
            "caseId": case_id,
            "inferenceMode": "real_baseline",
            "allowContractFallback": True,
            "planes": {
                "sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"},
                "axial": {"inputId": axial_id, "modelKey": "axial_t2_alkafri"},
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["effectiveInferenceMode"] == "contract"
    assert body["threeD"]["enabled"] is False
    assert body["threeD"]["status"] == "experimental_blocked_insufficient_geometry"
    assert body["threeD"]["assets"] == []
    assert "axial_not_real_baseline" in body["threeD"]["reconstruction"]["blockedReasons"]


def fake_v2_response(case_id: str, trace_id: str):
    from pfi_ai_service.multiplanar_v2_models import (
        CoordinateSpaceV2,
        GovernanceV2,
        MultiplanarReadinessV2,
        MultiplanarRunV2Response,
        PlaneInputV2,
        PlaneModelV2,
        PlaneQualityV2,
        PlaneRunV2Result,
        ReviewPolicyV2,
        ThreeDStatusV2,
        WorkspaceQualityV2,
    )

    quality = PlaneQualityV2(maskCount=0, landmarkCount=0, measurementCount=0)
    sagittal = PlaneRunV2Result(
        status="ready",
        plane="sagittal",
        runId="synthetic-sag",
        effectiveInferenceMode="contract",
        model=PlaneModelV2(key="sagittal_spider", version="contract", readiness="contract", trainingStatus=None, artifactHash=None, baselineReady=False, availableForRealInference=False, manifestStatus=None, manifestValid=False),
        input=PlaneInputV2(inputId="legacy", format=None, sizeBytes=None, nativeShape=None, canonicalShape=None, orientationTransform=None, spacingXyzMm=None, canonicalAxisSpacingMm=None, selectedSliceIndex=None, sliceCount=None, selectedAxis=None, inPlaneSpacingMm=None),
        coordinateSpace=CoordinateSpaceV2(name="synthetic_empty_pixel_space", width=1, height=1, units="pixel", origin="top_left", xDirection="right", yDirection="down", sourceSliceIndex=None, sourceAxis=None),
        series=[],
        assets=[],
        masks=[],
        landmarks=[],
        measurements=[],
        quality=quality,
        synthetic=True,
        fallbackReason="explicit_contract_mode",
    )
    return MultiplanarRunV2Response(
        status="completed",
        schemaVersion="pfi.multiplanar-run.v2",
        runId="multi-fake",
        traceId=trace_id,
        caseId=case_id,
        workspaceMode="sagittal_only",
        requestedInferenceMode="contract",
        effectiveInferenceMode="contract",
        requestedPlanes=["sagittal"],
        completedPlanes=["sagittal"],
        readiness=MultiplanarReadinessV2(sagittal=False, axial=False, dual=False),
        planes={"sagittal": sagittal, "axial": None},
        threeD=ThreeDStatusV2(enabled=False, status="blocked_missing_axial", sourcePlaneRunIds={"sagittal": "synthetic-sag", "axial": None}, requiredInputs=[]),
        quality=WorkspaceQualityV2(planeCount=1, maskCount=0, landmarkCount=0, measurementCount=0, byPlane={"sagittal": quality, "axial": None}),
        review=ReviewPolicyV2(status="pending", required=True, approvalRequiresHumanConfirmation=True),
        governance=GovernanceV2(humanReviewRequired=True, notClinicalDiagnosis=True),
        synthetic=True,
        fallbackReason="explicit_contract_mode",
    )

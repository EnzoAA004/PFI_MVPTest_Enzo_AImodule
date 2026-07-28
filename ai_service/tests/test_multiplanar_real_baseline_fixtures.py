from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.asset_registry import clear_asset_registry
from pfi_ai_service.real_inference_runtime import clear_model_cache


def test_multiplanar_run_executes_dual_plane_real_baseline_with_axial_candidate(monkeypatch, tmp_path) -> None:
    sagittal_fixture = Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy")
    axial_fixture = Path("ai_service/tests/fixtures/real_baseline/axial_sample_input.npy")
    assert sagittal_fixture.exists()
    assert axial_fixture.exists()

    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()

    response = TestClient(app).post(
        "/multiplanar/run",
        headers={"X-Trace-Id": "trace-ai012-multiplanar-fixture"},
        json={
            "caseId": "CASE-AI012-MULTI-FIXTURE",
            "sagittalInputPath": str(sagittal_fixture),
            "axialInputPath": str(axial_fixture),
            "sagittalModelKey": "sagittal_spider",
            "axialModelKey": "axial_t2_alkafri",
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": True,
                "traceId": "trace-ai012-multiplanar-fixture",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["runId"].startswith("multi-")
    assert body["metadata"]["multiplanarRunId"] == body["runId"]
    assert body["traceId"] == "trace-ai012-multiplanar-fixture"
    assert body["effectiveInferenceMode"] == "real_baseline"
    assert body["requestedInferenceMode"] == "real_baseline"
    assert body["sagittalRunReady"] is True
    assert body["axialRunReady"] is True
    assert body["dualRunReady"] is True

    sagittal = body["planes"]["sagittal"]
    axial = body["planes"]["axial"]
    assert sagittal["runId"]
    assert axial["runId"]
    assert sagittal["runId"] != axial["runId"]
    assert body["threeD"]["sourcePlaneRunIds"] == {
        "sagittal": sagittal["runId"],
        "axial": axial["runId"],
    }

    assert sagittal["aiOutput"]["inferenceMode"] == "real_baseline"
    assert sagittal["metadata"]["inferenceMode"] == "real_baseline"
    assert sagittal["traceId"] == "trace-ai012-multiplanar-fixture"
    assert sagittal["synthetic"] is False
    assert sagittal["fallbackReason"] is None
    output_files = sagittal["metadata"]["outputFiles"]
    assert output_files["overlayPath"]["generated"] is True

    assert axial["aiOutput"]["inferenceMode"] == "real_baseline"
    assert axial["metadata"]["inferenceMode"] == "real_baseline"
    assert axial["synthetic"] is False
    assert axial["fallbackReason"] is None
    assert axial["modelArtifact"]["baselineReady"] is False
    assert axial["modelArtifact"]["availableForRealInference"] is True
    assert axial["modelArtifact"]["readiness"] == "real_candidate_ready"
    assert axial["modelArtifact"]["manifest"]["trainingStatus"] == "candidate_below_quality_gate"
    assert axial["quality"]["maskCount"] > 0
    assert axial["quality"]["landmarkCount"] > 0
    assert axial["quality"]["measurementCount"] > 0
    assert axial["assets"]["overlay.png"]["generated"] is True
    assert axial["assets"]["mask-preview.png"]["generated"] is True


def test_multiplanar_real_fixtures_transport_experimental_3d_proxy_with_explicit_mapping(monkeypatch, tmp_path) -> None:
    sagittal_fixture = Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy")
    axial_fixture = Path("ai_service/tests/fixtures/real_baseline/axial_sample_input.npy")
    assert sagittal_fixture.exists()
    assert axial_fixture.exists()

    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    monkeypatch.setenv(
        "PFI_MULTIPLANAR_3D_ANATOMICAL_MAPPING_JSON",
        '{"vertebra_group":["raw_50"],"canal":["raw_100"]}',
    )
    clear_model_cache()
    clear_asset_registry()
    client = TestClient(app)
    sagittal_input = client.post("/inputs", json={"caseId": "CASE-P9A31-REAL-PROXY", "plane": "sagittal", "sourceKey": "fixture:sagittal_sample"})
    axial_input = client.post("/inputs", json={"caseId": "CASE-P9A31-REAL-PROXY", "plane": "axial", "sourceKey": "fixture:axial_sample"})
    assert sagittal_input.status_code == 200, sagittal_input.text
    assert axial_input.status_code == 200, axial_input.text

    response = client.post(
        "/v2/multiplanar/run",
        headers={"X-Trace-Id": "trace-p9a31-real-proxy"},
        json={
            "caseId": "CASE-P9A31-REAL-PROXY",
            "inferenceMode": "real_baseline",
            "allowContractFallback": False,
            "planes": {
                "sagittal": {"inputId": sagittal_input.json()["inputId"], "modelKey": "sagittal_spider"},
                "axial": {"inputId": axial_input.json()["inputId"], "modelKey": "axial_t2_alkafri"},
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["effectiveInferenceMode"] == "real_baseline"
    assert body["planes"]["axial"]["model"]["baselineReady"] is False
    assert body["planes"]["axial"]["model"]["readiness"] == "real_candidate_ready"
    assert body["planes"]["axial"]["model"]["runtimeQualification"] == "axial_candidate_runtime_ready"
    assert body["planes"]["axial"]["model"]["qualityGatePassed"] is False
    assert body["threeD"]["enabled"] is True
    assert body["threeD"]["status"] == "experimental_ready"
    assert body["threeD"]["reconstruction"]["kind"] == "experimental_geometric_proxy"
    assert body["threeD"]["reconstruction"]["method"] == "dual_plane_bbox_proxy"
    assert body["threeD"]["reconstruction"]["anatomicalReconstruction"] is False
    assert body["threeD"]["reconstruction"]["volumetricReconstruction"] is False

    mesh_response = client.get(body["threeD"]["assets"][0]["relativePath"])
    assert mesh_response.status_code == 200
    mesh = mesh_response.json()
    assert mesh["schemaVersion"] == "pfi.lumbar-geometric-proxy.v1"
    assert mesh["traceability"]["models"]["sagittal"]["artifactHash"]
    assert mesh["traceability"]["models"]["axial"]["artifactHash"] == "a48cbddd858b5615010fd809412f3d17dae6871fbe12a38f4720e6f6bc70f739"
    assert mesh["traceability"]["transforms"]["axial"]["depthSpacingMm"] is None

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.real_inference_runtime import clear_model_cache


SAGITTAL_SOURCE = "fixture:sagittal_sample"
AXIAL_SOURCE = "fixture:axial_sample"


def assert_no_internal_paths(value: object) -> None:
    text = str(value)
    forbidden = [
        "ai_service/tests/fixtures/real_baseline",
        "outputs\\",
        "outputs/",
        "models/final",
    ]
    assert not any(item in text for item in forbidden), text[:500]


def register_input(client: TestClient, *, case_id: str, plane: str, source_key: str) -> str:
    response = client.post(
        "/inputs",
        json={"caseId": case_id, "plane": plane, "sourceKey": source_key},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["inputId"].startswith("inp_")
    assert body["caseId"] == case_id
    assert body["plane"] == plane
    assert body["format"] == "npy"
    assert body["size"] > 0
    assert_no_internal_paths(body)
    return body["inputId"]


def test_inputs_register_returns_opaque_input_id_without_path() -> None:
    client = TestClient(app)
    input_id = register_input(
        client,
        case_id="CASE-AI013-REGISTER",
        plane="sagittal",
        source_key=SAGITTAL_SOURCE,
    )

    assert "/" not in input_id
    assert "\\" not in input_id


def test_pipeline_run_accepts_input_id_for_real_baseline(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()
    client = TestClient(app)
    case_id = "CASE-AI013-PIPELINE"
    input_id = register_input(client, case_id=case_id, plane="sagittal", source_key=SAGITTAL_SOURCE)

    response = client.post(
        "/pipeline/run",
        json={
            "caseId": case_id,
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "inputId": input_id,
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-ai013-pipeline",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["inputId"] == input_id
    assert body["aiOutput"]["inferenceMode"] == "real_baseline"
    assert body["traceId"] == "trace-ai013-pipeline"
    assert body["metadata"]["inputId"] == input_id
    assert body["metadata"]["outputFiles"]["maskPath"] == {"generated": True, "fileName": "mask.npy"}
    assert "inputPath" not in body
    assert "sourcePath" not in body["metadata"]
    assert_no_internal_paths(body)


def test_pipeline_run_rejects_unknown_input_id() -> None:
    response = TestClient(app).post(
        "/pipeline/run",
        json={
            "caseId": "CASE-AI013-MISSING",
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "inputId": "inp_missing",
            "metadata": {"inferenceMode": "real_baseline", "allowContractFallback": False},
        },
    )

    assert response.status_code == 404
    body = response.json()
    assert body["status"] == "error"
    assert body["message"] == "inputId no registrado"
    assert_no_internal_paths(body)


def test_multiplanar_run_accepts_input_ids_for_real_baseline(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()
    client = TestClient(app)
    case_id = "CASE-AI013-MULTI"
    sagittal_id = register_input(client, case_id=case_id, plane="sagittal", source_key=SAGITTAL_SOURCE)
    axial_id = register_input(client, case_id=case_id, plane="axial", source_key=AXIAL_SOURCE)

    response = client.post(
        "/multiplanar/run",
        headers={"X-Trace-Id": "trace-ai013-multi"},
        json={
            "caseId": case_id,
            "sagittalInputId": sagittal_id,
            "axialInputId": axial_id,
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-ai013-multi",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["effectiveInferenceMode"] == "real_baseline"
    assert body["runId"].startswith("multi-")
    assert body["traceId"] == "trace-ai013-multi"
    assert body["planes"]["sagittal"]["inputId"] == sagittal_id
    assert body["planes"]["axial"]["inputId"] == axial_id
    assert body["planes"]["sagittal"]["aiOutput"]["inferenceMode"] == "real_baseline"
    assert body["planes"]["axial"]["aiOutput"]["inferenceMode"] == "real_baseline"
    assert_no_internal_paths(body)

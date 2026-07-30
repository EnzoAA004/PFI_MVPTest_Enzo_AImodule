from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from pfi_ai_service.api import TRACE_ID_HEADER, app
from pfi_ai_service import pipeline as pipeline_module


INTERNAL_MARKERS = [
    "C:\\",
    "/tmp/",
    "/app/",
    "/models/",
    "/outputs/",
    "host.docker.internal",
    "localhost",
    "Bearer ",
    "eyJ",
    "sourcePath",
    "inputPath",
    "outputFiles",
    "stack",
    "traceback",
]


def assert_no_internal_detail(value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False)
    leaked = [marker for marker in INTERNAL_MARKERS if marker in serialized]
    assert leaked == [], serialized[:800]


def ready_artifact_status() -> dict[str, Any]:
    return {
        "availableForRealInference": True,
        "readiness": "real_baseline_ready",
        "version": "test-v1",
        "artifact": {"exists": True, "sha256": "abc123", "integrityStatus": "hashed"},
        "artifactHash": "abc123",
        "baselineReady": True,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }


def failing_real_inference(*_: Any, **__: Any) -> dict[str, Any]:
    raise RuntimeError(
        "boom C:\\internal\\models\\checkpoint.pt /tmp/private host.docker.internal Bearer eyJsecret"
    )


def real_payload(*, allow_fallback: bool, trace_id: str) -> dict[str, Any]:
    return {
        "caseId": "CASE-P10-AI-SEC",
        "plane": "sagittal",
        "modelKey": "sagittal_spider",
        "inputPath": "fixture-or-internal-path-not-public.npy",
        "metadata": {
            "inferenceMode": "real_baseline",
            "allowContractFallback": allow_fallback,
            "traceId": trace_id,
        },
    }


def test_readiness_and_model_surfaces_do_not_expose_internal_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_ROOT", str(tmp_path / "private-root"))
    monkeypatch.setenv("PFI_MODEL_DIR", str(tmp_path / "private-models"))
    client = TestClient(app, raise_server_exceptions=False)

    for route in ["/health", "/warmup", "/models", "/models/verify", "/models/runtime"]:
        response = client.get(route, headers={TRACE_ID_HEADER: "trace-p10-ai-readiness"})
        assert response.status_code == 200, response.text
        assert response.headers[TRACE_ID_HEADER] == "trace-p10-ai-readiness"
        assert_no_internal_detail(response.json())


def test_models_sync_does_not_expose_internal_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_MODEL_DIR", str(tmp_path / "private-models"))
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "private-outputs"))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/models/sync", headers={TRACE_ID_HEADER: "trace-p10-ai-sync"})

    assert response.status_code == 200, response.text
    assert response.headers[TRACE_ID_HEADER] == "trace-p10-ai-sync"
    body = response.json()
    assert body["status"] == "models_sync_completed"
    assert_no_internal_detail(body)


def test_http_errors_are_sanitized_and_preserve_trace_id(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/agent/worklist", headers={TRACE_ID_HEADER: "trace-p10-ai-error"})

    assert response.status_code == 404
    assert response.headers[TRACE_ID_HEADER] == "trace-p10-ai-error"
    body = response.json()
    assert body["traceId"] == "trace-p10-ai-error"
    assert body["message"] == "worklist no disponible"
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True
    assert_no_internal_detail(body)


def test_real_inference_failure_with_fallback_is_explicit_contract_and_sanitized(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(pipeline_module, "model_status", lambda *_: ready_artifact_status())
    monkeypatch.setattr(pipeline_module, "run_real_inference", failing_real_inference)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/pipeline/run",
        headers={TRACE_ID_HEADER: "trace-p10-ai-fallback"},
        json=real_payload(allow_fallback=True, trace_id="trace-p10-ai-fallback"),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["traceId"] == "trace-p10-ai-fallback"
    assert body["aiOutput"]["requestedInferenceMode"] == "real_baseline"
    assert body["aiOutput"]["inferenceMode"] == "contract"
    flags = body["aiOutput"]["agentDecision"]["flags"]
    assert "contract_fallback_after_real_inference_failure" in flags
    assert body["metadata"]["realInferenceFailure"]["type"] == "RuntimeError"
    assert body["metadata"]["realInferenceFailure"]["message"] == "real_inference_failed"
    assert_no_internal_detail(body)


def test_real_inference_failure_without_fallback_is_controlled_error(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(pipeline_module, "model_status", lambda *_: ready_artifact_status())
    monkeypatch.setattr(pipeline_module, "run_real_inference", failing_real_inference)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/pipeline/run",
        headers={TRACE_ID_HEADER: "trace-p10-ai-no-fallback"},
        json=real_payload(allow_fallback=False, trace_id="trace-p10-ai-no-fallback"),
    )

    assert response.status_code == 500
    body = response.json()
    assert body["traceId"] == "trace-p10-ai-no-fallback"
    assert body["status"] == "error"
    assert body["code"] == "AI_MODULE_ERROR"
    assert body["message"] == "Fallo interno controlado del AI Module"
    assert_no_internal_detail(body)


def test_unavailable_real_model_without_fallback_is_not_reported_as_success() -> None:
    client = TestClient(app)

    response = client.post(
        "/pipeline/run",
        headers={TRACE_ID_HEADER: "trace-p10-ai-model-missing"},
        json={
            "caseId": "CASE-P10-AI-MISSING",
            "plane": "sagittal",
            "modelKey": "unknown_model",
            "inputPath": "demo/CASE-P10-AI-MISSING",
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-p10-ai-model-missing",
            },
        },
    )

    assert response.status_code == 409
    body = response.json()
    assert body["traceId"] == "trace-p10-ai-model-missing"
    assert body["status"] == "error"
    assert "real_baseline" in body["message"]
    assert_no_internal_detail(body)

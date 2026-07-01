from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
AI_SERVICE_ROOT = REPO_ROOT / "ai_service"
sys.path.insert(0, str(AI_SERVICE_ROOT))

from pfi_ai_service.api import app  # noqa: E402


def assert_ok(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert_ok(health.status_code == 200, f"/health status={health.status_code}")
    assert_ok(health.json()["human_review_required"] is True, "/health missing human review flag")

    models = client.get("/models")
    assert_ok(models.status_code == 200, f"/models status={models.status_code}")
    assert_ok("models" in models.json(), "/models missing models key")

    payload = {
        "caseId": "case-001",
        "plane": "sagittal",
        "modelKey": "sagittal_spider",
        "inputPath": "studies/case-001",
        "metadata": {"source": "smoke-api-contract"},
    }
    pipeline = client.post("/pipeline/run", json=payload)
    assert_ok(pipeline.status_code == 200, f"/pipeline/run status={pipeline.status_code}: {pipeline.text}")
    body = pipeline.json()
    assert_ok(body["case_id"] == "case-001", "snake_case case_id missing")
    assert_ok(body["caseId"] == "case-001", "camelCase caseId missing")
    assert_ok(body["model_key"] == "sagittal_spider", "snake_case model_key missing")
    assert_ok(body["modelKey"] == "sagittal_spider", "camelCase modelKey missing")
    assert_ok(body["human_review_required"] is True, "human_review_required must be true")
    assert_ok(body["humanReviewRequired"] is True, "humanReviewRequired must be true")
    assert_ok(body["not_clinical_diagnosis"] is True, "not_clinical_diagnosis must be true")
    assert_ok(body["notClinicalDiagnosis"] is True, "notClinicalDiagnosis must be true")
    assert_ok("measurements" in body, "measurements missing")
    assert_ok("agent_decision" in body, "agent_decision missing")
    assert_ok("agentDecision" in body, "agentDecision missing")

    regression = client.get("/agent/regression-test")
    assert_ok(regression.status_code == 200, f"/agent/regression-test status={regression.status_code}")
    assert_ok(regression.json()["human_review_required"] is True, "regression missing human review flag")

    print("AI Module contract smoke test passed.")


if __name__ == "__main__":
    main()

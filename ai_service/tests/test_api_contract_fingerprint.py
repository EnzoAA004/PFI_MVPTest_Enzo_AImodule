from __future__ import annotations

from pfi_ai_service.api import contract_summary, health, warmup


def test_contract_summary_contains_stable_fingerprint() -> None:
    summary = contract_summary()

    assert summary["schemaVersion"] == "visual-review-contract-v1"
    assert summary["status"] == "stable"
    assert summary["generatedBy"] == "pfi-ai-module.contract_schema"
    assert summary["humanReviewRequired"] is True
    assert summary["notClinicalDiagnosis"] is True
    assert isinstance(summary["schemaHash"], str)
    assert len(summary["schemaHash"]) == 64


def test_health_exposes_contract_fingerprint() -> None:
    response = health()
    contract = response["contract"]

    assert contract["schemaVersion"] == "visual-review-contract-v1"
    assert contract["schemaHash"] == contract_summary()["schemaHash"]
    assert contract["humanReviewRequired"] is True
    assert contract["notClinicalDiagnosis"] is True


def test_warmup_exposes_contract_fingerprint() -> None:
    response = warmup()
    contract = response["contract"]

    assert contract["schemaVersion"] == "visual-review-contract-v1"
    assert contract["schemaHash"] == contract_summary()["schemaHash"]
    assert contract["humanReviewRequired"] is True
    assert contract["notClinicalDiagnosis"] is True

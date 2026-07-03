from __future__ import annotations

from pfi_ai_service.evaluation_contract import evaluation_contract


def test_eval_contract_is_ready() -> None:
    result = evaluation_contract()

    assert result["status"] == "evaluation_contract_ready"
    assert result["schemaVersion"] == "evaluation-contract-v1"
    assert len(result["metrics"]) >= 3
    assert result["humanReviewRequired"] is True
    assert result["notClinicalDiagnosis"] is True

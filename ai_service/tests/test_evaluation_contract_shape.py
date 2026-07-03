from __future__ import annotations

from pfi_ai_service.evaluation_contract import evaluation_contract


def test_evaluation_contract_has_metrics_and_evidence() -> None:
    result = evaluation_contract()
    metric_keys = {metric["key"] for metric in result["metrics"]}

    assert result["status"] == "evaluation_contract_ready"
    assert result["schemaVersion"] == "evaluation-contract-v1"
    assert "dice" in metric_keys
    assert "iou" in metric_keys
    assert "pipeline_schema_hash" in result["requiredEvidence"]
    assert result["humanReviewRequired"] is True

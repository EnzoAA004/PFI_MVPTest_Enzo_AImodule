from __future__ import annotations

import pytest

from pfi_ai_service.pipeline import PipelineRunRequest, run_pipeline


def ready_artifact() -> dict:
    return {
        "availableForRealInference": True,
        "baselineReady": True,
        "version": "sagittal-spider-final-v1",
        "artifactHash": "abc123",
        "readiness": "real_baseline_ready",
    }


def test_fallback_false_propagates_real_inference_error(monkeypatch) -> None:
    from pfi_ai_service import pipeline

    monkeypatch.setattr(pipeline, "model_status", lambda *_args, **_kwargs: ready_artifact())

    def fail(*_args, **_kwargs):
        raise RuntimeError("strict failure")

    monkeypatch.setattr(pipeline, "run_real_inference", fail)

    with pytest.raises(RuntimeError, match="strict failure"):
        run_pipeline(PipelineRunRequest(
            caseId="CASE-STRICT",
            plane="sagittal",
            modelKey="sagittal_spider",
            inputPath="input.npy",
            metadata={"inferenceMode": "real_baseline", "allowContractFallback": False},
        ))


def test_real_baseline_response_exposes_artifact_hash(monkeypatch) -> None:
    from pfi_ai_service import pipeline

    monkeypatch.setattr(pipeline, "model_status", lambda *_args, **_kwargs: ready_artifact())

    def response(request, run_id):
        return {
            "runId": run_id,
            "modelKey": request.model_key,
            "modelVersion": "sagittal-spider-final-v1",
            "artifactHash": "abc123",
            "inferenceMode": "real_baseline",
            "requestedInferenceMode": request.metadata["inferenceMode"],
            "allowContractFallback": request.metadata["allowContractFallback"],
            "metadata": {
                "artifactHash": "abc123",
                "inferenceMode": "real_baseline",
                "selectedAxis": 2,
                "sliceCount": 17,
            },
        }

    monkeypatch.setattr(pipeline, "run_real_inference", response)
    monkeypatch.setattr(pipeline, "write_json", lambda *_args, **_kwargs: None)

    result = run_pipeline(PipelineRunRequest(
        caseId="CASE-STRICT",
        plane="sagittal",
        modelKey="sagittal_spider",
        inputPath="input.npy",
        metadata={"inferenceMode": "real_baseline", "allowContractFallback": False},
    ))

    assert result["inferenceMode"] == "real_baseline"
    assert result["artifactHash"] == "abc123"
    assert result["metadata"]["artifactHash"] == "abc123"

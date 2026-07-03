from __future__ import annotations

from typing import Any, Dict

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .contract_schema import pipeline_contract_schema
from .model_artifacts import artifact_summary


def evaluation_contract() -> Dict[str, Any]:
    schema = pipeline_contract_schema()
    artifacts = artifact_summary()
    return {
        "status": "evaluation_contract_ready",
        "schemaVersion": "evaluation-contract-v1",
        "pipelineSchemaVersion": schema.get("schemaVersion"),
        "pipelineSchemaHash": schema.get("schemaHash"),
        "defaultInferenceMode": artifacts.get("defaultInferenceMode"),
        "realInferenceReady": artifacts.get("readyForRealInference"),
        "metrics": [
            metric("dice", "Dice Similarity Coefficient", "segmentation", "higher_is_better"),
            metric("iou", "Intersection over Union", "segmentation", "higher_is_better"),
            metric("hausdorff95", "Hausdorff distance percentile 95", "boundary", "lower_is_better"),
            metric("absolute_measurement_error_mm", "Absolute measurement error in millimeters", "measurement", "lower_is_better"),
            metric("review_agreement", "Professional review agreement", "qualitative", "higher_is_better"),
        ],
        "requiredEvidence": [
            "dataset_split_description",
            "model_artifact_hash",
            "pipeline_schema_hash",
            "metric_summary_json",
            "professional_review_feedback",
        ],
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def metric(key: str, label: str, category: str, direction: str) -> Dict[str, Any]:
    return {"key": key, "label": label, "category": category, "direction": direction, "required": True}

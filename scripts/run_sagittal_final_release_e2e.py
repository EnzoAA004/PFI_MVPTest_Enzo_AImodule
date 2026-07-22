from __future__ import annotations

import json
import os
from pathlib import Path

from pfi_ai_service.model_artifacts import model_status
from pfi_ai_service.model_materializer import sync_model_artifacts
from pfi_ai_service.pipeline import PipelineRunRequest, run_pipeline
from pfi_ai_service.real_inference_runtime import clear_model_cache
from pfi_ai_service.settings import MODEL_REGISTRY, get_settings


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(f"Variable requerida ausente: {name}")
    return value.strip()


def output_exists(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and Path(value).exists()


def main() -> None:
    if os.getenv("RUN_LIVE_GCS_E2E") != "1":
        print(json.dumps({
            "executionStatus": "skipped",
            "reason": "RUN_LIVE_GCS_E2E debe ser 1",
            "success": False,
        }, indent=2))
        return

    input_path = Path(require_env("PFI_E2E_INPUT_PATH"))
    if not input_path.exists():
        raise FileNotFoundError(f"PFI_E2E_INPUT_PATH no existe: {input_path}")

    sync_result = sync_model_artifacts(force=os.getenv("PFI_E2E_FORCE_SYNC") == "1")
    sagittal_sync = next((item for item in sync_result.get("items", []) if item.get("modelKey") == "sagittal_spider"), {})

    clear_model_cache()
    metadata = {
        "inferenceMode": "real_baseline",
        "allowContractFallback": False,
        "traceId": os.getenv("PFI_E2E_TRACE_ID", "live-gcs-e2e"),
    }
    if os.getenv("PFI_E2E_SLICE_INDEX"):
        metadata["sliceIndex"] = int(os.environ["PFI_E2E_SLICE_INDEX"])

    response = run_pipeline(PipelineRunRequest(
        caseId=os.getenv("PFI_E2E_CASE_ID", "CASE-GCS-E2E"),
        plane="sagittal",
        modelKey="sagittal_spider",
        inputPath=str(input_path),
        metadata=metadata,
    ))
    artifact = model_status("sagittal_spider", dict(MODEL_REGISTRY["sagittal_spider"]))
    output_files = response.get("metadata", {}).get("outputFiles", {})
    settings = get_settings()
    success = (
        response.get("inferenceMode") == "real_baseline"
        and response.get("metadata", {}).get("inputOrientationTransform") in {"none", "move_axis_0_to_last"}
        and artifact.get("artifactHash") == settings.sagittal_model_sha256
        and output_exists(output_files.get("maskPath"))
        and output_exists(output_files.get("confidencePath"))
        and output_exists(output_files.get("overlayPath"))
    )
    print(json.dumps({
        "executionStatus": "completed" if success else "failed",
        "releaseStatus": sagittal_sync.get("status"),
        "releaseId": sagittal_sync.get("releaseId"),
        "releaseContentSha256": sagittal_sync.get("releaseContentSha256"),
        "releaseManifestSha256": sagittal_sync.get("releaseManifestSha256"),
        "modelSha256": artifact.get("artifactHash"),
        "modelVersion": response.get("modelVersion"),
        "inputShapeNative": response.get("metadata", {}).get("inputShapeNative"),
        "inputShapeCanonical": response.get("metadata", {}).get("inputShapeCanonical"),
        "inputOrientationTransform": response.get("metadata", {}).get("inputOrientationTransform"),
        "selectedSlice": response.get("metadata", {}).get("selectedSlice"),
        "selectedAxis": response.get("metadata", {}).get("selectedAxis"),
        "sliceCount": response.get("metadata", {}).get("sliceCount"),
        "inferenceMode": response.get("inferenceMode"),
        "fallbackUsed": response.get("inferenceMode") != "real_baseline",
        "maskPathExists": output_exists(output_files.get("maskPath")),
        "confidencePathExists": output_exists(output_files.get("confidencePath")),
        "overlayPathExists": output_exists(output_files.get("overlayPath")),
        "humanReviewRequired": response.get("humanReviewRequired"),
        "notClinicalDiagnosis": response.get("notClinicalDiagnosis"),
        "success": success,
    }, indent=2))


if __name__ == "__main__":
    main()

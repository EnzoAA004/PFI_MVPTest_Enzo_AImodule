from __future__ import annotations

from fastapi import FastAPI

from .model_artifacts import registry_with_artifact_status
from .real_inference_runtime import clear_model_cache, runtime_status
from .security import sanitize_public_payload
from .subarticular_runtime_service import get_subarticular_runtime_status


def register_real_inference_routes(app: FastAPI) -> None:
    @app.get("/models/runtime")
    def model_runtime_status():
        status = runtime_status()
        status["segmentationModels"] = registry_with_artifact_status()
        status["degenerativeFindingModels"] = {
            "subarticular": get_subarticular_runtime_status(),
        }
        return sanitize_public_payload(status)

    @app.post("/models/cache/clear")
    def clear_runtime_model_cache():
        clear_model_cache()
        status = runtime_status()
        status["segmentationModels"] = registry_with_artifact_status()
        status["degenerativeFindingModels"] = {
            "subarticular": get_subarticular_runtime_status(),
        }
        return sanitize_public_payload(status)

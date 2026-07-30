from __future__ import annotations

from fastapi import FastAPI

from .real_inference_runtime import clear_model_cache, runtime_status
from .security import sanitize_public_payload


def register_real_inference_routes(app: FastAPI) -> None:
    @app.get("/models/runtime")
    def model_runtime_status():
        return sanitize_public_payload(runtime_status())

    @app.post("/models/cache/clear")
    def clear_runtime_model_cache():
        clear_model_cache()
        return sanitize_public_payload(runtime_status())

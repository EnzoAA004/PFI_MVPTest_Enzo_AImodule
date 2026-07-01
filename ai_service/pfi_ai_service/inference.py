from __future__ import annotations

from typing import Any, Dict

from .pipeline import PipelineRunRequest, run_pipeline


def run_sagittal_inference(request: PipelineRunRequest) -> Dict[str, Any]:
    return run_pipeline(request.model_copy(update={"plane": "sagittal"}))


def run_axial_inference(request: PipelineRunRequest) -> Dict[str, Any]:
    return run_pipeline(request.model_copy(update={"plane": "axial"}))

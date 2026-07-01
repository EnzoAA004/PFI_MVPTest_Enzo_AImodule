from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from .settings import get_settings, MODEL_REGISTRY
from .agent import build_agent_decisions, summarize_agent_decisions
from .agent_policy import regression_test_report
from .inference import run_axial_inference, run_sagittal_inference
from .pipeline import PipelineRunRequest, run_pipeline
from .reporting import build_markdown_summary

app = FastAPI(title="PFI AI Service", version="0.1.0")


def clean_for_json(value: Any) -> Any:
    """Convierte objetos pandas/numpy/NaN a JSON estricto.

    FastAPI/Starlette puede fallar si recibe NaN o tipos numpy dentro de
    diccionarios generados desde DataFrames. Este helper deja las respuestas
    listas para backend/frontend.
    """
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}

    if isinstance(value, list):
        return [clean_for_json(v) for v in value]

    if isinstance(value, tuple):
        return [clean_for_json(v) for v in value]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    # Tipos numpy/pandas escalares.
    if hasattr(value, "item"):
        try:
            return clean_for_json(value.item())
        except Exception:
            pass

    return value


@app.get("/health")
def health():
    settings = get_settings()
    return clean_for_json({
        "status": "ok",
        "pfi_root": str(settings.pfi_root),
        "human_review_required": True,
    })


@app.get("/models")
def models():
    settings = get_settings()
    return clean_for_json({
        "models": MODEL_REGISTRY,
        "paths": {
            "sagittal_model_path": str(settings.sagittal_model_path),
            "axial_model_path": str(settings.axial_model_path),
        },
    })


@app.post("/inference/sagittal")
def inference_sagittal(request: PipelineRunRequest):
    if request.plane != "sagittal":
        request = request.model_copy(update={"plane": "sagittal"})
    return clean_for_json(run_sagittal_inference(request))


@app.post("/inference/axial")
def inference_axial(request: PipelineRunRequest):
    if request.plane != "axial":
        request = request.model_copy(update={"plane": "axial"})
    return clean_for_json(run_axial_inference(request))


@app.post("/pipeline/run")
def pipeline_run(request: PipelineRunRequest):
    return clean_for_json(run_pipeline(request))


@app.get("/agent/worklist")
def agent_worklist():
    settings = get_settings()
    worklist_path = settings.e14_results_root / "E14_agent_worklist.csv"
    if not worklist_path.exists():
        raise HTTPException(status_code=404, detail=f"No existe {worklist_path}")

    df = pd.read_csv(worklist_path)
    return clean_for_json({
        "rows": int(len(df)),
        "items": df.to_dict(orient="records"),
    })


@app.get("/agent/report")
def agent_report():
    settings = get_settings()
    worklist_path = settings.e14_results_root / "E14_agent_worklist.csv"
    metrics_path = settings.e14_results_root / "E14_agent_metrics_summary.csv"

    if not worklist_path.exists():
        raise HTTPException(status_code=404, detail=f"No existe {worklist_path}")

    worklist = pd.read_csv(worklist_path)
    decisions = build_agent_decisions(worklist)

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if "agent_item_id" in metrics.columns:
            decisions = decisions.merge(metrics, on=["agent_item_id", "plane", "case_ref"], how="left")

    summary = summarize_agent_decisions(decisions)

    return clean_for_json({
        "summary": summary,
        "markdown": build_markdown_summary(summary),
        "items": decisions.to_dict(orient="records"),
    })


@app.get("/agent/report/{run_id}")
def agent_report_by_run(run_id: str):
    settings = get_settings()
    report_path = settings.output_dir / "agent_reports" / f"{run_id}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"No existe reporte para run_id={run_id}")

    return clean_for_json(json.loads(report_path.read_text(encoding="utf-8")))


@app.get("/agent/regression-test")
def agent_regression_test():
    return clean_for_json(regression_test_report())

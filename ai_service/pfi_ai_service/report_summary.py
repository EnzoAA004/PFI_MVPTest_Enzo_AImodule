from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def summarize_agent_report(report: Dict[str, Any]) -> Dict[str, Any]:
    ai_output = as_dict(report.get("aiOutput"))
    quality = as_dict(report.get("quality"))
    metadata = as_dict(report.get("metadata"))
    model_artifact = as_dict(report.get("modelArtifact"))
    artifact = as_dict(model_artifact.get("artifact"))
    measurements = report.get("measurementValues") or as_dict(report.get("measurements")).get("values") or []
    masks = report.get("masks") or []
    landmarks = report.get("landmarks") or []

    trace_id = report.get("traceId") or metadata.get("traceId") or metadata.get("correlationId")
    return {
        "runId": report.get("runId") or report.get("run_id"),
        "traceId": trace_id,
        "caseId": report.get("caseId") or report.get("case_id"),
        "studyId": report.get("studyId"),
        "patientId": report.get("patientId"),
        "studyDate": report.get("studyDate"),
        "plane": report.get("plane"),
        "modelKey": report.get("modelKey") or report.get("model_key"),
        "modelVersion": report.get("modelVersion"),
        "reviewStatus": report.get("reviewStatus"),
        "inferenceMode": ai_output.get("inferenceMode") or metadata.get("inferenceMode"),
        "requestedInferenceMode": ai_output.get("requestedInferenceMode") or metadata.get("requestedInferenceMode"),
        "modelReadiness": ai_output.get("modelReadiness") or metadata.get("modelReadiness"),
        "realInferenceAvailable": ai_output.get("realInferenceAvailable"),
        "modelArtifactExists": artifact.get("exists"),
        "modelArtifactHash": model_artifact.get("artifactHash") or artifact.get("sha256"),
        "artifactIntegrityStatus": model_artifact.get("artifactIntegrityStatus") or artifact.get("integrityStatus"),
        "maskCount": quality.get("maskCount", len(masks) if isinstance(masks, list) else 0),
        "landmarkCount": quality.get("landmarkCount", len(landmarks) if isinstance(landmarks, list) else 0),
        "measurementCount": quality.get("measurementCount", len(measurements) if isinstance(measurements, list) else 0),
        "measurementsDerivedFromContours": quality.get("measurementsDerivedFromContours"),
        "humanReviewRequired": bool(report.get("humanReviewRequired", True)),
        "notClinicalDiagnosis": bool(report.get("notClinicalDiagnosis", True)),
        "diagnosisGenerated": bool(metadata.get("diagnosisGenerated", False)),
        "deidentified": bool(metadata.get("deidentified", True)),
        "source": metadata.get("source"),
    }


def recent_agent_report_summaries(reports_dir: Path, limit: int = 20) -> Dict[str, Any]:
    safe_limit = max(1, min(int(limit or 20), 100))
    if not reports_dir.exists():
        return {
            "status": "ok",
            "count": 0,
            "limit": safe_limit,
            "items": [],
            "humanReviewRequired": True,
            "notClinicalDiagnosis": True,
        }

    report_files = sorted(
        [path for path in reports_dir.glob("*.json") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:safe_limit]
    items = []
    skipped = 0
    for path in report_files:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            summary = summarize_agent_report(report)
            summary["reportFile"] = path.name
            summary["reportModifiedAt"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            items.append(summary)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            skipped += 1

    return {
        "status": "ok",
        "count": len(items),
        "skipped": skipped,
        "limit": safe_limit,
        "items": items,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }


def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}

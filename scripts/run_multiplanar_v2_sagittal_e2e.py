from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

SAGITTAL_HASH = "cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944"
FORBIDDEN_KEYS = {
    "run_id",
    "case_id",
    "model_key",
    "input_path",
    "inputPath",
    "overlay_path",
    "overlayPath",
    "agent_decision",
    "human_review_required",
    "not_clinical_diagnosis",
    "measurementValues",
}
FORBIDDEN_TEXT = ("PAT-DEMO", "CASE-DEMO", "2026-07-01", "Sagittal T1", "Axial T1", "Axial T2 L4-L5", "/app/", "/home/", "C:\\", "host.docker.internal", "localhost", "trycloudflare.com")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke E2E local para POST /v2/multiplanar/run sagital-only.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input", default="101_t2.mha")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input no encontrado: {input_path}")

    case_id = "P9A-SPIDER-101-T2"
    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        with input_path.open("rb") as handle:
            upload = client.post(
                "/inputs",
                data={"caseId": case_id, "plane": "sagittal"},
                files={"file": (input_path.name, handle, "application/octet-stream")},
            )
        assert_status(upload, 200)
        input_id = upload.json()["inputId"]
        request = {
            "caseId": case_id,
            "traceId": "trace-p9a-spider-101-t2",
            "inferenceMode": "real_baseline",
            "allowContractFallback": False,
            "planes": {"sagittal": {"inputId": input_id, "modelKey": "sagittal_spider"}, "axial": None},
            "options": {"sliceIndex": None, "sliceAxis": None, "sliceWindowRadius": 3, "inputOrientationTransform": None},
        }
        response = client.post("/v2/multiplanar/run", json=request)
    assert_status(response, 200)
    body = response.json()
    validate_response(body)
    print(json.dumps({
        "status": "ok",
        "runId": body["runId"],
        "traceId": body["traceId"],
        "selectedSliceIndex": body["planes"]["sagittal"]["input"]["selectedSliceIndex"],
        "sliceCount": body["planes"]["sagittal"]["input"]["sliceCount"],
    }, indent=2))
    return 0


def assert_status(response: httpx.Response, expected: int) -> None:
    if response.status_code != expected:
        raise AssertionError(f"HTTP {response.status_code}, esperado {expected}: {response.text[:1000]}")


def validate_response(body: dict[str, Any]) -> None:
    assert body["schemaVersion"] == "pfi.multiplanar-run.v2"
    assert body["workspaceMode"] == "sagittal_only"
    assert body["effectiveInferenceMode"] == "real_baseline"
    assert body["planes"]["axial"] is None
    sagittal = body["planes"]["sagittal"]
    assert sagittal["status"] == "ready"
    assert sagittal["model"]["artifactHash"] == SAGITTAL_HASH
    assert sagittal["input"]["nativeShape"] == [352, 384, 17]
    assert sagittal["input"]["canonicalShape"] == [352, 384, 17]
    selected = sagittal["input"]["selectedSliceIndex"]
    slice_count = sagittal["input"]["sliceCount"]
    assert isinstance(selected, int)
    assert slice_count == 17
    assert 0 <= selected < slice_count
    assert len(sagittal["assets"]) >= 4
    assert len(sagittal["masks"]) == 3
    assert len(sagittal["landmarks"]) == 3
    assert len(sagittal["measurements"]) == 9
    assert body["governance"]["humanReviewRequired"] is True
    assert body["governance"]["notClinicalDiagnosis"] is True
    assert body["governance"]["diagnosisGenerated"] is False
    assert body["requestedInferenceMode"] == "real_baseline"
    assert "fallbackReason" not in json.dumps(body)
    assert not (walk_keys(body) & FORBIDDEN_KEYS)
    serialized = json.dumps(body)
    for text in FORBIDDEN_TEXT:
        assert text not in serialized


def walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys |= walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= walk_keys(item)
    return keys


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import pandas as pd

EXPECTED_CHECKPOINT_SHA256 = "d41262d57b13c146a48ab15f5e183cc6a55fc92724b7d0c286cea1f2ce26e84a"
SCHEMA_VERSION = "pfi.degenerative-findings.v1"
SEVERITIES = ("normal_mild", "moderate", "severe")
LEVELS = {"L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"}
FORBIDDEN_RESPONSE_TOKENS = (
    "PatientID",
    "PatientName",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
    "C:\\",
    "/models/",
    "/tmp/",
    "/app/",
    "/outputs/",
)


class E2EFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_path_env(name: str) -> Path:
    value = os.getenv(name)
    if not value or not value.strip():
        raise E2EFailure(f"{name} is required")
    path = Path(value)
    if not path.exists():
        raise E2EFailure(f"{name} does not exist")
    return path


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def validate_environment() -> tuple[Path, Path, Path]:
    checkpoint = require_path_env("PFI_SUBARTICULAR_CHECKPOINT_PATH")
    manifest = require_path_env("PFI_E2E_MANIFEST_PATH")
    images_root = require_path_env("PFI_E2E_TRAIN_IMAGES_ROOT")
    if "internal_test" in manifest.name.lower() or "internal_test" in str(manifest).lower():
        raise E2EFailure("internal_test manifest is not allowed")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != EXPECTED_CHECKPOINT_SHA256:
        raise E2EFailure("checkpoint SHA-256 mismatch")
    return checkpoint, manifest, images_root


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def manifest_candidates(manifest: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest, dtype=str)
    required = {
        "study_id",
        "split",
        "coordinate_series_id",
        "coordinate_instance_number",
        "coordinate_x",
        "coordinate_y",
        "side",
        "level",
        "sequence_category",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise E2EFailure(f"manifest missing required columns: {','.join(missing)}")
    mask = (
        df["split"].isin(["train", "validation"])
        & (df["sequence_category"] == "axial_t2")
        & df["side"].isin(["left", "right"])
        & df["level"].isin(LEVELS)
    )
    rows: list[dict[str, Any]] = []
    for _, row in df[mask].iterrows():
        x = finite_float(row["coordinate_x"])
        y = finite_float(row["coordinate_y"])
        instance = finite_float(row["coordinate_instance_number"])
        if x is None or y is None or instance is None or instance < 0:
            continue
        rows.append({
            "study_id": str(row["study_id"]),
            "series_id": str(row["coordinate_series_id"]),
            "split": str(row["split"]),
            "instance_number": int(round(instance)),
            "x": x,
            "y": y,
            "side": str(row["side"]),
            "level": str(row["level"]),
        })
    return pd.DataFrame(rows)


def zip_series_index(images_root: Path) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for zip_path in sorted(images_root.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            grouped: dict[tuple[str, str], list[str]] = {}
            for name in members:
                parts = PurePosixPath(name).parts
                if len(parts) < 3:
                    continue
                grouped.setdefault((parts[0], parts[1]), []).append(name)
            for (study_id, series_id), names in grouped.items():
                index.append({
                    "zip_path": zip_path,
                    "study_id": study_id,
                    "series_id": series_id,
                    "members": sorted(names),
                })
    return index


def directory_series_index(images_root: Path) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for study_dir in images_root.iterdir() if images_root.is_dir() else ():
        if not study_dir.is_dir():
            continue
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue
            files = sorted(path for path in series_dir.rglob("*") if path.is_file())
            if files:
                index.append({
                    "study_id": study_dir.name,
                    "series_id": series_dir.name,
                    "files": files,
                })
    return index


def select_sample(manifest: Path, images_root: Path) -> dict[str, Any]:
    candidates = manifest_candidates(manifest)
    if candidates.empty:
        raise E2EFailure("no valid manifest candidates")
    zip_index = zip_series_index(images_root)
    dir_index = directory_series_index(images_root)
    by_pair = {(item["study_id"], item["series_id"]): item for item in [*zip_index, *dir_index]}
    for row in candidates.to_dict(orient="records"):
        key = (row["study_id"], row["series_id"])
        source = by_pair.get(key)
        if source:
            return {**row, "source": source, "sample_hash": short_hash(f"{key[0]}:{key[1]}")}
    raise E2EFailure("no manifest candidate has matching DICOM source")


def write_series_zip(sample: dict[str, Any], destination: Path) -> int:
    source = sample["source"]
    count = 0
    if "zip_path" in source:
        with zipfile.ZipFile(source["zip_path"]) as src, zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as dst:
            for name in source["members"]:
                safe_name = PurePosixPath(name).name
                dst.writestr(f"study/axial_t2/{safe_name}", src.read(name))
                count += 1
    else:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as dst:
            for path in source["files"]:
                dst.write(path, f"study/axial_t2/{path.name}")
                count += 1
    if count == 0:
        raise E2EFailure("selected series has no files")
    return count


def assert_no_internal_details(payload: Any) -> None:
    text = json.dumps(payload, sort_keys=True)
    for token in FORBIDDEN_RESPONSE_TOKENS:
        if token in text:
            raise E2EFailure(f"response exposed forbidden token: {token}")


def wait_for_health(base_url: str, timeout_seconds: int = 60) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = type(exc).__name__
        time.sleep(1.0)
    raise E2EFailure(f"health did not become ready: {last_error}")


def validate_prediction(body: dict[str, Any]) -> dict[str, Any]:
    assert_no_internal_details(body)
    findings = body.get("degenerativeFindings")
    if not isinstance(findings, dict) or findings.get("schemaVersion") != SCHEMA_VERSION:
        raise E2EFailure("invalid degenerativeFindings schema")
    items = findings.get("findings")
    if not isinstance(items, list) or len(items) != 1:
        raise E2EFailure("expected exactly one finding")
    finding = items[0]
    if finding.get("findingType") != "subarticular_stenosis":
        raise E2EFailure("unexpected findingType")
    probabilities = finding.get("classification", {}).get("probabilities", {})
    if sorted(probabilities) != sorted(SEVERITIES):
        raise E2EFailure("unexpected probabilities keys")
    values = [float(probabilities[key]) for key in SEVERITIES]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise E2EFailure("probabilities are not finite probabilities")
    if abs(sum(values) - 1.0) > 1e-6:
        raise E2EFailure("probabilities do not sum to 1")
    label = finding.get("classification", {}).get("label")
    if label != max(probabilities, key=probabilities.get):
        raise E2EFailure("argmax does not match label")
    if finding.get("review") != {"required": True, "status": "pending"}:
        raise E2EFailure("invalid review block")
    if finding.get("notClinicalDiagnosis") is not True:
        raise E2EFailure("notClinicalDiagnosis must be true")
    if body.get("humanReviewRequired") is not True or body.get("notClinicalDiagnosis") is not True:
        raise E2EFailure("top-level governance flags invalid")
    if body.get("autonomousDiagnosis") is not False:
        raise E2EFailure("autonomousDiagnosis must be false")
    if finding.get("localization") != {"source": "external_coordinate", "researchOnly": True}:
        raise E2EFailure("invalid localization block")
    return finding


def post_predict(base_url: str, payload: dict[str, Any]) -> httpx.Response:
    return httpx.post(f"{base_url}/degenerative-findings/subarticular/predict", json=payload, timeout=60.0)


def run_negative_checks(base_url: str, valid_payload: dict[str, Any]) -> dict[str, int]:
    checks = {
        "missingInputId": ({**valid_payload, "inputId": "inp_missing"}, {404}),
        "invalidSide": ({**valid_payload, "side": "bilateral"}, {400, 422}),
        "invalidLevel": ({**valid_payload, "level": "T12-L1"}, {400, 422}),
        "outOfRangeCoordinate": ({**valid_payload, "x": 999999.0}, {400, 422}),
        "pathFieldForbidden": ({**valid_payload, "path": "C:/secret/input.dcm"}, {422}),
        "checkpointPathFieldForbidden": ({**valid_payload, "checkpointPath": "/models/frozen_subarticular_checkpoint.pt"}, {422}),
        "dicomIdentifierForbidden": ({**valid_payload, "PatientID": "forbidden"}, {422}),
    }
    observed: dict[str, int] = {}
    for name, (payload, expected) in checks.items():
        response = post_predict(base_url, payload)
        observed[name] = response.status_code
        if response.status_code not in expected:
            raise E2EFailure(f"negative check {name} returned HTTP {response.status_code}")
        try:
            assert_no_internal_details(response.json())
        except json.JSONDecodeError as exc:
            raise E2EFailure(f"negative check {name} did not return JSON") from exc
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("PFI_E2E_AI_SERVICE_URL", "http://127.0.0.1:18000"))
    args = parser.parse_args()

    _checkpoint, manifest, images_root = validate_environment()
    sample = select_sample(manifest, images_root)
    case_id = f"CASE-E2E-{sample['sample_hash']}"

    health = wait_for_health(args.base_url)
    with tempfile.TemporaryDirectory(prefix="pfi-subarticular-e2e-") as tmp:
        zip_path = Path(tmp) / "study.zip"
        file_count = write_series_zip(sample, zip_path)
        with zip_path.open("rb") as handle:
            ingest = httpx.post(
                f"{args.base_url}/inputs/study",
                files={"file": ("study.zip", handle, "application/zip")},
                data={"caseId": case_id},
                timeout=120.0,
            )
        if ingest.status_code != 200:
            raise E2EFailure(f"ingestion failed with HTTP {ingest.status_code}")
        ingest_body = ingest.json()
        assert_no_internal_details(ingest_body)
        axial = ingest_body.get("axial")
        if not isinstance(axial, dict) or axial.get("plane") != "axial":
            raise E2EFailure("ingestion did not return an axial input")
        input_id = str(axial.get("inputId") or "")
        if not input_id.startswith("inp_"):
            raise E2EFailure("invalid axial inputId")

        predict_payload = {
            "inputId": input_id,
            "instanceNumber": sample["instance_number"],
            "x": sample["x"],
            "y": sample["y"],
            "side": sample["side"],
            "level": sample["level"],
        }
        first = post_predict(args.base_url, predict_payload)
        if first.status_code != 200:
            raise E2EFailure(f"predict failed with HTTP {first.status_code}")
        first_body = first.json()
        finding = validate_prediction(first_body)

        second = post_predict(args.base_url, predict_payload)
        if second.status_code != 200:
            raise E2EFailure(f"second predict failed with HTTP {second.status_code}")
        second_body = second.json()
        validate_prediction(second_body)
        if first_body["degenerativeFindings"] != second_body["degenerativeFindings"]:
            raise E2EFailure("second predict was not deterministic")

        runtime_after = httpx.get(f"{args.base_url}/models/runtime", timeout=10.0).json()
        subarticular_after = runtime_after.get("degenerativeFindingModels", {}).get("subarticular", {})
        if subarticular_after.get("loaded") is not True:
            raise E2EFailure("subarticular runtime was not loaded after inference")
        negatives = run_negative_checks(args.base_url, predict_payload)

    output = {
        "status": "SUBARTICULAR_HTTP_REAL_E2E_OK",
        "serviceHealthStatus": health.get("status"),
        "sampleHash": sample["sample_hash"],
        "sampleSplit": sample["split"],
        "seriesFilesUploaded": file_count,
        "inputIdPrefix": input_id[:8],
        "predictRequest": {
            "inputIdPrefix": input_id[:8],
            "instanceNumber": predict_payload["instanceNumber"],
            "side": predict_payload["side"],
            "level": predict_payload["level"],
            "xRounded": round(float(predict_payload["x"]), 2),
            "yRounded": round(float(predict_payload["y"]), 2),
        },
        "prediction": {
            "findingType": finding["findingType"],
            "label": finding["classification"]["label"],
            "probabilitySum": round(sum(finding["classification"]["probabilities"].values()), 8),
            "reviewStatus": finding["review"]["status"],
            "localization": finding["localization"],
        },
        "negativeChecks": negatives,
        "runtimeAfterInference": {
            "status": subarticular_after.get("status"),
            "loaded": subarticular_after.get("loaded"),
            "configured": subarticular_after.get("configured"),
            "artifactPresent": subarticular_after.get("artifactPresent"),
        },
        "humanReviewRequired": first_body["humanReviewRequired"],
        "notClinicalDiagnosis": first_body["notClinicalDiagnosis"],
        "autonomousDiagnosis": first_body["autonomousDiagnosis"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

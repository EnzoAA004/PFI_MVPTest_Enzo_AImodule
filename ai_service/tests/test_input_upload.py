from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.real_inference_runtime import clear_model_cache


FIXTURE = Path("ai_service/tests/fixtures/real_baseline/sagittal_sample_input.npy")


def assert_no_internal_paths(value: object) -> None:
    text = str(value)
    forbidden = [
        "ai_service/tests/fixtures/real_baseline",
        "uploads/",
        "uploads\\",
        "outputs/",
        "outputs\\",
        "models/final",
    ]
    assert not any(item in text for item in forbidden), text[:500]


def upload_fixture(client: TestClient, *, case_id: str, filename: str = "sample.npy") -> dict:
    with FIXTURE.open("rb") as handle:
        response = client.post(
            "/inputs",
            data={"caseId": case_id, "plane": "sagittal"},
            files={"file": (filename, handle, "application/octet-stream")},
        )
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_valid_returns_input_id_without_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    body = upload_fixture(TestClient(app), case_id="CASE-AI014-UPLOAD")

    assert body["inputId"].startswith("inp_")
    assert body["caseId"] == "CASE-AI014-UPLOAD"
    assert body["plane"] == "sagittal"
    assert body["format"] == "npy"
    assert body["size"] == FIXTURE.stat().st_size
    assert_no_internal_paths(body)
    saved_files = list((tmp_path / "uploads").rglob("*.npy"))
    assert len(saved_files) == 1
    assert saved_files[0].name == f"{body['inputId']}.npy"


def test_upload_rejects_invalid_extension(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    response = TestClient(app).post(
        "/inputs",
        data={"caseId": "CASE-AI014-BAD-EXT", "plane": "sagittal"},
        files={"file": ("sample.exe", b"not allowed", "application/octet-stream")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["message"].startswith("extension no permitida")
    assert_no_internal_paths(body)


def test_upload_rejects_size_exceeded(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("PFI_MAX_UPLOAD_BYTES", "8")
    response = TestClient(app).post(
        "/inputs",
        data={"caseId": "CASE-AI014-TOO-LARGE", "plane": "sagittal"},
        files={"file": ("sample.npy", b"0123456789", "application/octet-stream")},
    )

    assert response.status_code == 413
    body = response.json()
    assert body["message"] == "archivo excede el limite de tama?o"
    assert_no_internal_paths(body)
    assert [item for item in (tmp_path / "uploads").rglob("*") if item.is_file()] == []


def test_upload_neutralizes_path_traversal_filename(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    body = upload_fixture(TestClient(app), case_id="CASE-AI014-TRAVERSAL", filename="../../evil.npy")

    saved_files = list((tmp_path / "uploads").rglob("*.npy"))
    assert len(saved_files) == 1
    saved = saved_files[0]
    assert saved.name == f"{body['inputId']}.npy"
    assert saved.resolve().is_relative_to((tmp_path / "uploads").resolve())
    assert not (tmp_path / "evil.npy").exists()
    assert_no_internal_paths(body)


def test_pipeline_real_baseline_with_uploaded_input_id(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_INFERENCE_DEVICE", "cpu")
    clear_model_cache()
    client = TestClient(app)
    case_id = "CASE-AI014-PIPELINE"
    upload = upload_fixture(client, case_id=case_id)

    response = client.post(
        "/pipeline/run",
        json={
            "caseId": case_id,
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "inputId": upload["inputId"],
            "metadata": {
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "traceId": "trace-ai014-upload-pipeline",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["aiOutput"]["inferenceMode"] == "real_baseline"
    assert body["inputId"] == upload["inputId"]
    assert body["traceId"] == "trace-ai014-upload-pipeline"
    assert body["metadata"]["outputFiles"]["imagePath"] == {"generated": True, "fileName": "input.png"}
    assert body["metadata"]["outputFiles"]["maskPath"] == {"generated": True, "fileName": "mask.npy"}
    assert_no_internal_paths(body)

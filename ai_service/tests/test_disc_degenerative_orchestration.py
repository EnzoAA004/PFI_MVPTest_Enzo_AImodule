import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.disc_degenerative_orchestration import (
    DiscFindingsFromSegmentationRequest,
    build_product_request_from_segmentation,
)


def write_report(
    root: Path,
    *,
    run_id: str,
    case_id: str,
    input_id: str,
    slice_offset: int,
) -> None:
    directory = root / "full_series_segmentation" / run_id / "sagittal"
    directory.mkdir(parents=True, exist_ok=True)
    levels = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
    report = {
        "schemaVersion": "pfi.full-series-segmentation.v1",
        "status": "completed",
        "runId": run_id,
        "caseId": case_id,
        "inputId": input_id,
        "plane": "sagittal",
        "coverageComplete": True,
        "discLocalizations": [
            {
                "source": "segmentation_derived_disc_level",
                "level": level,
                "sliceIndex": slice_offset + index,
                "bboxYx": [20 + index * 10, 28 + index * 10, 70, 120],
                "coordinateSpaceWidth": 256,
                "coordinateSpaceHeight": 256,
                "maskId": f"disc-{level}",
                "segmentationRunId": run_id,
                "automaticAnatomicalLocalizationValidated": False,
            }
            for index, level in enumerate(levels)
        ],
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_orchestration_keeps_t1_t2_localization_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "outputs"
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(output))
    write_report(
        output,
        run_id="series-t1",
        case_id="case-1",
        input_id="inp-t1",
        slice_offset=10,
    )
    write_report(
        output,
        run_id="series-t2",
        case_id="case-1",
        input_id="inp-t2",
        slice_offset=30,
    )

    product = build_product_request_from_segmentation(
        DiscFindingsFromSegmentationRequest(
            caseId="case-1",
            sources=[
                {
                    "role": "sagittal_t1",
                    "inputId": "inp-t1",
                    "segmentationRunId": "series-t1",
                },
                {
                    "role": "sagittal_t2",
                    "inputId": "inp-t2",
                    "segmentationRunId": "series-t2",
                },
            ],
        )
    )

    assert len(product.levels) == 5
    l4_l5 = next(item for item in product.levels if item.level == "L4-L5")
    by_role = {item.role: item for item in l4_l5.sourceSeries}
    assert by_role["sagittal_t1"].localization.sliceIndex == 13
    assert by_role["sagittal_t2"].localization.sliceIndex == 33
    assert by_role["sagittal_t1"].localization.segmentationRunId == "series-t1"
    assert by_role["sagittal_t2"].localization.segmentationRunId == "series-t2"


def test_orchestration_allows_single_modality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "outputs"
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(output))
    write_report(
        output,
        run_id="series-t2-only",
        case_id="case-2",
        input_id="inp-t2-only",
        slice_offset=4,
    )

    product = build_product_request_from_segmentation(
        DiscFindingsFromSegmentationRequest(
            caseId="case-2",
            sources=[
                {
                    "role": "sagittal_t2",
                    "inputId": "inp-t2-only",
                    "segmentationRunId": "series-t2-only",
                }
            ],
        )
    )
    assert len(product.levels) == 5
    assert all(len(item.sourceSeries) == 1 for item in product.levels)
    assert all(item.sourceSeries[0].role == "sagittal_t2" for item in product.levels)


def test_orchestration_endpoint_fails_closed_for_missing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    response = TestClient(app).post(
        "/v2/degenerative-findings/disc-multitask/from-series-segmentation",
        json={
            "caseId": "case-x",
            "sources": [
                {
                    "role": "sagittal_t2",
                    "inputId": "inp-x",
                    "segmentationRunId": "series-missing",
                }
            ],
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "DISC_LOCALIZATION_REPORT_INVALID"
    assert body["humanReviewRequired"] is True
    assert body["notClinicalDiagnosis"] is True

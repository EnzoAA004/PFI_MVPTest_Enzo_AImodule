from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.full_series_segmentation import (
    FullSeriesSegmentationRequest,
    SEMANTIC_COLORS,
    resolve_full_series_asset,
    run_full_series_segmentation,
)
from pfi_ai_service.input_registry import register_existing_path


class FakeSagittalSegmenter(nn.Module):
    """Produces five separated disc_group components on every input slice."""

    def forward(self, tensor):
        batch, _channels, height, width = tensor.shape
        logits = torch.full((batch, 4, height, width), -5.0, device=tensor.device)
        logits[:, 0] = 0.0
        for index in range(5):
            y0 = 25 + index * 40
            y1 = y0 + 12
            x0 = 80
            x1 = 175
            logits[:, 3, y0:y1, x0:x1] = 8.0
        return logits


class FakeCachedModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.model = FakeSagittalSegmenter().eval()
        self.checkpoint = {"sagittal_axis": 2}
        self.runtime_metadata = {"targetSize": (256, 256)}


def test_full_series_segments_every_slice_and_builds_disc_localizations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = np.zeros((80, 80, 6), dtype=np.float32)
    for index in range(volume.shape[2]):
        volume[:, :, index] = float(index + 1)
    source = tmp_path / "sagittal.npy"
    np.save(source, volume)

    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "uploads"))

    registered = register_existing_path(
        case_id="case-full-series",
        plane="sagittal",
        path=source,
        source_key="pytest",
        suffix=".npy",
    )

    import pfi_ai_service.full_series_segmentation as module

    monkeypatch.setattr(
        module,
        "model_status",
        lambda *_args, **_kwargs: {
            "availableForRealInference": True,
            "version": "pytest",
        },
    )
    monkeypatch.setattr(module, "load_model", lambda *_args, **_kwargs: FakeCachedModel())

    report = run_full_series_segmentation(
        FullSeriesSegmentationRequest(
            caseId="case-full-series",
            inputId=str(registered["inputId"]),
            plane="sagittal",
            modelKey="sagittal_spider",
        )
    )

    assert report["sliceCount"] == 6
    assert report["segmentedSliceCount"] == 6
    assert report["coverageComplete"] is True
    assert report["defaultPresentation"] == "original"
    assert report["overlayPresentation"] == "on_demand"
    assert report["anatomyColorPolicy"] == "stable_by_anatomical_role"
    assert report["semanticColors"]["disc"] == SEMANTIC_COLORS["disc"]
    assert report["automaticDiscLocalizationValidated"] is False
    assert report["humanReviewRequired"] is True
    assert report["notClinicalDiagnosis"] is True

    localizations = report["discLocalizations"]
    assert [item["level"] for item in localizations] == [
        "L1-L2",
        "L2-L3",
        "L3-L4",
        "L4-L5",
        "L5-S1",
    ]
    assert all(item["source"] == "segmentation_derived_disc_level" for item in localizations)
    assert all(item["automaticAnatomicalLocalizationValidated"] is False for item in localizations)

    for slice_item in report["slices"]:
        assert slice_item["status"] == "segmented"
        assert slice_item["segmentation"]["encoding"] == "rle-v1"
        disc_masks = [mask for mask in slice_item["masks"] if mask["label"] == "disc"]
        assert len(disc_masks) == 5
        assert all(mask["semanticColor"] == SEMANTIC_COLORS["disc"] for mask in disc_masks)
        assert all(mask["colorPolicy"] == "stable_by_anatomical_role" for mask in disc_masks)
        assert slice_item["measurements"]

    original = resolve_full_series_asset(report["runId"], "sagittal", 0, "original.png")
    overlay = resolve_full_series_asset(report["runId"], "sagittal", 0, "overlay.png")
    assert original.is_file()
    assert overlay.is_file()


def test_full_series_endpoint_validation_is_strict() -> None:
    response = TestClient(app).post(
        "/v2/series-segmentation/run",
        json={
            "caseId": "case-x",
            "inputId": "inp-x",
            "plane": "sagittal",
            "modelKey": "sagittal_spider",
            "patientId": "must-not-be-accepted",
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert "patientId" not in body

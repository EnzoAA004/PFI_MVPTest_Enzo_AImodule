import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.disc_degenerative_product_runtime import (
    MARGIN_RATIO,
    OUTPUT_SIZE,
    SegmentationDerivedDiscLocalization,
    crop_2p5d_from_segmentation_roi,
    normalize_volume_notebook66,
    product_runtime_status,
)


def notebook66_reference(volume: np.ndarray, *, center_z: int, bbox_yx: tuple[int, int, int, int]) -> np.ndarray:
    image = np.asarray(volume, dtype=np.float32)
    y0, y1, x0, x1 = bbox_yx
    height, width = y1 - y0, x1 - x0
    margin_y = max(8, int(round(height * MARGIN_RATIO)))
    margin_x = max(8, int(round(width * MARGIN_RATIO)))
    y0, y1 = max(0, y0 - margin_y), min(image.shape[1], y1 + margin_y)
    x0, x1 = max(0, x0 - margin_x), min(image.shape[2], x1 + margin_x)
    z_indices = [max(0, min(image.shape[0] - 1, center_z + delta)) for delta in (-1, 0, 1)]
    normalized = normalize_volume_notebook66(image)
    crop = normalized[z_indices, y0:y1, x0:x1]
    tensor = torch.from_numpy(crop).unsqueeze(0)
    return F.interpolate(
        tensor,
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).numpy().astype(np.float32)


def test_segmentation_roi_preprocessing_matches_notebook66_math() -> None:
    rng = np.random.default_rng(2026)
    volume = rng.normal(500.0, 120.0, size=(11, 90, 70)).astype(np.float32)
    bbox = (30, 46, 20, 37)
    localization = SegmentationDerivedDiscLocalization(
        source="segmentation_derived_disc_level",
        sliceIndex=5,
        bboxYx=bbox,
        coordinateSpaceWidth=70,
        coordinateSpaceHeight=90,
        maskId="mask-sagittal-disc-l4-l5",
    )

    actual = crop_2p5d_from_segmentation_roi(volume, localization)
    expected = notebook66_reference(volume, center_z=5, bbox_yx=bbox)

    assert actual.crop.shape == (3, 224, 224)
    assert actual.positions == (4, 5, 6)
    np.testing.assert_allclose(actual.crop, expected, rtol=0.0, atol=0.0)


def test_segmentation_roi_scales_model_coordinate_bbox_to_native_pixels() -> None:
    volume = np.arange(7 * 100 * 200, dtype=np.float32).reshape(7, 100, 200)
    localization = SegmentationDerivedDiscLocalization(
        source="segmentation_derived_disc_level",
        sliceIndex=3,
        bboxYx=(64, 128, 32, 96),
        coordinateSpaceWidth=256,
        coordinateSpaceHeight=256,
    )
    result = crop_2p5d_from_segmentation_roi(volume, localization)
    assert result.crop.shape == (3, 224, 224)
    assert result.positions == (2, 3, 4)
    assert all(np.isfinite(result.crop).ravel())


def test_product_runtime_reports_parity_but_not_automatic_localization() -> None:
    status = product_runtime_status()
    assert status["preprocessingParityValidated"] is True
    assert status["automaticDiscLocalizationValidated"] is False
    assert status["externalRoiAccepted"] is False


def test_product_endpoint_contract_is_strict_and_rejects_external_roi() -> None:
    response = TestClient(app).post(
        "/v2/degenerative-findings/disc-multitask/predict",
        json={
            "caseId": "case-1",
            "levels": [
                {
                    "level": "L4-L5",
                    "sourceSeries": [
                        {
                            "role": "sagittal_t2",
                            "inputId": "inp-x",
                            "available": True,
                            "localization": {
                                "source": "external_disc_roi",
                                "sliceIndex": 4,
                                "bboxYx": [10, 20, 10, 20],
                                "coordinateSpaceWidth": 256,
                                "coordinateSpaceHeight": 256
                            }
                        }
                    ]
                }
            ]
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_product_runtime_status_endpoint() -> None:
    response = TestClient(app).get(
        "/v2/degenerative-findings/disc-multitask/runtime"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["preprocessingParityValidated"] is True
    assert body["automaticDiscLocalizationValidated"] is False
    assert body["notClinicalDiagnosis"] is True

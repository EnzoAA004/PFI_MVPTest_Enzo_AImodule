from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pfi_ai_service.api import app
from pfi_ai_service.multiplanar_v2_executor import LegacyMultiplanarV1Adapter
from pfi_ai_service.multiplanar_v2_models import (
    CoordinateSpaceV2,
    GovernanceV2,
    MultiplanarReadinessV2,
    MultiplanarRunV2Response,
    PlaneAssetV2,
    PlaneInputV2,
    PlaneModelV2,
    PlaneQualityV2,
    PlaneRunV2Result,
    ReviewPolicyV2,
    ThreeDStatusV2,
    WorkspaceQualityV2,
)


def real_baseline_assets(run_id: str, plane: str) -> list[PlaneAssetV2]:
    return [
        PlaneAssetV2(assetName="input.png", role="input_preview", contentType="image/png", generated=True, relativePath=f"/assets/{run_id}/{plane}/input.png"),
        PlaneAssetV2(assetName="overlay.png", role="overlay", contentType="image/png", generated=True, relativePath=f"/assets/{run_id}/{plane}/overlay.png"),
        PlaneAssetV2(assetName="mask.npy", role="mask_array", contentType="application/octet-stream", generated=True, relativePath=f"/assets/{run_id}/{plane}/mask.npy"),
        PlaneAssetV2(assetName="confidence.npy", role="confidence_array", contentType="application/octet-stream", generated=True, relativePath=f"/assets/{run_id}/{plane}/confidence.npy"),
    ]


def fake_v2_response_with_assets(case_id: str, trace_id: str) -> MultiplanarRunV2Response:
    quality = PlaneQualityV2(maskCount=0, landmarkCount=0, measurementCount=0)
    sagittal = PlaneRunV2Result(
        status="ready",
        plane="sagittal",
        runId="real-sag",
        effectiveInferenceMode="real_baseline",
        model=PlaneModelV2(
            key="sagittal_spider",
            version="sagittal-spider-final-v1",
            readiness="real_baseline_ready",
            trainingStatus="final",
            artifactHash="cf11dcc0ad77a7c787e64a796a2fd7398ef906add461cef4b3d61f1a5238e944",
            baselineReady=True,
            availableForRealInference=True,
            manifestStatus="valid",
            manifestValid=True,
        ),
        input=PlaneInputV2(
            inputId="input-sag",
            format="mha",
            sizeBytes=2361432,
            nativeShape=[352, 384, 17],
            canonicalShape=[352, 384, 17],
            orientationTransform="none",
            spacingXyzMm=[0.7408, 0.6770, 3.0],
            canonicalAxisSpacingMm=[0.7408, 0.6770, 3.0],
            selectedSliceIndex=7,
            sliceCount=17,
            selectedAxis=2,
            inPlaneSpacingMm=[0.7408, 0.6770],
        ),
        coordinateSpace=CoordinateSpaceV2(name="real_baseline_pixel_space", width=256, height=256, units="pixel", origin="top_left", xDirection="right", yDirection="down", sourceSliceIndex=7, sourceAxis=2),
        series=[],
        assets=real_baseline_assets("real-sag", "sagittal"),
        masks=[],
        landmarks=[],
        measurements=[],
        quality=quality,
        synthetic=False,
        fallbackReason=None,
    )
    return MultiplanarRunV2Response(
        status="completed",
        schemaVersion="pfi.multiplanar-run.v2",
        runId="multi-real",
        traceId=trace_id,
        caseId=case_id,
        workspaceMode="sagittal_only",
        requestedInferenceMode="real_baseline",
        effectiveInferenceMode="real_baseline",
        requestedPlanes=["sagittal"],
        completedPlanes=["sagittal"],
        readiness=MultiplanarReadinessV2(sagittal=True, axial=False, dual=False),
        planes={"sagittal": sagittal, "axial": None},
        threeD=ThreeDStatusV2(enabled=False, status="blocked_missing_axial", sourcePlaneRunIds={"sagittal": "real-sag", "axial": None}, requiredInputs=[]),
        quality=WorkspaceQualityV2(planeCount=1, maskCount=0, landmarkCount=0, measurementCount=0, byPlane={"sagittal": quality, "axial": None}),
        review=ReviewPolicyV2(status="pending", required=True, approvalRequiresHumanConfirmation=True),
        governance=GovernanceV2(humanReviewRequired=True, notClinicalDiagnosis=True),
        synthetic=False,
        fallbackReason=None,
    )


def test_legacy_adapter_returns_assets_as_map_not_list() -> None:
    response = fake_v2_response_with_assets("CASE-P9A2-ADAPTER", "trace-p9a2-adapter")
    payload = LegacyMultiplanarV1Adapter().from_v2(response)

    sagittal = payload["planes"]["sagittal"]
    assets = sagittal["assets"]

    assert isinstance(assets, dict)
    assert not isinstance(assets, list)
    assert set(assets.keys()) >= {"input.png", "overlay.png", "mask.npy", "confidence.npy"}

    input_asset = assets["input.png"]
    assert input_asset["role"] == "input_preview"
    assert input_asset["contentType"] == "image/png"
    assert input_asset["generated"] is True
    assert input_asset["relativePath"] == "/assets/real-sag/sagittal/input.png"
    assert "assetName" not in input_asset

    overlay_asset = assets["overlay.png"]
    assert overlay_asset["role"] == "overlay"
    assert overlay_asset["contentType"] == "image/png"
    assert overlay_asset["generated"] is True
    assert overlay_asset["relativePath"] == "/assets/real-sag/sagittal/overlay.png"

    assert isinstance(sagittal["measurements"], dict)
    assert isinstance(sagittal["measurements"]["values"], list)
    assert isinstance(sagittal["series"], list)
    assert isinstance(sagittal["masks"], list)
    assert isinstance(sagittal["landmarks"], list)

    assert payload["humanReviewRequired"] is True
    assert payload["notClinicalDiagnosis"] is True


def test_multiplanar_run_endpoint_assets_serialize_as_object(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PFI_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("PFI_MODEL_DIR", str(tmp_path / "models"))

    class FakeExecutor:
        def run(self, request: Any, *, trace_id: str, allow_unregistered_inputs: bool = False) -> MultiplanarRunV2Response:
            return fake_v2_response_with_assets(request.caseId, trace_id)

    monkeypatch.setattr("pfi_ai_service.multiplanar_routes.CanonicalMultiplanarExecutor", lambda: FakeExecutor())

    client = TestClient(app)
    response = client.post(
        "/multiplanar/run",
        headers={"X-Trace-Id": "trace-p9a2-endpoint"},
        json={"caseId": "CASE-P9A2-ENDPOINT", "metadata": {"inferenceMode": "real_baseline"}},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assets = body["planes"]["sagittal"]["assets"]
    assert isinstance(assets, dict)
    assert not isinstance(assets, list)
    assert "input.png" in assets
    assert "overlay.png" in assets


def test_v2_multiplanar_run_endpoint_assets_remain_a_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.test_multiplanar_v2_contract import configure, fake_plane_response, ready_model, register_input

    configure(monkeypatch, tmp_path)
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.model_status", lambda key, info: ready_model(key, info["plane"], ready=True))
    monkeypatch.setattr("pfi_ai_service.multiplanar_v2_executor.run_pipeline", lambda request: fake_plane_response(request))

    client = TestClient(app)
    case_id = "CASE-P9A2-V2"
    sagittal_id = register_input(client, case_id, "sagittal")

    response = client.post(
        "/v2/multiplanar/run",
        headers={"X-Trace-Id": "trace-p9a2-v2"},
        json={
            "caseId": case_id,
            "inferenceMode": "real_baseline",
            "allowContractFallback": False,
            "planes": {"sagittal": {"inputId": sagittal_id, "modelKey": "sagittal_spider"}, "axial": None},
        },
    )

    assert response.status_code == 200, response.text
    assets = response.json()["planes"]["sagittal"]["assets"]
    assert isinstance(assets, list)
    assert {asset["assetName"] for asset in assets} == {"input.png", "overlay.png", "mask.npy", "confidence.npy"}

"""P10.9 AI product checkpoint contract.

This is the machine-readable handoff surface for Backend integration. It distinguishes
implemented code paths from validation gates that still require real checkpoint/E2E
evidence, so consumers do not infer readiness from endpoint existence alone.
"""
from __future__ import annotations

from typing import Any

from .disc_degenerative_product_runtime import product_runtime_status
from .disc_degenerative_runtime import get_disc_degenerative_runtime_status
from .full_series_segmentation import SEMANTIC_COLORS
from .subarticular_runtime_service import get_subarticular_runtime_status


def ai_product_checkpoint_contract() -> dict[str, Any]:
    disc_runtime = get_disc_degenerative_runtime_status(verify_hash=False)
    product_runtime = product_runtime_status()
    return {
        "schemaVersion": "pfi.p10-9.ai-product-checkpoint.v1",
        "status": "integration_in_progress",
        "frontendRequiredForValidation": False,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "autonomousDiagnosis": False,
        "study": {
            "deidentificationRequired": True,
            "allSeriesPreservedForReview": True,
            "segmentationScope": "supported_analyzable_series_all_slices",
            "unsupportedSeriesStillViewable": True,
            "allStudySeriesAutomaticallySegmented": False,
        },
        "fullSeriesSegmentation": {
            "schemaVersion": "pfi.full-series-segmentation.v1",
            "implemented": True,
            "coverageDefinition": "every_slice_of_supported_analyzable_series",
            "originalFirst": True,
            "overlayOnDemand": True,
            "returnsRleSegmentation": True,
            "returnsMasks": True,
            "returnsMeasurements": True,
            "returnsOriginalPng": True,
            "returnsOverlayPng": True,
            "realCheckpointE2EValidated": False,
        },
        "anatomyPresentation": {
            "colorPolicy": "stable_by_anatomical_role",
            "semanticColors": SEMANTIC_COLORS,
            "anatomyKeyFormat": "<role>[:<level>]",
            "severityEncodedByColor": False,
        },
        "measurements": {
            "descriptiveOnly": True,
            "professionalReviewRequired": True,
            "valueUnitPointsPlaneLevelSliceTraceability": True,
            "clinicalThresholdsFrozen": False,
            "automaticP10_8ProtocolValidation": False,
        },
        "p10_6": {
            "subarticular": get_subarticular_runtime_status(),
            "regressionBaseline": "ci_green_before_p10_9_extensions",
            "noRetraining": True,
        },
        "p10_7": {
            "checkpointRuntime": disc_runtime,
            "productPreprocessing": product_runtime,
            "preprocessingParityCodeTested": bool(
                product_runtime.get("preprocessingParityValidated")
            ),
            "segmentationDerivedLocalizationImplemented": True,
            "automaticDiscLocalizationValidated": False,
            "realCheckpointE2EValidated": False,
            "crossModalityPixelRegistrationAssumed": False,
            "singleModalityAllowed": True,
        },
        "e2eGates": {
            "p10_6Regression": True,
            "p10_7PreprocessingParity": bool(
                product_runtime.get("preprocessingParityValidated")
            ),
            "fullSeriesSyntheticCoverage": False,
            "p10_7RealCheckpointSmoke": False,
            "automaticDiscLocalizationRealStudy": False,
            "aiBackendHttp": False,
            "assetRetrieval": False,
            "piiAudit": False,
            "checkpointReadyForFrontendHandoff": False,
        },
    }

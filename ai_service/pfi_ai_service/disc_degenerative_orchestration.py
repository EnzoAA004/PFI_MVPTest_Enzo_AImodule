"""Bridge from full-series sagittal segmentation reports to P10.7 inference.

T1 and T2 remain independent sources. A localization produced on one series is never
silently reused on another series. This preserves the training design and avoids
assuming pixel registration between modalities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .disc_degenerative_product_runtime import (
    ProductDiscLevelPredictItem,
    ProductDiscMultitaskPredictRequest,
    ProductDiscSourceSeries,
    SegmentationDerivedDiscLocalization,
    predict_product_disc_degenerative,
)
from .full_series_segmentation import SAFE_ID
from .settings import get_settings


class SegmentationSourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["sagittal_t1", "sagittal_t2"]
    inputId: str
    segmentationRunId: str

    @field_validator("inputId", "segmentationRunId")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must_not_be_empty")
        return value


class DiscFindingsFromSegmentationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caseId: str
    sources: list[SegmentationSourceReference] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def unique_roles(self) -> "DiscFindingsFromSegmentationRequest":
        roles = [source.role for source in self.sources]
        if len(roles) != len(set(roles)):
            raise ValueError("duplicate_source_role")
        return self


class SegmentationReportError(Exception):
    pass


def _report_path(run_id: str) -> Path:
    if not SAFE_ID.fullmatch(run_id):
        raise SegmentationReportError("invalid_segmentation_run_id")
    return (
        get_settings().output_dir
        / "full_series_segmentation"
        / run_id
        / "sagittal"
        / "report.json"
    )


def load_segmentation_report(run_id: str) -> dict[str, Any]:
    path = _report_path(run_id)
    if not path.is_file():
        raise SegmentationReportError("segmentation_report_not_found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SegmentationReportError("segmentation_report_invalid") from exc
    if not isinstance(payload, dict):
        raise SegmentationReportError("segmentation_report_invalid")
    return payload


def _validated_localizations(
    report: dict[str, Any],
    source: SegmentationSourceReference,
    case_id: str,
) -> dict[str, SegmentationDerivedDiscLocalization]:
    if report.get("schemaVersion") != "pfi.full-series-segmentation.v1":
        raise SegmentationReportError("segmentation_report_schema_mismatch")
    if report.get("status") != "completed" or report.get("coverageComplete") is not True:
        raise SegmentationReportError("segmentation_report_incomplete")
    if report.get("caseId") != case_id:
        raise SegmentationReportError("segmentation_case_mismatch")
    if report.get("inputId") != source.inputId:
        raise SegmentationReportError("segmentation_input_mismatch")
    if report.get("plane") != "sagittal":
        raise SegmentationReportError("segmentation_plane_mismatch")
    if report.get("runId") != source.segmentationRunId:
        raise SegmentationReportError("segmentation_run_mismatch")

    result: dict[str, SegmentationDerivedDiscLocalization] = {}
    for item in report.get("discLocalizations") or []:
        if not isinstance(item, dict):
            continue
        level = str(item.get("level") or "")
        if not level:
            continue
        if item.get("source") != "segmentation_derived_disc_level":
            raise SegmentationReportError("unsupported_localization_source")
        result[level] = SegmentationDerivedDiscLocalization(
            source="segmentation_derived_disc_level",
            sliceIndex=int(item["sliceIndex"]),
            bboxYx=tuple(item["bboxYx"]),
            coordinateSpaceWidth=int(item["coordinateSpaceWidth"]),
            coordinateSpaceHeight=int(item["coordinateSpaceHeight"]),
            maskId=str(item.get("maskId")) if item.get("maskId") else None,
            segmentationRunId=source.segmentationRunId,
        )
    if not result:
        raise SegmentationReportError("no_disc_localizations_available")
    return result


def build_product_request_from_segmentation(
    request: DiscFindingsFromSegmentationRequest,
) -> ProductDiscMultitaskPredictRequest:
    by_role: dict[str, tuple[SegmentationSourceReference, dict[str, SegmentationDerivedDiscLocalization]]] = {}
    all_levels: set[str] = set()

    for source in request.sources:
        report = load_segmentation_report(source.segmentationRunId)
        localizations = _validated_localizations(report, source, request.caseId)
        by_role[source.role] = (source, localizations)
        all_levels.update(localizations.keys())

    levels: list[ProductDiscLevelPredictItem] = []
    for level in ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"):
        if level not in all_levels:
            continue
        source_series: list[ProductDiscSourceSeries] = []
        for role in ("sagittal_t1", "sagittal_t2"):
            pair = by_role.get(role)
            if pair is None:
                continue
            source, localizations = pair
            localization = localizations.get(level)
            if localization is None:
                continue
            source_series.append(
                ProductDiscSourceSeries(
                    role=role,
                    inputId=source.inputId,
                    available=True,
                    localization=localization,
                )
            )
        if source_series:
            levels.append(
                ProductDiscLevelPredictItem(
                    level=level,
                    sourceSeries=source_series,
                )
            )

    if not levels:
        raise SegmentationReportError("no_common_product_levels_available")

    return ProductDiscMultitaskPredictRequest(caseId=request.caseId, levels=levels)


def predict_from_segmentation_reports(request: DiscFindingsFromSegmentationRequest) -> dict[str, Any]:
    product_request = build_product_request_from_segmentation(request)
    envelope = predict_product_disc_degenerative(product_request)
    envelope["orchestration"] = {
        "source": "full_series_segmentation_reports",
        "sourceRoles": [source.role for source in request.sources],
        "automaticDiscLocalizationValidated": False,
        "crossModalityPixelRegistrationAssumed": False,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }
    return envelope

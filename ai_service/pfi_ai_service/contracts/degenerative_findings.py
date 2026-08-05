from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

SCHEMA_VERSION = "pfi.degenerative-findings.v1"

FINDING_TYPES = frozenset(
    {
        "central_canal_stenosis",
        "neural_foraminal_narrowing",
        "subarticular_stenosis",
    }
)
SEVERITY_LABELS = ("normal_mild", "moderate", "severe")
SPINE_LEVELS = frozenset({"L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"})
SERIES_ROLES = frozenset({"sagittal_t2", "sagittal_t1", "axial_t2"})
EVALUATION_STATUSES = frozenset({"evaluated", "not_evaluated", "unsupported", "failed"})
REVIEW_STATUSES = frozenset({"pending", "accepted", "observed", "rejected", "edited"})
LOCALIZATION_SOURCES = frozenset({"slice_index", "external_coordinate", "model_generated_roi", "not_available"})

FORBIDDEN_IDENTIFIER_KEYS = frozenset(
    {
        "SeriesInstanceUID",
        "StudyInstanceUID",
        "SOPInstanceUID",
        "FrameOfReferenceUID",
        "PatientID",
        "PatientName",
        "AccessionNumber",
        "patientId",
        "seriesInstanceUid",
        "studyInstanceUid",
        "sopInstanceUid",
    }
)
FORBIDDEN_TERMS = frozenset({"diagnosis", "disease", "lesion"})
PROBABILITY_TOLERANCE = 1e-6


class DegenerativeFindingsContractError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("Invalid pfi.degenerative-findings.v1 payload")
        self.errors = tuple(errors)


@dataclass(frozen=True)
class Anatomy:
    level: str
    side: str | None


@dataclass(frozen=True)
class Classification:
    label: str
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class Evaluation:
    status: str
    reasonCode: str | None = None


@dataclass(frozen=True)
class SourceSeries:
    role: str
    position: int


@dataclass(frozen=True)
class Localization:
    source: str
    researchOnly: bool


@dataclass(frozen=True)
class ModelProvenance:
    modelId: str
    modelSha256: str


@dataclass(frozen=True)
class Review:
    required: bool
    status: str


@dataclass(frozen=True)
class DegenerativeFinding:
    findingId: str
    findingType: str
    anatomy: Anatomy
    classification: Classification
    evaluation: Evaluation
    sourceSeries: SourceSeries
    localization: Localization
    model: ModelProvenance
    review: Review
    notClinicalDiagnosis: bool


@dataclass(frozen=True)
class DegenerativeFindingsPayload:
    schemaVersion: str
    findings: tuple[DegenerativeFinding, ...]


def validate_degenerative_findings_payload(payload: Mapping[str, Any]) -> None:
    errors = _collect_errors(payload)
    if errors:
        raise DegenerativeFindingsContractError(errors)


def parse_degenerative_findings_payload(payload: Mapping[str, Any]) -> DegenerativeFindingsPayload:
    validate_degenerative_findings_payload(payload)
    findings = tuple(_parse_finding(item) for item in payload["findings"])
    return DegenerativeFindingsPayload(schemaVersion=SCHEMA_VERSION, findings=findings)


def _parse_finding(item: Mapping[str, Any]) -> DegenerativeFinding:
    anatomy = item["anatomy"]
    classification = item["classification"]
    evaluation = item["evaluation"]
    source_series = item["sourceSeries"]
    localization = item["localization"]
    model = item["model"]
    review = item["review"]
    return DegenerativeFinding(
        findingId=item["findingId"],
        findingType=item["findingType"],
        anatomy=Anatomy(level=anatomy["level"], side=anatomy.get("side")),
        classification=Classification(label=classification["label"], probabilities=dict(classification["probabilities"])),
        evaluation=Evaluation(status=evaluation["status"], reasonCode=evaluation.get("reasonCode")),
        sourceSeries=SourceSeries(role=source_series["role"], position=source_series["position"]),
        localization=Localization(source=localization["source"], researchOnly=localization["researchOnly"]),
        model=ModelProvenance(modelId=model["modelId"], modelSha256=model["modelSha256"]),
        review=Review(required=review["required"], status=review["status"]),
        notClinicalDiagnosis=item["notClinicalDiagnosis"],
    )


def _collect_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["root must be an object"]

    _reject_forbidden_content(payload, "$", errors)

    if payload.get("schemaVersion") != SCHEMA_VERSION:
        errors.append("schemaVersion must be pfi.degenerative-findings.v1")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be an array")
        return errors

    for index, finding in enumerate(findings):
        path = f"findings[{index}]"
        if not isinstance(finding, Mapping):
            errors.append(f"{path} must be an object")
            continue
        _validate_finding(finding, path, errors)
    return errors


def _validate_finding(finding: Mapping[str, Any], path: str, errors: list[str]) -> None:
    finding_id = finding.get("findingId")
    finding_type = finding.get("findingType")
    if not _non_empty_string(finding_id):
        errors.append(f"{path}.findingId must be a non-empty stable string")
    if finding_type not in FINDING_TYPES:
        errors.append(f"{path}.findingType is not supported")

    anatomy = _required_mapping(finding, "anatomy", path, errors)
    classification = _required_mapping(finding, "classification", path, errors)
    evaluation = _required_mapping(finding, "evaluation", path, errors)
    source_series = _required_mapping(finding, "sourceSeries", path, errors)
    localization = _required_mapping(finding, "localization", path, errors)
    model = _required_mapping(finding, "model", path, errors)
    review = _required_mapping(finding, "review", path, errors)

    if anatomy:
        _validate_anatomy(anatomy, finding_type, f"{path}.anatomy", errors)
    if classification:
        _validate_classification(classification, f"{path}.classification", errors)
    if evaluation:
        _validate_evaluation(evaluation, f"{path}.evaluation", errors)
    if source_series:
        _validate_source_series(source_series, f"{path}.sourceSeries", errors)
    if localization:
        _validate_localization(localization, f"{path}.localization", errors)
    if model:
        _validate_model(model, f"{path}.model", errors)
    if review:
        _validate_review(review, f"{path}.review", errors)
    if finding.get("notClinicalDiagnosis") is not True:
        errors.append(f"{path}.notClinicalDiagnosis must be true")


def _validate_anatomy(anatomy: Mapping[str, Any], finding_type: Any, path: str, errors: list[str]) -> None:
    if anatomy.get("level") not in SPINE_LEVELS:
        errors.append(f"{path}.level is not a supported lumbar level")
    side = anatomy.get("side")
    if finding_type == "central_canal_stenosis" and side is not None:
        errors.append(f"{path}.side must be null for central_canal_stenosis")
    if finding_type in {"neural_foraminal_narrowing", "subarticular_stenosis"} and side not in {"left", "right"}:
        errors.append(f"{path}.side must be left or right for foraminal/subarticular findings")


def _validate_classification(classification: Mapping[str, Any], path: str, errors: list[str]) -> None:
    label = classification.get("label")
    probabilities = classification.get("probabilities")
    if label not in SEVERITY_LABELS:
        errors.append(f"{path}.label is not a supported severity")
    if not isinstance(probabilities, Mapping):
        errors.append(f"{path}.probabilities must be an object")
        return
    if set(probabilities.keys()) != set(SEVERITY_LABELS):
        errors.append(f"{path}.probabilities must contain exactly normal_mild, moderate and severe")
        return
    values: dict[str, float] = {}
    for key, value in probabilities.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
            errors.append(f"{path}.probabilities.{key} must be finite")
            return
        number = float(value)
        if number < 0 or number > 1:
            errors.append(f"{path}.probabilities.{key} must be between 0 and 1")
        values[key] = number
    if abs(sum(values.values()) - 1.0) > PROBABILITY_TOLERANCE:
        errors.append(f"{path}.probabilities must sum to 1")
    if values and label in SEVERITY_LABELS:
        argmax = max(values, key=values.get)
        if label != argmax:
            errors.append(f"{path}.label must match the maximum probability")


def _validate_evaluation(evaluation: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if evaluation.get("status") not in EVALUATION_STATUSES:
        errors.append(f"{path}.status is not supported")
    reason_code = evaluation.get("reasonCode")
    if reason_code is not None and not _non_empty_string(reason_code):
        errors.append(f"{path}.reasonCode must be a non-empty string when present")


def _validate_source_series(source_series: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if source_series.get("role") not in SERIES_ROLES:
        errors.append(f"{path}.role is not supported")
    position = source_series.get("position")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        errors.append(f"{path}.position must be a non-negative integer")


def _validate_localization(localization: Mapping[str, Any], path: str, errors: list[str]) -> None:
    source = localization.get("source")
    if source not in LOCALIZATION_SOURCES:
        errors.append(f"{path}.source is not supported")
    research_only = localization.get("researchOnly")
    if not isinstance(research_only, bool):
        errors.append(f"{path}.researchOnly must be boolean")
    if source == "external_coordinate" and research_only is not True:
        errors.append(f"{path}.researchOnly must be true for external_coordinate")


def _validate_model(model: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if not _non_empty_string(model.get("modelId")):
        errors.append(f"{path}.modelId must be a non-empty string")
    sha = model.get("modelSha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha.lower()):
        errors.append(f"{path}.modelSha256 must be a 64-character SHA-256 hex string")


def _validate_review(review: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if review.get("required") is not True:
        errors.append(f"{path}.required must be true")
    if review.get("status") not in REVIEW_STATUSES:
        errors.append(f"{path}.status is not supported")


def _required_mapping(parent: Mapping[str, Any], key: str, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{path}.{key} must be an object")
        return None
    return value


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _reject_forbidden_content(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_str = str(key)
            if key_str in FORBIDDEN_IDENTIFIER_KEYS:
                errors.append(f"{path}.{key_str} is a forbidden DICOM/patient identifier")
            if key_str.lower() in FORBIDDEN_TERMS:
                errors.append(f"{path}.{key_str} uses forbidden clinical terminology")
            _reject_forbidden_content(child, f"{path}.{key_str}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_content(child, f"{path}[{index}]", errors)

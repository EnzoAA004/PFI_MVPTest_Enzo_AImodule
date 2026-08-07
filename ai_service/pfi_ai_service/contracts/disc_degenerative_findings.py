from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

SCHEMA_VERSION = "pfi.disc-degenerative-findings.v1"

FINDING_TYPES = (
    "pfirrmann_grade",
    "modic_change",
    "upper_endplate_change",
    "lower_endplate_change",
    "spondylolisthesis",
    "disc_herniation",
    "disc_narrowing",
    "disc_bulging",
)
TASK_ORDER = FINDING_TYPES
CATEGORICAL_TASKS = {"pfirrmann_grade": 5, "modic_change": 4}
BINARY_TASKS = (
    "upper_endplate_change",
    "lower_endplate_change",
    "spondylolisthesis",
    "disc_herniation",
    "disc_narrowing",
    "disc_bulging",
)

PFIRRMANN_LABELS = ("I", "II", "III", "IV", "V")
MODIC_LABELS = ("none", "I", "II", "III")
BINARY_LABELS = ("absent", "present")
SPINE_LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
REVIEW_STATUSES = ("pending", "accepted", "observed", "rejected", "edited")
DEPLOYMENT_STATUSES = ("supported_internal", "experimental", "not_product_supported")
LOCALIZATION_SOURCES = ("segmentation_derived_disc_level", "external_disc_roi")
SERIES_ROLES = ("sagittal_t1", "sagittal_t2")
MODEL_ID = "spider_degenerative_multitask_sagittal_t1_t2_2p5d"
EXPECTED_CHECKPOINT_SHA256 = "16eccff327e6794b127fe372ecd03ea619a0f69d939b84ae1aa2e904191c6293"

DEPLOYMENT_STATUS_BY_TASK = {
    "upper_endplate_change": "supported_internal",
    "lower_endplate_change": "supported_internal",
    "disc_narrowing": "supported_internal",
    "disc_bulging": "supported_internal",
    "pfirrmann_grade": "experimental",
    "modic_change": "not_product_supported",
    "spondylolisthesis": "not_product_supported",
    "disc_herniation": "not_product_supported",
}

PROBABILITY_TOLERANCE = 1e-5
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


class DiscDegenerativeFindingsContractError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("Invalid pfi.disc-degenerative-findings.v1 payload")
        self.errors = tuple(errors)


@dataclass(frozen=True)
class LevelMapping:
    level: str
    ivd_label: int
    ivd_index: int
    raw_disc_label: int


def level_to_ivd_mapping(level: str) -> LevelMapping:
    if level not in SPINE_LEVELS:
        raise ValueError("unsupported_lumbar_level")
    ivd_label = SPINE_LEVELS.index(level) + 1
    return LevelMapping(level=level, ivd_label=ivd_label, ivd_index=ivd_label - 1, raw_disc_label=200 + ivd_label)


def classification_labels(task: str) -> tuple[str, ...]:
    if task == "pfirrmann_grade":
        return PFIRRMANN_LABELS
    if task == "modic_change":
        return MODIC_LABELS
    if task in BINARY_TASKS:
        return BINARY_LABELS
    raise ValueError("unsupported_task")


def classification_kind(task: str) -> str:
    return "categorical" if task in CATEGORICAL_TASKS else "binary"


def validate_disc_degenerative_findings_envelope(envelope: Mapping[str, Any]) -> None:
    errors = _collect_envelope_errors(envelope)
    if errors:
        raise DiscDegenerativeFindingsContractError(errors)


def validate_disc_degenerative_findings_payload(payload: Mapping[str, Any]) -> None:
    errors = _collect_payload_errors(payload, "discDegenerativeFindings")
    if errors:
        raise DiscDegenerativeFindingsContractError(errors)


def _collect_envelope_errors(envelope: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(envelope, Mapping):
        return ["root must be an object"]
    _reject_forbidden_content(envelope, "$", errors)
    payload = envelope.get("discDegenerativeFindings")
    if not isinstance(payload, Mapping):
        errors.append("discDegenerativeFindings must be an object")
        return errors
    errors.extend(_collect_payload_errors(payload, "discDegenerativeFindings"))
    if envelope.get("humanReviewRequired") is not True:
        errors.append("humanReviewRequired must be true")
    if envelope.get("notClinicalDiagnosis") is not True:
        errors.append("notClinicalDiagnosis must be true")
    if envelope.get("autonomousDiagnosis") is not False:
        errors.append("autonomousDiagnosis must be false")
    return errors


def _collect_payload_errors(payload: Mapping[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return [f"{path} must be an object"]
    _reject_forbidden_content(payload, path, errors)
    if payload.get("schemaVersion") != SCHEMA_VERSION:
        errors.append(f"{path}.schemaVersion must be {SCHEMA_VERSION}")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        errors.append(f"{path}.findings must be an array")
        return errors
    for index, finding in enumerate(findings):
        _validate_finding(finding, f"{path}.findings[{index}]", errors)
    return errors


def _validate_finding(finding: Any, path: str, errors: list[str]) -> None:
    if not isinstance(finding, Mapping):
        errors.append(f"{path} must be an object")
        return
    finding_type = finding.get("findingType")
    if not _non_empty_string(finding.get("findingId")):
        errors.append(f"{path}.findingId must be a non-empty opaque string")
    if finding_type not in FINDING_TYPES:
        errors.append(f"{path}.findingType is not supported")

    anatomy = _required_mapping(finding, "anatomy", path, errors)
    classification = _required_mapping(finding, "classification", path, errors)
    evidence = _required_mapping(finding, "evidence", path, errors)
    evaluation = _required_mapping(finding, "evaluation", path, errors)
    localization = _required_mapping(finding, "localization", path, errors)
    model = _required_mapping(finding, "model", path, errors)
    review = _required_mapping(finding, "review", path, errors)

    if anatomy:
        if anatomy.get("level") not in SPINE_LEVELS:
            errors.append(f"{path}.anatomy.level is not supported")
        if anatomy.get("side") is not None:
            errors.append(f"{path}.anatomy.side must be null")
    if classification and finding_type in FINDING_TYPES:
        _validate_classification(classification, str(finding_type), f"{path}.classification", errors)
    if evidence:
        if evidence.get("deploymentStatus") not in DEPLOYMENT_STATUSES:
            errors.append(f"{path}.evidence.deploymentStatus is not supported")
        if finding_type in DEPLOYMENT_STATUS_BY_TASK and evidence.get("deploymentStatus") != DEPLOYMENT_STATUS_BY_TASK[finding_type]:
            errors.append(f"{path}.evidence.deploymentStatus does not match findingType")
        if evidence.get("evaluationDataset") != "SPIDER_internal_test":
            errors.append(f"{path}.evidence.evaluationDataset must be SPIDER_internal_test")
        if evidence.get("externalValidationAvailable") is not False:
            errors.append(f"{path}.evidence.externalValidationAvailable must be false")
    if evaluation and evaluation.get("status") not in {"evaluated", "not_evaluated", "unsupported", "failed"}:
        errors.append(f"{path}.evaluation.status is not supported")
    _validate_source_series(finding.get("sourceSeries"), f"{path}.sourceSeries", errors)
    if localization:
        if localization.get("source") not in LOCALIZATION_SOURCES:
            errors.append(f"{path}.localization.source is not supported")
        if localization.get("researchOnly") is not True:
            errors.append(f"{path}.localization.researchOnly must be true")
        if localization.get("automaticAnatomicalLocalizationValidated") is not False:
            errors.append(f"{path}.localization.automaticAnatomicalLocalizationValidated must be false")
    if model:
        if model.get("modelId") != MODEL_ID:
            errors.append(f"{path}.model.modelId is not supported")
        _validate_sha(model.get("modelSha256"), f"{path}.model.modelSha256", errors)
    if review:
        if review.get("required") is not True:
            errors.append(f"{path}.review.required must be true")
        if review.get("status") not in REVIEW_STATUSES:
            errors.append(f"{path}.review.status is not supported")
    if finding.get("notClinicalDiagnosis") is not True:
        errors.append(f"{path}.notClinicalDiagnosis must be true")


def _validate_classification(classification: Mapping[str, Any], task: str, path: str, errors: list[str]) -> None:
    expected_labels = classification_labels(task)
    if classification.get("kind") != classification_kind(task):
        errors.append(f"{path}.kind does not match findingType")
    label = classification.get("label")
    probabilities = classification.get("probabilities")
    if label not in expected_labels:
        errors.append(f"{path}.label is not supported")
    if not isinstance(probabilities, Mapping):
        errors.append(f"{path}.probabilities must be an object")
        return
    if set(probabilities.keys()) != set(expected_labels):
        errors.append(f"{path}.probabilities has unexpected labels")
        return
    values = {}
    for key, value in probabilities.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
            errors.append(f"{path}.probabilities.{key} must be finite")
            return
        number = float(value)
        if number < 0.0 or number > 1.0:
            errors.append(f"{path}.probabilities.{key} must be between 0 and 1")
        values[str(key)] = number
    if abs(sum(values.values()) - 1.0) > PROBABILITY_TOLERANCE:
        errors.append(f"{path}.probabilities must sum to 1")
    if values and label in expected_labels and label != max(values, key=values.get):
        errors.append(f"{path}.label must match the maximum probability")


def _validate_source_series(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{path} must be a non-empty array")
        return
    seen_roles: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{item_path} must be an object")
            continue
        role = item.get("role")
        if role not in SERIES_ROLES:
            errors.append(f"{item_path}.role is not supported")
        else:
            seen_roles.add(str(role))
        if not isinstance(item.get("available"), bool):
            errors.append(f"{item_path}.available must be boolean")
        positions = item.get("positions")
        if positions is not None:
            if not isinstance(positions, list) or any(isinstance(pos, bool) or not isinstance(pos, int) or pos < 0 for pos in positions):
                errors.append(f"{item_path}.positions must contain non-negative integers")
    available = [item for item in value if isinstance(item, Mapping) and item.get("available") is True]
    if not available:
        errors.append(f"{path} must include at least one available modality")
    if not seen_roles.issubset(set(SERIES_ROLES)):
        errors.append(f"{path} has unsupported roles")


def _validate_sha(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        errors.append(f"{path} must be a 64-character SHA-256 hex string")


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

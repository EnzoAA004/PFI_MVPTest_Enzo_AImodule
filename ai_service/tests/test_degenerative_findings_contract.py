from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pfi_ai_service.contracts.degenerative_findings import (
    SCHEMA_VERSION,
    DegenerativeFindingsContractError,
    parse_degenerative_findings_payload,
    validate_degenerative_findings_payload,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "degenerative_findings"
SCHEMA_PATH = Path(__file__).parents[2] / "docs" / "contracts" / "degenerative-findings-v1.schema.json"
DOC_PATH = Path(__file__).parents[2] / "docs" / "contracts" / "DEGENERATIVE_FINDINGS_V1.md"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def first_finding(payload: dict) -> dict:
    return payload["findings"][0]


def assert_invalid(payload: dict, expected: str) -> None:
    with pytest.raises(DegenerativeFindingsContractError) as exc:
        validate_degenerative_findings_payload(payload)
    assert any(expected in error for error in exc.value.errors), exc.value.errors


def test_schema_artifact_declares_contract_version_and_fields() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["properties"]["schemaVersion"]["const"] == SCHEMA_VERSION
    assert "finding" in schema["$defs"]
    finding_required = schema["$defs"]["finding"]["required"]
    for key in ["findingId", "findingType", "classification", "evaluation", "sourceSeries", "localization", "model", "review", "notClinicalDiagnosis"]:
        assert key in finding_required


def test_valid_central_canal_finding_parses_to_immutable_contract() -> None:
    parsed = parse_degenerative_findings_payload(load_fixture("valid_central.json"))

    assert parsed.schemaVersion == SCHEMA_VERSION
    assert parsed.findings[0].findingType == "central_canal_stenosis"
    assert parsed.findings[0].anatomy.level == "L4-L5"
    assert parsed.findings[0].anatomy.side is None
    assert parsed.findings[0].review.required is True
    with pytest.raises(Exception):
        parsed.findings[0].findingType = "diagnosis"  # type: ignore[misc]


def test_valid_left_foraminal_finding() -> None:
    parsed = parse_degenerative_findings_payload(load_fixture("valid_foraminal_left.json"))

    finding = parsed.findings[0]
    assert finding.findingType == "neural_foraminal_narrowing"
    assert finding.anatomy.side == "left"
    assert finding.localization.source == "external_coordinate"
    assert finding.localization.researchOnly is True


def test_valid_right_subarticular_finding() -> None:
    parsed = parse_degenerative_findings_payload(load_fixture("valid_subarticular_right.json"))

    finding = parsed.findings[0]
    assert finding.findingType == "subarticular_stenosis"
    assert finding.anatomy.side == "right"
    assert finding.sourceSeries.role == "axial_t2"
    assert finding.localization.source == "model_generated_roi"


def test_probability_sum_must_be_one() -> None:
    assert_invalid(load_fixture("invalid_probability_sum.json"), "probabilities must sum to 1")


def test_label_must_match_probability_argmax() -> None:
    payload = load_fixture("valid_central.json")
    first_finding(payload)["classification"]["label"] = "severe"

    assert_invalid(payload, "label must match the maximum probability")


def test_laterality_rules_are_explicit() -> None:
    central = load_fixture("valid_central.json")
    first_finding(central)["anatomy"]["side"] = "left"
    assert_invalid(central, "side must be null")

    foraminal = load_fixture("valid_foraminal_left.json")
    first_finding(foraminal)["anatomy"]["side"] = None
    assert_invalid(foraminal, "side must be left or right")


def test_invalid_lumbar_level_is_rejected() -> None:
    payload = load_fixture("valid_central.json")
    first_finding(payload)["anatomy"]["level"] = "T12-L1"

    assert_invalid(payload, "level is not a supported lumbar level")


def test_forbidden_dicom_identifier_is_rejected() -> None:
    assert_invalid(load_fixture("invalid_forbidden_identifier.json"), "forbidden DICOM/patient identifier")


def test_external_coordinate_requires_research_only() -> None:
    assert_invalid(load_fixture("invalid_external_coordinate_not_research.json"), "researchOnly must be true")


def test_review_required_must_be_true() -> None:
    payload = load_fixture("valid_central.json")
    first_finding(payload)["review"]["required"] = False

    assert_invalid(payload, "review.required must be true")


def test_not_clinical_diagnosis_must_be_true() -> None:
    payload = load_fixture("valid_central.json")
    first_finding(payload)["notClinicalDiagnosis"] = False

    assert_invalid(payload, "notClinicalDiagnosis must be true")


def test_not_evaluated_with_reason_code_is_valid() -> None:
    payload = load_fixture("valid_central.json")
    finding = first_finding(payload)
    finding["evaluation"] = {"status": "not_evaluated", "reasonCode": "missing_required_sequence"}

    parsed = parse_degenerative_findings_payload(payload)

    assert parsed.findings[0].evaluation.status == "not_evaluated"
    assert parsed.findings[0].evaluation.reasonCode == "missing_required_sequence"


def test_source_series_position_must_be_non_negative_integer() -> None:
    payload = load_fixture("valid_central.json")
    first_finding(payload)["sourceSeries"]["position"] = -1

    assert_invalid(payload, "position must be a non-negative integer")


def test_forbidden_diagnosis_terms_are_not_part_of_docs_or_contract_schema() -> None:
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8").lower()
    doc_text = DOC_PATH.read_text(encoding="utf-8").lower()
    protected_schema_text = schema_text.replace("notclinicaldiagnosis", "")
    protected_doc_text = doc_text.replace("notclinicaldiagnosis", "").replace("diagnosis, disease or lesion claims", "")

    assert "diagnosis" not in protected_schema_text
    assert "disease" not in protected_schema_text
    assert "lesion" not in protected_schema_text
    assert "diagnosis" not in protected_doc_text
    assert "disease" not in protected_doc_text
    assert "lesion" not in protected_doc_text


def test_validation_does_not_mutate_payload() -> None:
    payload = load_fixture("valid_subarticular_right.json")
    original = copy.deepcopy(payload)

    validate_degenerative_findings_payload(payload)

    assert payload == original

from __future__ import annotations

from pfi_ai_service.contract_schema import contract_verification, pipeline_contract_schema


def test_pipeline_contract_schema_declares_governance() -> None:
    schema = pipeline_contract_schema()

    assert schema["schemaVersion"] == "visual-review-contract-v1"
    assert schema["status"] == "stable"
    assert schema["generatedBy"] == "pfi-ai-module.contract_schema"
    assert schema["humanReviewRequired"] is True
    assert schema["notClinicalDiagnosis"] is True
    assert "rootFields" in schema
    assert "aiOutput" in schema
    assert "quality" in schema


def test_pipeline_contract_schema_contains_frontend_keys() -> None:
    schema = pipeline_contract_schema()
    root_fields = schema["rootFields"]

    for key in ["series", "masks", "landmarks", "measurementValues", "modelArtifact", "quality", "metadata"]:
        assert key in root_fields

    guarantees = " ".join(schema["guarantees"])
    assert "humanReviewRequired=true" in guarantees
    assert "notClinicalDiagnosis=true" in guarantees


def test_pipeline_contract_schema_hash_is_stable() -> None:
    first = pipeline_contract_schema()
    second = pipeline_contract_schema()

    assert first["schemaHash"] == second["schemaHash"]
    assert isinstance(first["schemaHash"], str)
    assert len(first["schemaHash"]) == 64


def test_pipeline_contract_verification_is_valid() -> None:
    verification = contract_verification()

    assert verification["valid"] is True
    assert verification["hashValid"] is True
    assert verification["governanceValid"] is True
    assert verification["missingRootFields"] == []
    assert verification["schemaHash"] == verification["recomputedHash"]
    assert verification["humanReviewRequired"] is True
    assert verification["notClinicalDiagnosis"] is True

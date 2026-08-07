import json

import pytest

from pfi_ai_service.contracts.disc_degenerative_findings import (
    EXPECTED_CHECKPOINT_SHA256,
    SCHEMA_VERSION,
    DiscDegenerativeFindingsContractError,
    level_to_ivd_mapping,
    validate_disc_degenerative_findings_envelope,
)


def valid_envelope() -> dict:
    return {
        "discDegenerativeFindings": {
            "schemaVersion": SCHEMA_VERSION,
            "findings": [
                {
                    "findingId": "opaque-finding",
                    "findingType": "disc_bulging",
                    "anatomy": {"level": "L4-L5", "side": None},
                    "classification": {
                        "kind": "binary",
                        "label": "present",
                        "probabilities": {"absent": 0.2, "present": 0.8},
                    },
                    "evidence": {
                        "deploymentStatus": "supported_internal",
                        "evaluationDataset": "SPIDER_internal_test",
                        "externalValidationAvailable": False,
                    },
                    "evaluation": {"status": "evaluated"},
                    "sourceSeries": [
                        {"role": "sagittal_t1", "available": True, "positions": [10, 11, 12]},
                        {"role": "sagittal_t2", "available": False, "positions": []},
                    ],
                    "localization": {
                        "source": "segmentation_derived_disc_level",
                        "researchOnly": True,
                        "automaticAnatomicalLocalizationValidated": False,
                    },
                    "model": {
                        "modelId": "spider_degenerative_multitask_sagittal_t1_t2_2p5d",
                        "modelSha256": EXPECTED_CHECKPOINT_SHA256,
                    },
                    "review": {"required": True, "status": "pending"},
                    "notClinicalDiagnosis": True,
                }
            ],
        },
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
        "autonomousDiagnosis": False,
    }


def test_disc_degenerative_contract_accepts_valid_payload() -> None:
    payload = valid_envelope()
    validate_disc_degenerative_findings_envelope(payload)
    text = json.dumps(payload)
    assert "diagnosis" not in text.replace("notClinicalDiagnosis", "")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["discDegenerativeFindings"].update({"schemaVersion": "wrong"}),
        lambda payload: payload.update({"autonomousDiagnosis": True}),
        lambda payload: payload["discDegenerativeFindings"]["findings"][0]["classification"].update({"label": "absent"}),
        lambda payload: payload["discDegenerativeFindings"]["findings"][0]["evidence"].update({"deploymentStatus": "experimental"}),
        lambda payload: payload["discDegenerativeFindings"]["findings"][0].update({"PatientID": "secret"}),
    ],
)
def test_disc_degenerative_contract_rejects_invalid_payloads(mutation) -> None:
    payload = valid_envelope()
    mutation(payload)
    with pytest.raises(DiscDegenerativeFindingsContractError):
        validate_disc_degenerative_findings_envelope(payload)


def test_level_mapping_is_derived_from_spider_ivd_label_order() -> None:
    assert level_to_ivd_mapping("L1-L2").ivd_index == 0
    assert level_to_ivd_mapping("L5-S1").ivd_index == 4
    assert level_to_ivd_mapping("L5-S1").raw_disc_label == 205
    with pytest.raises(ValueError):
        level_to_ivd_mapping("T12-L1")

#!/usr/bin/env python3
"""Real frozen-checkpoint smoke for the P10.9 AI checkpoint.

No training, no dataset access and no sealed-test access. The script loads the exact
frozen P10.6 and P10.7 artifacts, performs one deterministic synthetic forward through
each model and verifies that the returned contracts remain review-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pfi_ai_service.disc_degenerative_runtime import (
    DiscDegenerativeClassifier,
    PreparedDiscLevel,
)
from pfi_ai_service.subarticular_frozen_classifier import (
    EXPECTED_CHECKPOINT_SHA256 as EXPECTED_SUBARTICULAR_SHA256,
    SubarticularFrozenClassifier,
    SubarticularFrozenClassifierConfig,
)
from pfi_ai_service.contracts.disc_degenerative_findings import (
    EXPECTED_CHECKPOINT_SHA256 as EXPECTED_DISC_SHA256,
)


def assert_probability_map(probabilities: dict[str, float]) -> None:
    values = [float(value) for value in probabilities.values()]
    assert values
    assert all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
    assert abs(sum(values) - 1.0) < 1e-5


def smoke_subarticular(path: Path) -> dict[str, object]:
    classifier = SubarticularFrozenClassifier(
        SubarticularFrozenClassifierConfig(
            checkpoint_path=path,
            map_location="cpu",
        )
    ).load()
    sample = np.linspace(0.0, 1.0, 3 * 224 * 224, dtype=np.float32).reshape(3, 224, 224)
    prediction = classifier.predict_preprocessed(
        sample,
        side="left",
        level="L4-L5",
        source_position=4,
        localization_source="external_coordinate",
        research_only=True,
    )
    assert prediction.checkpointSha256 == EXPECTED_SUBARTICULAR_SHA256
    assert prediction.humanReviewRequired is True
    assert prediction.notClinicalDiagnosis is True
    assert prediction.autonomousDiagnosis is False
    assert_probability_map(dict(prediction.probabilities))
    return {
        "status": "PASS",
        "checkpointSha256": prediction.checkpointSha256,
        "predictedSeverity": prediction.predictedSeverity,
        "probabilitySum": sum(float(v) for v in prediction.probabilities.values()),
        "humanReviewRequired": prediction.humanReviewRequired,
        "notClinicalDiagnosis": prediction.notClinicalDiagnosis,
    }


def smoke_disc_multitask(path: Path) -> dict[str, object]:
    classifier = DiscDegenerativeClassifier(
        path,
        expected_sha256=EXPECTED_DISC_SHA256,
        map_location="cpu",
    ).load()
    t1 = np.linspace(0.0, 1.0, 3 * 224 * 224, dtype=np.float32).reshape(3, 224, 224)
    t2 = np.flip(t1, axis=2).copy()
    envelope = classifier.predict_preprocessed(
        [
            PreparedDiscLevel(
                level="L4-L5",
                t1=t1,
                t2=t2,
                t1_positions=(3, 4, 5),
                t2_positions=(6, 7, 8),
                localization_source="segmentation_derived_disc_level",
            )
        ]
    )
    findings = envelope["discDegenerativeFindings"]["findings"]
    assert len(findings) == 8
    assert envelope["humanReviewRequired"] is True
    assert envelope["notClinicalDiagnosis"] is True
    assert envelope["autonomousDiagnosis"] is False
    for finding in findings:
        assert finding["model"]["modelSha256"] == EXPECTED_DISC_SHA256
        assert finding["localization"]["source"] == "segmentation_derived_disc_level"
        assert finding["localization"]["automaticAnatomicalLocalizationValidated"] is False
        assert finding["review"] == {"required": True, "status": "pending"}
        assert finding["notClinicalDiagnosis"] is True
        assert_probability_map(finding["classification"]["probabilities"])
    return {
        "status": "PASS",
        "checkpointSha256": EXPECTED_DISC_SHA256,
        "findingCount": len(findings),
        "findingTypes": [finding["findingType"] for finding in findings],
        "humanReviewRequired": envelope["humanReviewRequired"],
        "notClinicalDiagnosis": envelope["notClinicalDiagnosis"],
        "automaticDiscLocalizationValidated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subarticular", type=Path, required=True)
    parser.add_argument("--disc-multitask", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.subarticular.is_file():
        raise FileNotFoundError(args.subarticular)
    if not args.disc_multitask.is_file():
        raise FileNotFoundError(args.disc_multitask)

    report = {
        "schemaVersion": "pfi.p10-9.real-checkpoint-smoke.v1",
        "trainingExecuted": False,
        "sealedTestAccessed": False,
        "patientDataUsed": False,
        "subarticular": smoke_subarticular(args.subarticular),
        "discMultitask": smoke_disc_multitask(args.disc_multitask),
        "realCheckpointForwardValidated": True,
        "automaticDiscLocalizationRealStudyValidated": False,
        "humanReviewRequired": True,
        "notClinicalDiagnosis": True,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

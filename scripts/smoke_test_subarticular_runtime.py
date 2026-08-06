from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
AI_SERVICE_ROOT = REPO_ROOT / "ai_service"
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from pfi_ai_service.contracts.degenerative_findings import validate_degenerative_findings_payload
from pfi_ai_service.subarticular_runtime_service import clear_subarticular_classifier_cache, get_subarticular_classifier


def main() -> int:
    if not os.getenv("PFI_SUBARTICULAR_CHECKPOINT_PATH"):
        print(json.dumps({"status": "SUBARTICULAR_REAL_CHECKPOINT_SMOKE_NOT_CONFIGURED"}, indent=2))
        return 2
    clear_subarticular_classifier_cache()
    classifier = get_subarticular_classifier()
    sample = np.arange(3 * 224 * 224, dtype=np.float32).reshape(3, 224, 224) % 255
    prediction = classifier.predict_preprocessed(
        sample,
        side="left",
        level="L4-L5",
        source_position=0,
        localization_source="external_coordinate",
        research_only=True,
    )
    validate_degenerative_findings_payload(prediction.degenerativeFindings)
    probabilities = prediction.degenerativeFindings["findings"][0]["classification"]["probabilities"]
    normalized = abs(sum(probabilities.values()) - 1.0) <= 1e-6
    finite = all(np.isfinite(value) for value in probabilities.values())
    label = prediction.degenerativeFindings["findings"][0]["classification"]["label"]
    argmax_ok = label == max(probabilities, key=probabilities.get)
    ok = finite and normalized and argmax_ok and prediction.humanReviewRequired and prediction.notClinicalDiagnosis and not prediction.autonomousDiagnosis
    print(json.dumps({
        "status": "SUBARTICULAR_REAL_CHECKPOINT_SMOKE_OK" if ok else "SUBARTICULAR_REAL_CHECKPOINT_SMOKE_FAILED",
        "checkpointHashVerified": True,
        "modelLoaded": classifier.is_loaded,
        "device": classifier.model_metadata()["device"],
        "probabilitiesFinite": finite,
        "probabilitiesNormalized": normalized,
        "argmaxMatchesLabel": argmax_ok,
        "contractValidated": True,
        "humanReviewRequired": prediction.humanReviewRequired,
        "notClinicalDiagnosis": prediction.notClinicalDiagnosis,
        "autonomousDiagnosis": prediction.autonomousDiagnosis,
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

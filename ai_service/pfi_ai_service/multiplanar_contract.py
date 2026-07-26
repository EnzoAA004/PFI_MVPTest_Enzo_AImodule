from __future__ import annotations

from typing import Any, Dict

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .model_artifacts import artifact_summary, registry_with_artifact_status


def multiplanar_workspace_contract() -> Dict[str, Any]:
    models = registry_with_artifact_status()
    sagittal = models.get("sagittal_spider", {})
    axial = models.get("axial_t2_alkafri", {})
    summary = artifact_summary()
    sagittal_ready = bool(sagittal.get("baselineReady"))
    axial_ready = bool(axial.get("baselineReady"))
    return {
        "status": "multiplanar_ready" if sagittal_ready and axial_ready else "multiplanar_preparation",
        "schemaVersion": "multiplanar-workspace-v1",
        "workspaceMode": "dual_plane_with_3d_context",
        "planes": {
            "sagittal": plane_contract("sagittal", "sagittal_spider", sagittal),
            "axial": plane_contract("axial", "axial_t2_alkafri", axial),
        },
        "threeD": {
            "enabled": False,
            "status": "planned_from_registered_masks",
            "source": "derived_from_sagittal_and_axial_masks",
            "requiredInputs": ["sagittal_masks", "axial_masks", "spacing", "slice_index_mapping"],
            "editable": False,
        },
        "sync": {
            "sliceLinking": True,
            "sharedCaseId": True,
            "sharedRunId": True,
            "sharedTraceId": True,
            "landmarkPropagation": "planned",
            "maskEditPropagation": "planned",
        },
        "review": {
            "professionalReviewRequired": HUMAN_REVIEW_REQUIRED,
            "editableMasks": True,
            "editableLandmarks": True,
            "editableMeasurements": True,
            "approvalRequiresHumanConfirmation": True,
        },
        "modelArtifactSummary": summary,
        "readyForRealBaseline": sagittal_ready and axial_ready,
        "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
        "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
    }


def plane_contract(plane: str, model_key: str, model_status: Dict[str, Any]) -> Dict[str, Any]:
    artifact = model_status.get("artifact", {}) if isinstance(model_status, dict) else {}
    manifest = model_status.get("manifest", {}) if isinstance(model_status, dict) else {}
    return {
        "plane": plane,
        "modelKey": model_key,
        "modelVersion": model_status.get("version"),
        "readiness": model_status.get("readiness", "contract_only_missing_artifact"),
        "baselineReady": bool(model_status.get("baselineReady")),
        "availableForRealInference": bool(model_status.get("availableForRealInference")),
        "artifactHash": model_status.get("artifactHash"),
        "artifactExists": bool(artifact.get("exists")),
        "externalArtifactConfigured": bool(model_status.get("externalArtifactConfigured")),
        "manifestStatus": manifest.get("status"),
        "manifestValid": bool(manifest.get("valid")),
        "outputs": ["series", "masks", "landmarks", "measurements", "quality", "metadata"],
        "viewerRole": "primary" if plane == "sagittal" else "secondary",
    }


def multiplanar_workspace_contract_v2() -> Dict[str, Any]:
    models = registry_with_artifact_status()
    sagittal = models.get("sagittal_spider", {})
    axial = models.get("axial_t2_alkafri", {})
    sagittal_ready = bool(sagittal.get("availableForRealInference"))
    axial_ready = bool(axial.get("availableForRealInference"))
    return {
        "schemaVersion": "pfi.multiplanar-contract.v2",
        "status": "ready" if sagittal_ready else "degraded",
        "capabilities": {
            "supportedWorkspaceModes": ["sagittal_only", "axial_only", "dual_plane"],
            "supportedInferenceModes": ["contract", "mock", "real_baseline"],
            "executionPolicy": "strict_all_requested",
            "canonicalPublicCaseConvention": "camelCase",
            "freeMetadataAccepted": False,
        },
        "models": {
            "sagittal": plane_contract_v2("sagittal", "sagittal_spider", sagittal),
            "axial": plane_contract_v2("axial", "axial_t2_alkafri", axial),
        },
        "readiness": {
            "sagittal": sagittal_ready,
            "axial": axial_ready,
            "dual": sagittal_ready and axial_ready,
        },
        "threeD": {
            "enabled": False,
            "statuses": {
                "sagittal_only": "blocked_missing_axial",
                "axial_only": "blocked_missing_sagittal",
                "dual_plane": "pending_registered_reconstruction",
            },
        },
        "errors": [
            "INVALID_MULTIPLANAR_REQUEST",
            "NO_PLANE_REQUESTED",
            "INPUT_NOT_FOUND",
            "MODEL_NOT_FOUND",
            "MODEL_PLANE_MISMATCH",
            "MODEL_NOT_READY",
            "REAL_INFERENCE_FAILED",
            "UNSUPPORTED_INFERENCE_MODE",
            "CONTRACT_FALLBACK_DISABLED",
        ],
        "examples": {
            "sagittalOnly": {
                "caseId": "CASE-101",
                "inferenceMode": "real_baseline",
                "allowContractFallback": False,
                "planes": {"sagittal": {"inputId": "inp_...", "modelKey": "sagittal_spider"}, "axial": None},
                "options": {"sliceIndex": None, "sliceAxis": None, "sliceWindowRadius": 3, "inputOrientationTransform": None},
            },
            "axialOnlyBlocked": {
                "caseId": "CASE-101",
                "inferenceMode": "real_baseline",
                "planes": {"sagittal": None, "axial": {"inputId": "inp_...", "modelKey": "axial_t2_alkafri"}},
            },
            "dualBlockedByAxial": {
                "caseId": "CASE-101",
                "inferenceMode": "real_baseline",
                "planes": {
                    "sagittal": {"inputId": "inp_sag_...", "modelKey": "sagittal_spider"},
                    "axial": {"inputId": "inp_ax_...", "modelKey": "axial_t2_alkafri"},
                },
            },
            "noPlanesError": {"caseId": "CASE-101", "planes": {"sagittal": None, "axial": None}},
        },
        "governance": {
            "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
            "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
            "deidentified": True,
            "diagnosisGenerated": False,
        },
    }


def plane_contract_v2(plane: str, model_key: str, status: Dict[str, Any]) -> Dict[str, Any]:
    manifest = status.get("manifest", {}) if isinstance(status, dict) else {}
    return {
        "plane": plane,
        "key": model_key,
        "version": status.get("version"),
        "readiness": status.get("readiness", "contract_only_missing_artifact"),
        "trainingStatus": status.get("trainingStatus") or manifest.get("trainingStatus"),
        "artifactHash": status.get("artifactHash"),
        "baselineReady": bool(status.get("baselineReady")),
        "availableForRealInference": bool(status.get("availableForRealInference")),
        "manifestStatus": manifest.get("status"),
        "manifestValid": bool(manifest.get("valid")),
    }

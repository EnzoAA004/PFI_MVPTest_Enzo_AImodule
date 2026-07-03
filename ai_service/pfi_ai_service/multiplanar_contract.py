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

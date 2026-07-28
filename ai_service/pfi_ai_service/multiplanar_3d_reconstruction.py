from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .asset_registry import AssetRegistryError, register_workspace_asset, resolve_run_asset, workspace_asset_path
from .reporting import write_json
from .multiplanar_v2_models import PlaneAssetV2, PlaneRunV2Result, ThreeDStatusV2


@dataclass(frozen=True)
class ReconstructionInput:
    plane: str
    run_id: str
    mask: np.ndarray
    spacing_mm: tuple[float, float] | None
    selected_slice_index: int | None
    slice_count: int | None
    selected_axis: int | None
    model_key: str
    model_version: str | None
    artifact_hash: str | None
    class_ids_by_key: dict[str, int]


def build_lumbar_3d_status(run_id: str, planes: dict[str, PlaneRunV2Result | None]) -> ThreeDStatusV2:
    source_ids = {
        "sagittal": planes.get("sagittal").runId if planes.get("sagittal") else None,
        "axial": planes.get("axial").runId if planes.get("axial") else None,
    }
    required = ["sagittal_masks", "axial_masks", "explicit_anatomical_mapping", "real_baseline_both_planes"]
    missing = missing_reconstruction_requirements(planes)
    if missing:
        status = "blocked_missing_axial" if "axial_plane" in missing else "blocked_missing_sagittal" if "sagittal_plane" in missing else "experimental_blocked_insufficient_geometry"
        return ThreeDStatusV2(
            enabled=False,
            status=status,  # type: ignore[arg-type]
            sourcePlaneRunIds=source_ids,  # type: ignore[arg-type]
            requiredInputs=required,
            reconstruction={
                "experimental": True,
                "available": False,
                "blockedReasons": missing,
                "kind": "experimental_geometric_proxy",
                "method": "dual_plane_bbox_proxy",
                "anatomicalReconstruction": False,
                "volumetricReconstruction": False,
            },
            warnings=["No se genera proxy 3D si faltan mascaras reales, trazabilidad de modelo o un mapping anatomico explicito."],
        )

    anatomical_mapping = load_explicit_anatomical_mapping()
    if not anatomical_mapping:
        return blocked_missing_mapping(source_ids, required)

    try:
        sagittal = reconstruction_input(planes["sagittal"])
        axial = reconstruction_input(planes["axial"])
        mesh = build_sparse_lumbar_proxy(run_id, sagittal, axial, anatomical_mapping)
    except (AssetRegistryError, OSError, ValueError) as exc:
        status = "experimental_blocked_missing_anatomical_mapping" if "mapping" in str(exc).lower() else "experimental_blocked_insufficient_geometry"
        return ThreeDStatusV2(
            enabled=False,
            status=status,  # type: ignore[arg-type]
            sourcePlaneRunIds=source_ids,  # type: ignore[arg-type]
            requiredInputs=required,
            reconstruction={
                "experimental": True,
                "available": False,
                "blockedReasons": [type(exc).__name__],
                "kind": "experimental_geometric_proxy",
                "method": "dual_plane_bbox_proxy",
                "anatomicalReconstruction": False,
                "volumetricReconstruction": False,
            },
            warnings=["El proxy 3D experimental fallo sin degradar a una geometria sintetica ni inferir equivalencias anatomicas."],
        )

    mesh_path = workspace_asset_path(run_id, "lumbar-3d-mesh.json")
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(mesh_path, mesh)
    register_workspace_asset(run_id, "lumbar-3d-mesh.json", mesh_path)
    return ThreeDStatusV2(
        enabled=True,
        status="experimental_ready",
        sourcePlaneRunIds=source_ids,  # type: ignore[arg-type]
        requiredInputs=required,
        assets=[
            PlaneAssetV2(
                assetName="lumbar-3d-mesh.json",
                role="mesh_3d",
                contentType="application/json",
                generated=True,
                relativePath=f"/assets/{run_id}/workspace/lumbar-3d-mesh.json",
            )
        ],
        reconstruction={
            "experimental": True,
            "available": True,
            "kind": "experimental_geometric_proxy",
            "method": "dual_plane_bbox_proxy",
            "anatomicalReconstruction": False,
            "volumetricReconstruction": False,
            "coordinateSystem": "local_proxy_space",
            "source": "real_segmentation_masks",
            "meshFormat": "pfi.lumbar-geometric-proxy.v1",
            "structureCount": len(mesh["structures"]),
            "vertexCount": len(mesh["vertices"]),
            "faceCount": len(mesh["faces"]),
            "traceability": mesh["traceability"],
            "mappingSource": "config",
            "mappingValidated": False,
            "explicitOperatorProvidedMapping": mesh["traceability"]["parameters"]["explicitOperatorProvidedMapping"],
            "limitations": mesh["limitations"],
        },
        warnings=[
            "Proxy geometrico experimental: no es reconstruccion anatomica 3D final.",
            "No hay reconstruccion volumetrica sin stack completo, geometria DICOM y registracion validada.",
        ],
    )


def blocked_missing_mapping(source_ids: dict[str, str | None], required: list[str]) -> ThreeDStatusV2:
    return ThreeDStatusV2(
        enabled=False,
        status="experimental_blocked_missing_anatomical_mapping",
        sourcePlaneRunIds=source_ids,  # type: ignore[arg-type]
        requiredInputs=required,
        assets=[],
        reconstruction={
            "experimental": True,
            "available": False,
            "blockedReasons": ["missing_explicit_anatomical_mapping"],
            "kind": "experimental_geometric_proxy",
            "method": "dual_plane_bbox_proxy",
            "anatomicalReconstruction": False,
            "volumetricReconstruction": False,
        },
        warnings=["No se cruzan IDs numericos sagital/axial; se requiere mapping anatomico explicito validado."],
    )


def missing_reconstruction_requirements(planes: dict[str, PlaneRunV2Result | None]) -> list[str]:
    missing: list[str] = []
    sagittal = planes.get("sagittal")
    axial = planes.get("axial")
    if sagittal is None:
        missing.append("sagittal_plane")
    if axial is None:
        missing.append("axial_plane")
    for plane_name, plane in (("sagittal", sagittal), ("axial", axial)):
        if plane is None:
            continue
        if plane.effectiveInferenceMode != "real_baseline" or plane.synthetic or plane.fallbackReason is not None:
            missing.append(f"{plane_name}_not_real_baseline")
        if not plane.model.availableForRealInference:
            missing.append(f"{plane_name}_not_available_for_real_inference")
        if not plane.model.artifactHash:
            missing.append(f"{plane_name}_artifact_hash")
        if not plane.model.manifestValid:
            missing.append(f"{plane_name}_manifest_valid")
        if not any(asset.assetName == "mask.npy" and asset.generated for asset in plane.assets):
            missing.append(f"{plane_name}_mask_asset")
        if plane.quality.maskCount <= 0:
            missing.append(f"{plane_name}_empty_masks")
    return missing


def reconstruction_input(plane: PlaneRunV2Result | None) -> ReconstructionInput:
    if plane is None:
        raise ValueError("missing plane")
    record = resolve_run_asset(plane.runId, plane.plane, "mask.npy")
    mask = np.asarray(np.load(record.path), dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"mask ndim unsupported: {mask.ndim}")
    spacing = plane.input.inPlaneSpacingMm if plane.input.inPlaneSpacingMm and len(plane.input.inPlaneSpacingMm) >= 2 else None
    class_ids_by_key = {mask.classKey: int(mask.classId) for mask in plane.masks if mask.classId is not None}
    return ReconstructionInput(
        plane=plane.plane,
        run_id=plane.runId,
        mask=mask,
        spacing_mm=(float(spacing[0]), float(spacing[1])) if spacing else None,
        selected_slice_index=int(plane.input.selectedSliceIndex) if plane.input.selectedSliceIndex is not None else None,
        slice_count=int(plane.input.sliceCount) if plane.input.sliceCount is not None else None,
        selected_axis=int(plane.input.selectedAxis) if plane.input.selectedAxis is not None else None,
        model_key=plane.model.key,
        model_version=plane.model.version,
        artifact_hash=plane.model.artifactHash,
        class_ids_by_key=class_ids_by_key,
    )


def load_explicit_anatomical_mapping() -> dict[str, list[str]]:
    raw = os.getenv("PFI_MULTIPLANAR_3D_ANATOMICAL_MAPPING_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    mapping: dict[str, list[str]] = {}
    seen_axial_keys: set[str] = set()
    for sagittal_key, axial_keys in parsed.items():
        if not isinstance(sagittal_key, str) or not sagittal_key.strip() or background_key(sagittal_key):
            return {}
        if not isinstance(axial_keys, list) or not axial_keys:
            return {}
        clean_axial_keys: list[str] = []
        for item in axial_keys:
            if not isinstance(item, str) or not item.strip() or background_key(item):
                return {}
            normalized = item.strip()
            if normalized in clean_axial_keys or normalized in seen_axial_keys:
                return {}
            clean_axial_keys.append(normalized)
            seen_axial_keys.add(normalized)
        mapping[sagittal_key.strip()] = clean_axial_keys
    return mapping


def background_key(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"background", "background_250", "raw_250", "class_0"}


def build_sparse_lumbar_proxy(
    run_id: str,
    sagittal: ReconstructionInput,
    axial: ReconstructionInput,
    anatomical_mapping: dict[str, list[str]],
) -> dict[str, Any]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    structures: list[dict[str, Any]] = []
    used_mapping: dict[str, list[str]] = {}
    unknown_sagittal = [key for key in anatomical_mapping if key not in sagittal.class_ids_by_key]
    unknown_axial = [
        key
        for axial_keys in anatomical_mapping.values()
        for key in axial_keys
        if key not in axial.class_ids_by_key
    ]
    if unknown_sagittal or unknown_axial:
        raise ValueError("missing valid anatomical mapping")
    for sagittal_key, axial_keys in anatomical_mapping.items():
        sagittal_id = sagittal.class_ids_by_key.get(sagittal_key)
        axial_ids = [axial.class_ids_by_key[key] for key in axial_keys if key in axial.class_ids_by_key]
        if sagittal_id is None or not axial_ids:
            continue
        sag_box = normalized_box(sagittal.mask == sagittal_id)
        ax_box = normalized_box(np.isin(axial.mask, axial_ids))
        if sag_box is None or ax_box is None:
            continue
        start = len(vertices)
        x_min, x_max = ax_box["xMin"], ax_box["xMax"]
        y_min, y_max = sag_box["xMin"], sag_box["xMax"]
        z_min = min(sag_box["yMin"], ax_box["yMin"])
        z_max = max(sag_box["yMax"], ax_box["yMax"])
        vertices.extend([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ])
        faces.extend([
            [start, start + 1, start + 2],
            [start, start + 2, start + 3],
            [start + 4, start + 6, start + 5],
            [start + 4, start + 7, start + 6],
            [start, start + 4, start + 5],
            [start, start + 5, start + 1],
            [start + 1, start + 5, start + 6],
            [start + 1, start + 6, start + 2],
            [start + 2, start + 6, start + 7],
            [start + 2, start + 7, start + 3],
            [start + 3, start + 7, start + 4],
            [start + 3, start + 4, start],
        ])
        structures.append({
            "label": sagittal_key,
            "sagittalClassId": sagittal_id,
            "axialClassIds": axial_ids,
            "mappedAxialClassKeys": axial_keys,
            "sourceMasks": {"sagittal": sagittal.run_id, "axial": axial.run_id},
            "vertexStart": start,
            "vertexCount": 8,
            "faceStart": len(faces) - 12,
            "faceCount": 12,
        })
        used_mapping[sagittal_key] = axial_keys
    if not structures:
        raise ValueError("missing valid anatomical mapping")
    return {
        "schemaVersion": "pfi.lumbar-geometric-proxy.v1",
        "runId": run_id,
        "experimental": True,
        "kind": "experimental_geometric_proxy",
        "method": "dual_plane_bbox_proxy",
        "anatomicalReconstruction": False,
        "volumetricReconstruction": False,
        "coordinateSystem": "local_proxy_space",
        "units": "normalized",
        "vertices": vertices,
        "faces": faces,
        "structures": structures,
        "traceability": {
            "sourcePlaneRunIds": {"sagittal": sagittal.run_id, "axial": axial.run_id},
            "models": {
                "sagittal": model_trace(sagittal),
                "axial": model_trace(axial),
            },
            "transforms": {
                "sagittal": slice_trace(sagittal),
                "axial": slice_trace(axial),
            },
            "parameters": {
                "method": "dual_plane_bbox_proxy",
                "mappingSource": "config",
                "mappingValidated": False,
                "explicitOperatorProvidedMapping": used_mapping,
            },
            "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
            "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
        },
        "limitations": [
            "Proxy geometrico experimental derivado de bounding boxes 2D por plano.",
            "No es reconstruccion anatomica real ni mesh paciente-especifico validado.",
            "No reemplaza una reconstruccion volumetrica con stack completo, orden DICOM, ImagePositionPatient, ImageOrientationPatient, FrameOfReferenceUID, spacing entre cortes y registracion validada.",
            "Debe validarse con evidencia E2E reproducible antes de uso clinico.",
        ],
    }


def physical_box(mask: np.ndarray, spacing_mm: tuple[float, float]) -> dict[str, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    row_spacing, col_spacing = spacing_mm
    return {
        "xMinMm": round(float(xs.min()) * col_spacing, 4),
        "xMaxMm": round(float(xs.max() + 1) * col_spacing, 4),
        "yMinMm": round(float(ys.min()) * row_spacing, 4),
        "yMaxMm": round(float(ys.max() + 1) * row_spacing, 4),
    }


def normalized_box(mask: np.ndarray) -> dict[str, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    return {
        "xMin": round(float(xs.min()) / max(width - 1, 1), 6),
        "xMax": round(float(xs.max() + 1) / max(width, 1), 6),
        "yMin": round(float(ys.min()) / max(height - 1, 1), 6),
        "yMax": round(float(ys.max() + 1) / max(height, 1), 6),
    }


def model_trace(item: ReconstructionInput) -> dict[str, Any]:
    return {
        "modelKey": item.model_key,
        "modelVersion": item.model_version,
        "artifactHash": item.artifact_hash,
        "runId": item.run_id,
    }


def slice_trace(item: ReconstructionInput) -> dict[str, Any]:
    metadata_available = any(
        value is not None
        for value in (item.selected_slice_index, item.slice_count, item.selected_axis, item.spacing_mm)
    )
    return {
        "selectedSliceIndex": item.selected_slice_index,
        "sliceCount": item.slice_count,
        "selectedAxis": item.selected_axis,
        "inPlaneSpacingMm": list(item.spacing_mm) if item.spacing_mm is not None else None,
        "metadataSource": "runtime_input_metadata" if metadata_available else "unavailable",
        "depthSpacingMm": None,
        "depthSource": "not_available_for_proxy",
    }

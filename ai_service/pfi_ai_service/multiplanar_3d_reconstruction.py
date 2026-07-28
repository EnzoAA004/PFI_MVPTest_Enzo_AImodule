from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .agent_policy import HUMAN_REVIEW_REQUIRED, NOT_CLINICAL_DIAGNOSIS
from .asset_registry import AssetRegistryError, register_workspace_asset, resolve_run_asset
from .reporting import write_json
from .settings import get_settings
from .multiplanar_v2_models import PlaneAssetV2, PlaneRunV2Result, ThreeDStatusV2


@dataclass(frozen=True)
class ReconstructionInput:
    plane: str
    run_id: str
    mask: np.ndarray
    spacing_mm: tuple[float, float]
    selected_slice_index: int
    slice_count: int
    selected_axis: int
    model_key: str
    model_version: str | None
    artifact_hash: str | None


def build_lumbar_3d_status(run_id: str, planes: dict[str, PlaneRunV2Result | None]) -> ThreeDStatusV2:
    source_ids = {
        "sagittal": planes.get("sagittal").runId if planes.get("sagittal") else None,
        "axial": planes.get("axial").runId if planes.get("axial") else None,
    }
    required = ["sagittal_masks", "axial_masks", "physical_spacing", "slice_index_mapping", "real_baseline_both_planes"]
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
                "method": "dual_plane_mask_geometry",
            },
            warnings=["No se genera 3D real si faltan mascaras reales, spacing o mapeo de slices."],
        )

    try:
        sagittal = reconstruction_input(planes["sagittal"])
        axial = reconstruction_input(planes["axial"])
        mesh = build_sparse_lumbar_mesh(run_id, sagittal, axial)
    except (AssetRegistryError, OSError, ValueError) as exc:
        return ThreeDStatusV2(
            enabled=False,
            status="experimental_blocked_insufficient_geometry",
            sourcePlaneRunIds=source_ids,  # type: ignore[arg-type]
            requiredInputs=required,
            reconstruction={
                "experimental": True,
                "available": False,
                "blockedReasons": [type(exc).__name__],
                "method": "dual_plane_mask_geometry",
            },
            warnings=["La reconstruccion 3D experimental fallo sin degradar a una geometria sintetica."],
        )

    output_dir = get_settings().output_dir / "multiplanar_3d" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "lumbar-3d-mesh.json"
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
            "method": "dual_plane_mask_geometry",
            "coordinateSystem": "patient_relative_mm",
            "source": "real_segmentation_masks",
            "meshFormat": "pfi.lumbar-sparse-mesh.v1",
            "structureCount": len(mesh["structures"]),
            "vertexCount": len(mesh["vertices"]),
            "faceCount": len(mesh["faces"]),
            "traceability": mesh["traceability"],
            "limitations": mesh["limitations"],
        },
        warnings=[
            "Reconstruccion experimental: requiere validacion E2E antes de presentarse como 3D clinico.",
            "No es una extrusion 2D aislada: fusiona mascaras reales sagitales y axiales con spacing y slice mapping disponibles.",
        ],
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
        if not plane.input.inPlaneSpacingMm:
            missing.append(f"{plane_name}_spacing")
        if plane.input.selectedSliceIndex is None or plane.input.sliceCount is None or plane.input.selectedAxis is None:
            missing.append(f"{plane_name}_slice_index_mapping")
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
    spacing = plane.input.inPlaneSpacingMm
    if spacing is None or len(spacing) < 2:
        raise ValueError("missing in-plane spacing")
    if plane.input.selectedSliceIndex is None or plane.input.sliceCount is None or plane.input.selectedAxis is None:
        raise ValueError("missing slice mapping")
    return ReconstructionInput(
        plane=plane.plane,
        run_id=plane.runId,
        mask=mask,
        spacing_mm=(float(spacing[0]), float(spacing[1])),
        selected_slice_index=int(plane.input.selectedSliceIndex),
        slice_count=int(plane.input.sliceCount),
        selected_axis=int(plane.input.selectedAxis),
        model_key=plane.model.key,
        model_version=plane.model.version,
        artifact_hash=plane.model.artifactHash,
    )


def build_sparse_lumbar_mesh(run_id: str, sagittal: ReconstructionInput, axial: ReconstructionInput) -> dict[str, Any]:
    labels = [int(value) for value in sorted((set(np.unique(sagittal.mask).astype(int)) & set(np.unique(axial.mask).astype(int))) - {0})]
    if not labels:
        raise ValueError("no shared foreground classes")
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    structures: list[dict[str, Any]] = []
    for label in labels:
        sag_box = physical_box(sagittal.mask == label, sagittal.spacing_mm)
        ax_box = physical_box(axial.mask == label, axial.spacing_mm)
        if sag_box is None or ax_box is None:
            continue
        start = len(vertices)
        x_min, x_max = ax_box["xMinMm"], ax_box["xMaxMm"]
        y_min, y_max = sagittal_slice_span_mm(sagittal)
        z_min = min(sag_box["yMinMm"], ax_box["yMinMm"])
        z_max = max(sag_box["yMaxMm"], ax_box["yMaxMm"])
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
            "label": f"class_{label}",
            "classId": label,
            "sourceMasks": {"sagittal": sagittal.run_id, "axial": axial.run_id},
            "vertexStart": start,
            "vertexCount": 8,
            "faceStart": len(faces) - 12,
            "faceCount": 12,
        })
    if not structures:
        raise ValueError("empty reconstructed structures")
    return {
        "schemaVersion": "pfi.lumbar-sparse-mesh.v1",
        "runId": run_id,
        "experimental": True,
        "coordinateSystem": "patient_relative_mm",
        "units": "mm",
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
            "parameters": {"method": "dual_plane_mask_geometry", "sharedForegroundClasses": labels},
            "humanReviewRequired": HUMAN_REVIEW_REQUIRED,
            "notClinicalDiagnosis": NOT_CLINICAL_DIAGNOSIS,
        },
        "limitations": [
            "Artefacto experimental derivado de mascaras reales 2D por plano.",
            "No reemplaza una reconstruccion volumetrica con stack completo y registracion DICOM validada.",
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


def sagittal_slice_span_mm(sagittal: ReconstructionInput) -> tuple[float, float]:
    if sagittal.slice_count <= 1:
        return -0.5, 0.5
    center = (sagittal.slice_count - 1) / 2.0
    offset = float(sagittal.selected_slice_index) - center
    thickness = max(float(sum(sagittal.spacing_mm) / len(sagittal.spacing_mm)), 1.0)
    y_center = offset * thickness
    return round(y_center - thickness / 2.0, 4), round(y_center + thickness / 2.0, 4)


def model_trace(item: ReconstructionInput) -> dict[str, Any]:
    return {
        "modelKey": item.model_key,
        "modelVersion": item.model_version,
        "artifactHash": item.artifact_hash,
        "runId": item.run_id,
    }


def slice_trace(item: ReconstructionInput) -> dict[str, Any]:
    return {
        "selectedSliceIndex": item.selected_slice_index,
        "sliceCount": item.slice_count,
        "selectedAxis": item.selected_axis,
        "inPlaneSpacingMm": list(item.spacing_mm),
    }

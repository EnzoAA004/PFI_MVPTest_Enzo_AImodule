from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

from .quality import parse_list_like, to_float


EXPECTED_MODEL_BY_PLANE = {
    "axial": "axial_t2_alkafri",
    "sagittal": "sagittal_spider",
}


def expected_model_for_plane(plane: str) -> str:
    return EXPECTED_MODEL_BY_PLANE.get(str(plane).strip().lower(), "unknown")


def evaluate_agent_item(row: Dict[str, Any]) -> Dict[str, Any]:
    plane = str(row.get("plane", "")).strip().lower()
    model_key = str(row.get("model_key", "")).strip()
    expected_model = expected_model_for_plane(plane)

    flags = parse_list_like(row.get("flags", []))
    n_components = int(to_float(row.get("n_components"), 0) or 0)
    mean_fg_conf = to_float(row.get("mean_fg_confidence"), None)
    fg_ratio = to_float(row.get("foreground_ratio"), None)

    reasons: List[str] = []

    if expected_model != "unknown" and model_key != expected_model:
        reasons.append(f"modelo inesperado: esperado={expected_model}, recibido={model_key}")

    for flag in flags:
        reasons.append(f"flag E13: {flag}")

    if plane == "axial" and n_components > 10:
        reasons.append("muchos componentes en mascara axial")

    if plane == "sagittal" and n_components > 10:
        reasons.append("componentes multiples; revisar continuidad anatomica")

    if mean_fg_conf is not None and mean_fg_conf < 0.75:
        reasons.append("confianza baja en foreground")

    if fg_ratio is not None and fg_ratio < 0.002:
        reasons.append("foreground muy bajo")
    if fg_ratio is not None and plane == "axial" and fg_ratio > 0.25:
        reasons.append("foreground axial alto")
    if fg_ratio is not None and plane == "sagittal" and fg_ratio > 0.45:
        reasons.append("foreground sagital alto")

    unique_reasons = []
    for r in reasons:
        if r not in unique_reasons:
            unique_reasons.append(r)

    high_priority = (
        any("modelo inesperado" in r for r in unique_reasons)
        or any("confianza baja en foreground" in r for r in unique_reasons)
        or any("foreground muy bajo" in r for r in unique_reasons)
    )

    medium_priority = bool(unique_reasons)

    if high_priority:
        status = "requiere_revision_prioritaria"
        priority = "alta"
        action = "Revisar overlay y considerar repetir preprocesamiento/inferencia antes de usar el resultado."
    elif medium_priority:
        status = "requiere_revision_con_atencion"
        priority = "media"
        action = "Revisar visualmente el overlay; si la anatomia es coherente, puede pasar a validacion profesional."
    else:
        status = "listo_para_revision_estandar"
        priority = "baja"
        action = "Enviar a revision profesional estandar como resultado asistido por IA."

    return {
        "expected_model": expected_model,
        "agent_status": status,
        "review_priority": priority,
        "agent_reasons": unique_reasons,
        "recommended_action": action,
        "human_review_required": True,
    }


def build_agent_decisions(worklist_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in worklist_df.iterrows():
        base = row.to_dict()
        decision = evaluate_agent_item(base)
        rows.append({**base, **decision})
    return pd.DataFrame(rows)


def summarize_agent_decisions(decisions_df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_items": int(len(decisions_df)),
        "plane_distribution": decisions_df["plane"].value_counts().to_dict() if "plane" in decisions_df else {},
        "priority_distribution": decisions_df["review_priority"].value_counts().to_dict() if "review_priority" in decisions_df else {},
        "status_distribution": decisions_df["agent_status"].value_counts().to_dict() if "agent_status" in decisions_df else {},
    }

    if "mean_fg_confidence" in decisions_df:
        summary["mean_fg_confidence"] = float(pd.to_numeric(decisions_df["mean_fg_confidence"], errors="coerce").mean())

    if "dice_macro_useful_classes" in decisions_df:
        summary["mean_dice_macro_useful_classes"] = float(
            pd.to_numeric(decisions_df["dice_macro_useful_classes"], errors="coerce").mean()
        )

    return summary
